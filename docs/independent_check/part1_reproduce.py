"""
Independent re-analysis (written from scratch; does not import experiments/cancer_age.py).
Only the data-filter definitions are taken from docs/CANCER_AGE_PREREG.md.
Run: python docs/independent_check/part1_reproduce.py   -> part1_result.json
"""
import json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(777)


# ---------------------------------------------------------------- tools I wrote
def auc(y, s, w=None):
    y = np.asarray(y); s = np.asarray(s)
    if w is None:
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], np.asarray(w)[o]
    # weighted AUC with ties handled by grouping equal scores
    u, idx = np.unique(s, return_index=True)
    wp = np.add.reduceat(w * y, idx); wn = np.add.reduceat(w * (1 - y), idx)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def rcs(x, knots):
    """Harrell restricted cubic spline basis (k-2 nonlinear columns + linear)."""
    k = knots; t = lambda u: np.maximum(u, 0) ** 3
    cols = [x]
    for j in range(len(k) - 2):
        c = (t(x - k[j]) - t(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2]) + t(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2
        cols.append(c)
    return np.column_stack(cols)


def logit_fit(X, y, w=None, iters=50, ridge=1e-8):
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X1.shape[1])
    w = np.ones(len(y)) if w is None else w
    for _ in range(iters):
        eta = X1 @ b; p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        g = X1.T @ (w * (y - p)) - ridge * b
        H = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-9: break
    return b


def logit_pred(b, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ b, -30, 30)))


def cal_slope(y, p):
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    b = logit_fit(z.reshape(-1, 1), y.astype(float)); return float(b[1]), float(b[0])


# ---------------------------------------------------------------- data (filters per PREREG)
def load(months_rule="le", landmark_months=0):
    d = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    n0 = len(d)
    d = d[d.age.between(35, 84) & (d.prior_cancer == 0)]
    n1 = len(d)
    # complete 10-year follow-up: alive with >=120 months observed, OR died (any cause)
    d = d[(d.months >= 120) | (d.mortstat == 1)]
    n2 = len(d)
    cancer_death = (d.mortstat == 1) & (d.ucod == 2)
    within = (d.months <= 120) if months_rule == "le" else (d.months < 120)
    d = d.assign(y=(cancer_death & within).astype(int))
    if landmark_months:
        d = d[~((d.mortstat == 1) & (d.months < landmark_months))]
    nbefore = len(d)
    d = d.dropna(subset=["smoke_status", "bmi"]).copy()
    d.attrs["flow"] = dict(raw=n0, age_prior=n1, followup_ok=n2, before_missing=nbefore, final=len(d))
    d["bmi"] = d.bmi.clip(15, 55)
    d["quit"] = d.quit_years.fillna(0).clip(0, 50)
    d["cpd"] = d.cigs_per_day.fillna(0).clip(0, 60)
    return d


def X_of(d, model, knots):
    cols = [rcs(d.age.to_numpy(float), knots)]
    if model >= 1:
        s = d.smoke_status.to_numpy()
        fm, cu = (s == 1).astype(float), (s == 2).astype(float)
        cols.append(np.column_stack([fm, cu, cu * d.cpd.to_numpy() / 20, fm * d.quit.to_numpy() / 10]))
    if model >= 2:
        b = (d.bmi.to_numpy() - 27) / 5
        cols.append(np.column_stack([b, b ** 2]))
    return np.hstack(cols)


class Model:
    def __init__(self, tr, model):
        self.model = model; self.k = {}; self.b = {}
        for sx in (0.0, 1.0):
            s = tr[tr.male == sx]
            self.k[sx] = np.quantile(s.age, [0.05, 0.275, 0.5, 0.725, 0.95])
            self.b[sx] = logit_fit(X_of(s, model, self.k[sx]), s.y.to_numpy(float))

    def predict(self, d):
        p = np.zeros(len(d))
        for sx in (0.0, 1.0):
            m = (d.male == sx).to_numpy()
            if m.any(): p[m] = logit_pred(self.b[sx], X_of(d[m], self.model, self.k[sx]))
        return p


def boot_gain(y, a, b, n=400):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; g = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        g.append(auc(y[i], b[i]) - auc(y[i], a[i]))
    return [float(x) for x in np.percentile(g, [2.5, 97.5])]


def coverage_curve(y, key, k_list, tiesplit=True):
    """Share of all events among the top-k by key. Ties at the cutoff are split proportionally
    (exact expectation of a random tie-break), so age-only is not subject to tie-break noise."""
    o = np.argsort(-key, kind="mergesort"); ks = key[o]; ys = y[o]
    cum = np.concatenate([[0], np.cumsum(ys)]); out = []
    for k in k_list:
        if not tiesplit: out.append(cum[k] / y.sum()); continue
        v = ks[k - 1]
        lo = np.searchsorted(-ks, -v, side="left"); hi = np.searchsorted(-ks, -v, side="right")
        ev_tie = (cum[hi] - cum[lo]) * (k - lo) / (hi - lo)
        out.append((cum[lo] + ev_tie) / y.sum())
    return np.array(out)


def main():
    R = {}
    d = load(); R["flow"] = d.attrs["flow"]
    tr, te = d[d.year <= 2002].reset_index(drop=True), d[d.year >= 2005].reset_index(drop=True)
    R["n"] = dict(fit=len(tr), test=len(te), fit_events=int(tr.y.sum()), test_events=int(te.y.sum()),
                  year_overlap=sorted(set(tr.year) & set(te.year)), test_years=sorted(set(te.year)), fit_years=sorted(set(tr.year)))
    M = {m: Model(tr, m) for m in (0, 1, 2)}
    P = {m: M[m].predict(te) for m in M}; y = te.y.to_numpy()
    R["auc"] = {f"F{m}": auc(y, P[m]) for m in P}
    R["auc_gain_F2_vs_F0"] = dict(gain=R["auc"]["F2"] - R["auc"]["F0"], ci=boot_gain(y, P[0], P[2]))
    R["calib_F2"] = dict(zip(("slope", "intercept"), cal_slope(y, P[2])))
    R["calib_F0"] = dict(zip(("slope", "intercept"), cal_slope(y, P[0])))
    R["OE_overall_F2"] = float(y.sum() / P[2].sum())
    R["bands_F2"] = {}
    for lo, hi in ((35, 49), (50, 59), (60, 69), (70, 84)):
        m = te.age.between(lo, hi).to_numpy()
        R["bands_F2"][f"{lo}-{hi}"] = dict(n=int(m.sum()), obs=int(y[m].sum()), exp=float(P[2][m].sum()), OE=float(y[m].sum() / P[2][m].sum()))

    # ---- cost-matched invitation, 40-74
    m = te.age.between(40, 74).to_numpy(); ti = te[m].reset_index(drop=True); yi = ti.y.to_numpy(); n = len(ti)
    R["inv"] = dict(n=n, events=int(yi.sum()))
    p2, p0 = P[2][m], P[0][m]
    shares = [0.1, 0.2, 0.3, 0.4]; ks = [int(s * n) for s in shares]
    ages = ti.age.to_numpy(float)
    cov = {"age_only": coverage_curve(yi, ages, ks), "age_plus_sex_F0": coverage_curve(yi, p0, ks), "F2": coverage_curve(yi, p2, ks)}
    R["coverage"] = {"shares": shares, **{k: [float(x) for x in v] for k, v in cov.items()}}
    # bootstrap (people resampled; model fixed -> conditional on fitted model)
    B = {k: [] for k in ("F2_minus_age", "F2_minus_F0", "F0_minus_age")}
    for _ in range(400):
        i = rng.integers(0, n, n)
        a = coverage_curve(yi[i], ages[i], ks); f0 = coverage_curve(yi[i], p0[i], ks); f2 = coverage_curve(yi[i], p2[i], ks)
        B["F2_minus_age"].append(f2 - a); B["F2_minus_F0"].append(f2 - f0); B["F0_minus_age"].append(f0 - a)
    R["coverage_boot_ci"] = {k: {str(s): [float(np.percentile(np.array(v)[:, j], 2.5)), float(np.percentile(np.array(v)[:, j], 97.5))] for j, s in enumerate(shares)} for k, v in B.items()}
    # fixed target coverage
    full = {k: None for k in cov}
    keys = {"age_only": ages, "age_plus_sex_F0": p0, "F2": p2}
    allk = np.arange(1, n + 1, 50)
    for name, key in keys.items():
        c = coverage_curve(yi, key, allk); full[name] = c
    R["invites_for_target"] = {}
    for t in (0.4, 0.5, 0.6):
        R["invites_for_target"][str(t)] = {nm: float(allk[np.searchsorted(c, t)] / n) for nm, c in full.items()}

    # ---- sensitivity table: top-20% coverage, F2 vs oldest 20%
    def run(label, tr_, te_):
        mm = {k: Model(tr_, k) for k in (0, 2)}
        t = te_[te_.age.between(40, 74)].reset_index(drop=True); yy = t.y.to_numpy(); k = [int(0.2 * len(t))]
        pp0, pp2 = mm[0].predict(t), mm[2].predict(t)
        yall = te_.y.to_numpy()
        return dict(label=label, n_fit=len(tr_), n_test=len(te_), ev_test=int(yall.sum()),
                    auc_F0=auc(yall, mm[0].predict(te_)), auc_F2=auc(yall, mm[2].predict(te_)),
                    slope_F2=cal_slope(yall, mm[2].predict(te_))[0],
                    cov20_age=float(coverage_curve(yy, t.age.to_numpy(float), k)[0]),
                    cov20_F0=float(coverage_curve(yy, pp0, k)[0]), cov20_F2=float(coverage_curve(yy, pp2, k)[0]))
    S = [run("main", tr, te)]
    d2 = load("lt"); S.append(run("outcome months<120 (strict)", d2[d2.year <= 2002], d2[d2.year >= 2005]))
    d3 = load(landmark_months=24); S.append(run("landmark: drop deaths in first 24 months (reverse-causation check)", d3[d3.year <= 2002], d3[d3.year >= 2005]))
    S.append(run("fit 1997-1998 only (outcomes known by test baseline)", d[d.year <= 1998], te))
    S.append(run("fit on test years 2005-2009 -> test on 1997-2002 (reversed)", te, tr))
    S.append(run("test 2005-2007 only", tr, te[te.year <= 2007]))
    S.append(run("test 2008-2009 only", tr, te[te.year >= 2008]))
    R["sensitivity"] = S

    # ---- weighted check (survey weights; model unweighted fit, weighted evaluation)
    w = te.weight.to_numpy(float)
    R["weighted"] = dict(auc_F0=auc(y, P[0], w), auc_F2=auc(y, P[2], w))

    # ---- missingness: who is dropped for missing smoking/BMI?
    raw = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    raw = raw[raw.age.between(35, 84) & (raw.prior_cancer == 0) & ((raw.months >= 120) | (raw.mortstat == 1))]
    raw["y"] = ((raw.mortstat == 1) & (raw.ucod == 2) & (raw.months <= 120)).astype(int)
    raw["miss"] = raw.smoke_status.isna() | raw.bmi.isna()
    R["missing"] = dict(share_dropped=float(raw.miss.mean()), cancer_death_rate_dropped=float(raw[raw.miss].y.mean()),
                        rate_kept=float(raw[~raw.miss].y.mean()),
                        age_mean_dropped=float(raw[raw.miss].age.mean()), age_mean_kept=float(raw[~raw.miss].age.mean()),
                        missing_bmi_only=float(raw.bmi.isna().mean()))
    # ---- outcome coding facts
    full = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    full = full[full.age.between(35, 84) & (full.prior_cancer == 0)]
    R["followup_facts"] = dict(
        alive_with_lt120=int(((full.mortstat == 0) & (full.months < 120)).sum()),
        min_months_alive=float(full[full.mortstat == 0].months.min()),
        deaths_months_le0=int(((full.mortstat == 1) & (full.months <= 0)).sum()),
        cancer_deaths_after_120=int(((full.mortstat == 1) & (full.ucod == 2) & (full.months > 120)).sum()),
        other_death_in_10y=int(((full.mortstat == 1) & (full.ucod != 2) & (full.months <= 120)).sum()),
        cancer_death_in_10y=int(((full.mortstat == 1) & (full.ucod == 2) & (full.months <= 120)).sum()),
        ucod_missing_among_dead=int(((full.mortstat == 1) & full.ucod.isna()).sum()),
        year_2003_in_neither=bool(2003 not in set(tr.year) | set(te.year)))
    json.dump(R, open(os.path.join(HERE, "part1_result.json"), "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
