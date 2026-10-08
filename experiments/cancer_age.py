"""
Cancer-risk age and cost-matched screening invitation, from free information only.
Questions and bars: docs/CANCER_AGE_PREREG.md (committed before any model was fitted).

Fit on NHIS interview years 1997-2002, test on 2005-2009, external check in NHANES 1999-2008 and
NHANES III. Outcome: death from cancer within ten years.

Run:  python experiments/cancer_age.py       needs data/nhis_cancer_risk.csv.gz
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cancer_age_result.json")
RNG = np.random.default_rng(20261010)
CANCER = 2
BANDS = [(35, 49), (50, 59), (60, 69), (70, 84)]


# ------------------------------------------------------------------ data
def load_nhis():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    d = d[(d["age"] >= 35) & (d["age"] <= 84) & (d["prior_cancer"] == 0) & (d["year"] <= 2009)].copy()
    d = d[(d["months"] >= 120) | (d["mortstat"] == 1)]
    d["y"] = ((d["mortstat"] == 1) & (d["ucod"] == CANCER) & (d["months"] <= 120)).astype(int)
    e = d["educ_raw"]
    d["educ"] = np.select([e.between(0, 12), e.between(13, 14), e.between(15, 17), e.between(18, 21)], [0, 1, 2, 3], default=-1)
    h, r = d["hisp_raw"], d["race_raw"]
    hisp = h.between(1, 11)
    nh = h == 12
    d["race4"] = np.select([hisp, nh & (r == 1), nh & (r == 2), nh], [2, 0, 1, 3], default=-1)   # 0 NH White, 1 NH Black, 2 Hispanic, 3 other, -1 unknown
    d = d.dropna(subset=["smoke_status", "bmi"]).reset_index(drop=True)
    d["bmi"] = d["bmi"].clip(15, 55)
    d["quit_years"] = d["quit_years"].fillna(0).clip(0, 50)
    d["cigs_per_day"] = d["cigs_per_day"].fillna(0).clip(0, 60)
    return d


def spline_basis(age_fit):
    st = SplineTransformer(degree=3, n_knots=5, knots="quantile", extrapolation="constant", include_bias=False)
    st.fit(np.asarray(age_fit).reshape(-1, 1))
    return st


def design(d, model, st, cats):
    cols = [st.transform(d[["age"]].to_numpy())]
    if model >= 1:
        s = d["smoke_status"].to_numpy()
        cols += [np.column_stack([(s == 1).astype(float), (s == 2).astype(float), d["cigs_per_day"] / 20.0, (s == 1) * d["quit_years"] / 10.0, (s == 1) * (d["quit_years"] / 10.0) ** 2])]
    if model >= 2:
        b = d["bmi"].to_numpy()
        cols += [np.column_stack([(b < 18.5), (b >= 25) & (b < 30), (b >= 30) & (b < 35), b >= 35]).astype(float)]
    if model >= 3:
        e = d["educ"].to_numpy()
        v = d["vigorous_per_week"].fillna(0).to_numpy()
        cols += [np.column_stack([(e == 0), (e == 1), (e == 2), (e == -1), d["married"].fillna(0).to_numpy() == 1, d["self_health"].fillna(cats["sh_med"]).to_numpy() - 3.0,
                                  (v >= 1) & (v < 3), v >= 3]).astype(float)]
    if model >= 4:
        rc = d["race4"].to_numpy()
        cols += [np.column_stack([(rc == 1), (rc == 2), (rc == 3), (rc == -1)]).astype(float)]
    return np.hstack(cols)


class Fit:
    """One logistic model per sex, with the age spline fitted on the fitting years."""

    def __init__(self, d, model):
        self.model, self.cats = model, {"sh_med": float(d["self_health"].median())}
        self.st, self.m = {}, {}
        for sx in (0.0, 1.0):
            s = d[d["male"] == sx]
            self.st[sx] = spline_basis(s["age"])
            self.m[sx] = LogisticRegression(C=1e6, max_iter=5000).fit(design(s, model, self.st[sx], self.cats), s["y"])

    def predict(self, d):
        p = np.zeros(len(d))
        for sx in (0.0, 1.0):
            msk = (d["male"] == sx).to_numpy()
            if msk.any():
                p[msk] = self.m[sx].predict_proba(design(d[msk], self.model, self.st[sx], self.cats))[:, 1]
        return p


# ------------------------------------------------------------------ helpers
def slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    m = LogisticRegression(C=1e9, max_iter=2000).fit(z.reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def boot_auc_gain(y, a, b, n=500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    g = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        g.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return [round(float(x), 4) for x in np.percentile(g, [2.5, 97.5])]


def bands(df, p, y):
    out, ok_all = {}, True
    for lo, hi in BANDS:
        m = ((df["age"] >= lo) & (df["age"] <= hi)).to_numpy()
        obs, exp = int(y[m].sum()), float(p[m].sum())
        pl, ph = poisson.ppf(0.025, exp), poisson.ppf(0.975, exp)
        ok = bool(abs(obs - exp) <= 0.2 * exp or pl <= obs <= ph)
        out[f"{lo}-{hi}"] = {"n": int(m.sum()), "observed": obs, "expected": round(exp, 1), "ratio": round(obs / exp, 3) if exp else None, "ok": ok}
        ok_all &= ok
    return out, ok_all


def risk_age(p_person, sex, ages_grid, r0):
    """Age at which the age-and-sex curve reaches this predicted risk."""
    out = np.zeros(len(p_person))
    for sx in (0.0, 1.0):
        m = sex == sx
        curve = np.maximum.accumulate(r0[sx])
        out[m] = np.interp(p_person[m], curve, ages_grid)
    return out


def coverage(y, order_key, ks):
    order = np.argsort(-order_key, kind="stable")
    cum = np.cumsum(y[order])
    return np.array([cum[k - 1] / y.sum() for k in ks])


def main():
    d = load_nhis()
    tr, te = d[d["year"] <= 2002].reset_index(drop=True), d[d["year"] >= 2005].reset_index(drop=True)
    print(f"NHIS adults 35-84, no prior cancer: fit {len(tr):,} (1997-2002, {int(tr['y'].sum())} cancer deaths), test {len(te):,} (2005-2009, {int(te['y'].sum())} cancer deaths)")
    res = {"n_fit": int(len(tr)), "n_test": int(len(te)), "cancer_fit": int(tr["y"].sum()), "cancer_test": int(te["y"].sum()), "models": {}}
    fits = {m: Fit(tr, m) for m in range(5)}
    yte = te["y"].to_numpy()
    P = {m: fits[m].predict(te) for m in range(5)}
    names = {0: "F0 age + sex", 1: "F1 + smoking", 2: "F2 + BMI", 3: "F3 + schooling, marital, health, activity", 4: "F4 + race and ethnicity"}
    print("\nAUC in the temporal test (2005-2009):")
    for m in range(5):
        s, i = slope_intercept(yte, P[m])
        res["models"][names[m]] = {"auc": round(float(roc_auc_score(yte, P[m])), 4), "slope": round(s, 3), "intercept": round(i, 3)}
        print(f"  {names[m]:<44} AUC {res['models'][names[m]]['auc']:.4f}  slope {s:.3f}  intercept {i:.3f}")
    gains = {}
    for a, b, lab in ((0, 2, "F2 vs F0"), (2, 3, "F3 vs F2"), (3, 4, "F4 vs F3"), (1, 2, "F2 vs F1")):
        g = float(roc_auc_score(yte, P[b]) - roc_auc_score(yte, P[a]))
        gains[lab] = {"gain": round(g, 4), "ci": boot_auc_gain(yte, P[a], P[b])}
        print(f"  gain {lab}: {g:+.4f} {gains[lab]['ci']}")
    res["gains"] = gains
    chosen = 3 if (gains["F3 vs F2"]["gain"] >= 0.01 and gains["F3 vs F2"]["ci"][0] > 0) else 2
    res["chosen_model"] = names[chosen]
    print(f"cheapest model that suffices (rule fixed in advance): {names[chosen]}")
    pc = P[chosen]

    # bars 1 to 3
    s, i = slope_intercept(yte, pc)
    b1 = 0.8 <= s <= 1.2 and -0.3 <= i <= 0.3
    bd, b2 = bands(te, pc, yte)
    b3 = gains["F2 vs F0"]["gain"] >= 0.02 and gains["F2 vs F0"]["ci"][0] > 0
    res.update({"bar1": {"slope": round(s, 3), "intercept": round(i, 3), "passes": bool(b1)}, "bar2": {"bands": bd, "passes": bool(b2)}, "bar3": {**gains["F2 vs F0"], "passes": bool(b3)}})
    print(f"\nBAR 1 calibration slope {s:.3f}, intercept {i:.3f}: {'MET' if b1 else 'MISSED'}")
    print("BAR 2 calibration by age band:", "  ".join(f"{k} {v['observed']}/{v['expected']}{'' if v['ok'] else ' MISS'}" for k, v in bd.items()), "->", "MET" if b2 else "MISSED")
    print(f"BAR 3 F2 over F0: {gains['F2 vs F0']['gain']:+.4f} {gains['F2 vs F0']['ci']}: {'MET' if b3 else 'MISSED'}")

    # bar 4 fairness
    f4 = {}
    ok4 = True
    for code, lab in ((0, "non-Hispanic White"), (1, "non-Hispanic Black"), (2, "Hispanic"), (3, "other")):
        m = (te["race4"] == code).to_numpy()
        if int(yte[m].sum()) < 30:
            f4[lab] = {"n": int(m.sum()), "cancer_deaths": int(yte[m].sum()), "note": "too few cancer deaths"}
            continue
        sl, ic = slope_intercept(yte[m], pc[m])
        au = float(roc_auc_score(yte[m], pc[m]))
        f4[lab] = {"n": int(m.sum()), "cancer_deaths": int(yte[m].sum()), "slope": round(sl, 3), "intercept": round(ic, 3), "auc": round(au, 4),
                   "observed_over_expected": round(float(yte[m].sum() / pc[m].sum()), 3)}
        ok4 &= 0.8 <= sl <= 1.2
    aucs = [v["auc"] for v in f4.values() if "auc" in v]
    spread = max(aucs) - min(aucs)
    ok4 &= spread <= 0.03
    res["bar4"] = {"groups": f4, "auc_spread": round(spread, 4), "passes": bool(ok4)}
    print("BAR 4 fairness:", {k: (v.get("slope"), v.get("auc")) for k, v in f4.items()}, f"AUC spread {spread:.3f} ->", "MET" if ok4 else "MISSED")

    # cancer-risk age
    ages = np.arange(35, 96, 0.5)
    r0 = {}
    for sx in (0.0, 1.0):
        grid = pd.DataFrame({"age": ages, "male": sx})
        r0[sx] = fits[0].predict(grid)
    rap_age = risk_age(pc, te["male"].to_numpy(), ages, r0)
    rap = rap_age - te["age"].to_numpy()
    te["rap"], te["risk_age"] = rap, rap_age
    ok5 = True
    b5 = {}
    for sx, lab in ((0.0, "women"), (1.0, "men")):
        m = ((te["male"] == sx) & (te["rap"] >= 5)).to_numpy()
        obs, exp = int(yte[m].sum()), float(pc[m].sum())
        pl, ph = poisson.ppf(0.025, exp), poisson.ppf(0.975, exp)
        ok = bool(pl <= obs <= ph or abs(obs - exp) <= 0.2 * exp)
        b5[lab] = {"n": int(m.sum()), "observed": obs, "expected": round(exp, 1), "ok": ok}
        ok5 &= ok
    res["bar5"] = {"groups": b5, "passes": bool(ok5)}
    print("BAR 5 risk age 5+ years above age, observed/expected:", {k: (v["observed"], v["expected"]) for k, v in b5.items()}, "->", "MET" if ok5 else "MISSED")
    res["rap_distribution"] = {k: round(float(np.percentile(rap, k)), 2) for k in (5, 25, 50, 75, 95)}
    for lab, mask in (("never smokers", te["smoke_status"] == 0), ("former smokers", te["smoke_status"] == 1), ("current smokers", te["smoke_status"] == 2)):
        res["rap_distribution"][lab] = round(float(rap[mask.to_numpy()].mean()), 2)
    cur = (te["smoke_status"] == 2) & (te["cigs_per_day"] >= 20) & te["age"].between(40, 49)
    res["rap_distribution"]["current smokers 20+/day aged 40-49"] = round(float(rap[cur.to_numpy()].mean()), 2) if cur.any() else None
    print("risk advancement period (years above own age): ", res["rap_distribution"])

    # ---------------- cost-matched invitation
    inv = te[te["age"].between(40, 74)].reset_index(drop=True)
    pinv = P[chosen][te["age"].between(40, 74).to_numpy()]
    yinv = inv["y"].to_numpy()
    agekey = inv["age"].to_numpy() + RNG.random(len(inv)) * 0.01
    shares = np.arange(0.05, 0.80, 0.05)
    ks = np.maximum((shares * len(inv)).astype(int), 1)
    covA, covB = coverage(yinv, agekey, ks), coverage(yinv, pinv, ks)
    diffs = []
    for _ in range(300):
        i = RNG.integers(0, len(inv), len(inv))
        diffs.append(coverage(yinv[i], pinv[i], ks) - coverage(yinv[i], agekey[i], ks))
    diffs = np.array(diffs)
    cost = {"n": int(len(inv)), "cancer_deaths": int(yinv.sum()), "shares": [round(float(s), 2) for s in shares],
            "coverage_age": [round(float(x), 4) for x in covA], "coverage_risk": [round(float(x), 4) for x in covB],
            "diff": [round(float(x), 4) for x in (covB - covA)], "diff_lo": [round(float(x), 4) for x in np.percentile(diffs, 2.5, axis=0)],
            "diff_hi": [round(float(x), 4) for x in np.percentile(diffs, 97.5, axis=0)]}
    # fixed coverage: how many invitations for 50% of cancer deaths
    full_age = np.cumsum(yinv[np.argsort(-agekey, kind="stable")]) / yinv.sum()
    full_risk = np.cumsum(yinv[np.argsort(-pinv, kind="stable")]) / yinv.sum()
    saving = {}
    for target in (0.4, 0.5, 0.6):
        ka, kb = int(np.searchsorted(full_age, target) + 1), int(np.searchsorted(full_risk, target) + 1)
        saving[str(target)] = {"invite_age": ka, "invite_risk": kb, "share_invited_age": round(ka / len(inv), 3), "share_invited_risk": round(kb / len(inv), 3),
                               "fewer_invitations": round(1 - kb / ka, 3)}
    cost["fixed_coverage"] = saving
    # earliness: among people invited under 50 at 30% invited
    k30 = int(0.30 * len(inv))
    inv_age_idx, inv_risk_idx = np.argsort(-agekey, kind="stable")[:k30], np.argsort(-pinv, kind="stable")[:k30]
    young = (inv["age"] < 50).to_numpy()
    cost["earliness_at_30pct_invited"] = {"age_policy_invites_under_50": int(young[inv_age_idx].sum()), "risk_policy_invites_under_50": int(young[inv_risk_idx].sum()),
                                          "cancer_deaths_under_50_covered_age": int(yinv[inv_age_idx][young[inv_age_idx]].sum()), "cancer_deaths_under_50_covered_risk": int(yinv[inv_risk_idx][young[inv_risk_idx]].sum()),
                                          "cancer_deaths_under_50_total": int(yinv[young].sum())}
    res["cost_matched"] = cost
    print("\nCOST-MATCHED INVITATION, ages 40-74 (temporal test)")
    for s, a, b, lo, hi in zip(cost["shares"], cost["coverage_age"], cost["coverage_risk"], cost["diff_lo"], cost["diff_hi"]):
        if round(s * 100) % 10 == 0:
            print(f"   invite {s:.0%}: age-only covers {a:.1%}, risk-based covers {b:.1%}, difference {b - a:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    for t, v in saving.items():
        print(f"   to cover {float(t):.0%} of cancer deaths: age-only invites {v['share_invited_age']:.0%}, risk-based {v['share_invited_risk']:.0%} ({v['fewer_invitations']:.0%} fewer invitations)")
    print("   earliness at 30% invited:", cost["earliness_at_30pct_invited"])
    bar_cost = bool(np.all(np.array(cost["diff_lo"][1:12]) > 0))
    res["bar_cost_matched"] = {"passes_temporal": bar_cost}
    print(f"   BAR (risk-based covers more at every share 10%-60%, interval above zero): {'MET' if bar_cost else 'MISSED'}")

    # ---------------- screening-start descriptive
    m4 = te["age"].between(40, 49).to_numpy()
    sub = te[m4]
    high = (sub["risk_age"] >= 50).to_numpy()
    desc = {"n_40_49": int(m4.sum()), "share_risk_age_50plus": round(float(high.mean()), 4), "cancer_deaths_40_49": int(sub["y"].sum()),
            "share_of_deaths_in_high_group": round(float(sub["y"][high].sum() / max(sub["y"].sum(), 1)), 4)}
    for k, col in (("male", "male"), ("current smoker", None), ("college+", None)):
        pass
    desc["composition_of_high_group"] = {"male": round(float(sub["male"][high].mean()), 3), "current smoker": round(float((sub["smoke_status"][high] == 2).mean()), 3),
                                         "former smoker": round(float((sub["smoke_status"][high] == 1).mean()), 3), "BMI 30+": round(float((sub["bmi"][high] >= 30).mean()), 3),
                                         "less than high school": round(float((sub["educ"][high] == 0).mean()), 3), "college+": round(float((sub["educ"][high] == 3).mean()), 3)}
    desc["composition_of_all_40_49"] = {"male": round(float(sub["male"].mean()), 3), "current smoker": round(float((sub["smoke_status"] == 2).mean()), 3),
                                        "BMI 30+": round(float((sub["bmi"] >= 30).mean()), 3), "less than high school": round(float((sub["educ"] == 0).mean()), 3), "college+": round(float((sub["educ"] == 3).mean()), 3)}
    k = int(high.sum())
    oldest = np.argsort(-sub["age"].to_numpy() - RNG.random(len(sub)) * 0.01, kind="stable")[:k]
    desc["share_of_deaths_flagging_oldest_same_number"] = round(float(sub["y"].to_numpy()[oldest].sum() / max(sub["y"].sum(), 1)), 4)
    res["screening_start"] = desc
    print("\nSCREENING START, adults 40-49:", {k_: v for k_, v in desc.items() if k_ not in ("composition_of_high_group", "composition_of_all_40_49")})
    print("   composition of the high-risk-age group:", desc["composition_of_high_group"])
    print("   composition of all 40-49:", desc["composition_of_all_40_49"])

    # ---------------- external: NHANES and NHANES III with the variables they share (age, sex, ever smoked, BMI)
    ext = {}
    tr_e = tr.copy()
    tr_e["smoke_status"] = (tr_e["smoke_status"] > 0).astype(float)
    tr_e["cigs_per_day"], tr_e["quit_years"] = 0.0, 0.0
    f0e, f2e = Fit(tr_e, 0), Fit(tr_e, 2)
    f1e_design = Fit(tr_e, 1)
    for name, loader in (("NHANES 1999-2008", load_nhanes), ("NHANES III 1988-1994", load_nhanes3)):
        x = loader()
        yx = x["y"].to_numpy()
        p0, p2 = f0e.predict(x), f2e.predict(x)
        s2, i2 = slope_intercept(yx, p2)
        bd2, okb = bands(x, p2, yx)
        gx = float(roc_auc_score(yx, p2) - roc_auc_score(yx, p0))
        ext[name] = {"n": int(len(x)), "cancer_deaths": int(yx.sum()), "auc_F0": round(float(roc_auc_score(yx, p0)), 4), "auc_F2": round(float(roc_auc_score(yx, p2)), 4), "gain": round(gx, 4),
                     "gain_ci": boot_auc_gain(yx, p0, p2), "slope": round(s2, 3), "intercept": round(i2, 3), "bands": bd2, "bands_ok": bool(okb),
                     "passes_bar1": bool(0.8 <= s2 <= 1.2 and -0.3 <= i2 <= 0.3)}
        # cost-matched in the external data, ages 40-74
        m = x["age"].between(40, 74).to_numpy()
        xi, yi = x[m], yx[m]
        kk = np.maximum((np.arange(0.1, 0.65, 0.1) * len(xi)).astype(int), 1)
        ca = coverage(yi, xi["age"].to_numpy() + RNG.random(len(xi)) * 0.01, kk)
        cb = coverage(yi, p2[m], kk)
        dd = []
        for _ in range(300):
            i = RNG.integers(0, len(xi), len(xi))
            dd.append(coverage(yi[i], p2[m][i], kk) - coverage(yi[i], xi["age"].to_numpy()[i] + RNG.random(len(i)) * 0.01, kk))
        ext[name]["cost_matched"] = {"shares": [round(float(s), 2) for s in np.arange(0.1, 0.65, 0.1)], "coverage_age": [round(float(v), 4) for v in ca], "coverage_risk": [round(float(v), 4) for v in cb],
                                     "diff_lo": [round(float(v), 4) for v in np.percentile(np.array(dd), 2.5, axis=0)], "diff_hi": [round(float(v), 4) for v in np.percentile(np.array(dd), 97.5, axis=0)]}
        ext[name]["cost_matched"]["passes"] = bool(np.all(np.array(ext[name]["cost_matched"]["diff_lo"]) > 0))
        print(f"\nEXTERNAL {name}: n={len(x):,}, cancer deaths {int(yx.sum())}; AUC F0 {ext[name]['auc_F0']:.3f} -> F2 {ext[name]['auc_F2']:.3f} (gain {gx:+.4f} {ext[name]['gain_ci']}); slope {s2:.3f} intercept {i2:.3f}")
        print("   bands:", "  ".join(f"{k_} {v['observed']}/{v['expected']}{'' if v['ok'] else ' MISS'}" for k_, v in bd2.items()))
        print("   cost-matched coverage (age vs risk) at 10-60% invited:", [(f"{a:.0%}", f"{b:.0%}") for a, b in zip(ext[name]['cost_matched']['coverage_age'], ext[name]['cost_matched']['coverage_risk'])],
              "-> interval above zero everywhere:", ext[name]["cost_matched"]["passes"])
    res["external"] = ext
    res["all_bars_temporal"] = bool(b1 and b2 and b3 and ok4 and ok5)
    print(f"\nTEMPORAL BARS 1-5 all met: {res['all_bars_temporal']}")
    # keep the fitted age curve and calibration deciles for the figures
    q = pd.qcut(pd.Series(pc), 10, duplicates="drop")
    res["deciles"] = [{"predicted": round(float(g.mean()), 5), "observed": round(float(yte[g.index].mean()), 5), "n": int(len(g))} for _, g in pd.Series(pc).groupby(q, observed=True)]
    res["r0_curve"] = {"ages": [float(a) for a in ages[::2]], "women": [round(float(v), 5) for v in r0[0.0][::2]], "men": [round(float(v), 5) for v in r0[1.0][::2]]}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}")


def load_nhanes():
    x = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    x["year"] = x["cycle"].str[:4].astype(int)
    x = x[(x["year"] <= 2007) & x["age"].between(35, 84)].copy()
    x = x[(x["followup_months"] >= 120) | (x["died"] == 1)]
    x["y"] = ((x["died"] == 1) & (x["ucod_leading"] == CANCER) & (x["followup_months"] <= 120)).astype(int)
    x["smoke_status"] = np.where(x["SMQ020"] == 1, 1.0, np.where(x["SMQ020"] == 2, 0.0, np.nan))
    x["bmi"] = x["BMXBMI"].clip(15, 55)
    x["male"] = (x["gender"] == 1).astype(float)
    x["cigs_per_day"], x["quit_years"] = 0.0, 0.0
    for c in ("vigorous_per_week", "married", "self_health"):
        x[c] = np.nan
    x["educ"], x["race4"] = -1, -1
    return x.dropna(subset=["smoke_status", "bmi"]).reset_index(drop=True)


def load_nhanes3():
    x = pd.read_csv(os.path.join(ROOT, "data", "nhanes3_markers.csv.gz"))
    for c in ("months", "died", "ucod", "age", "male", "smoked", "bmi", "prior_cancer"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[x["age"].between(35, 84) & (x["prior_cancer"] == 0)].dropna(subset=["months", "died", "smoked", "bmi"])
    x = x[(x["months"] >= 120) | (x["died"] == 1)]
    x["y"] = ((x["died"] == 1) & (x["ucod"] == CANCER) & (x["months"] <= 120)).astype(int)
    x["smoke_status"] = x["smoked"]
    x["bmi"] = x["bmi"].clip(15, 55)
    x["cigs_per_day"], x["quit_years"] = 0.0, 0.0
    for c in ("vigorous_per_week", "married", "self_health"):
        x[c] = np.nan
    x["educ"], x["race4"] = -1, -1
    return x.reset_index(drop=True)


if __name__ == "__main__":
    main()
