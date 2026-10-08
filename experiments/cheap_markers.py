"""
Cheap blood markers of cancer risk: cancer signal or general frailty?
Questions and bars: docs/CHEAP_MARKERS_PREREG.md (committed before this was run).

Twelve blood-count indices, each added to a baseline of age, sex, smoking and BMI:
  T1  added discrimination for ten-year cancer death in an unseen era
  T2  specificity: among decedents, cancer versus other cause
  T3  negative control: the same index against non-cancer death
  T4  all twelve together
  T5  the literature's own design: association with self-reported cancer history

Run:  python experiments/cheap_markers.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cheap_markers_result.json")
RNG = np.random.default_rng(20261008)
CANCER = 2
NAMES = ["NLR", "PLR", "MLR", "SII", "SIRI", "NPAR", "ALI", "PNI", "RDW", "WBC", "Albumin", "Haemoglobin"]


def indices(d):
    n, l, m, p = d["LBDNENO"], d["LBDLYMNO"], d["LBDMONO"], d["LBXPLTSI"]
    alb = d["LBXSAL"]
    nlr = n / l
    out = pd.DataFrame({
        "NLR": nlr, "PLR": p / l, "MLR": m / l, "SII": p * n / l, "SIRI": n * m / l, "NPAR": d["LBXNEPCT"] / alb,
        "ALI": d["BMXBMI"] * alb / nlr, "PNI": alb * 10 + 5 * l, "RDW": d["LBXRDW"], "WBC": d["LBXWBCSI"],
        "Albumin": alb, "Haemoglobin": d["LBXHGB"]})
    out = out.where(out > 0)
    return np.log(out)


def irls(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    W = p * (1 - p) + 1e-9
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def m0(d, smoked_mean, bmi_mean):
    a = d["age"].to_numpy() - 60.0
    return np.column_stack([a, a ** 2 / 20.0, (d["gender"] == 1).astype(float).to_numpy(), d["smoked"].fillna(smoked_mean).to_numpy(),
                            (d["BMXBMI"].fillna(bmi_mean).to_numpy() - 27.0) / 5.0])


def boot_gain(y, a, b, n=1000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    g = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        g.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return [round(float(x), 4) for x in np.percentile(g, [2.5, 97.5])]


def fit_pred(Xtr, ytr, Xte, C=1e9):
    return LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan))
    d = d[d["age"] >= 40]
    need = ["LBDNENO", "LBDLYMNO", "LBDMONO", "LBXPLTSI", "LBXNEPCT", "LBXSAL", "LBXRDW", "LBXWBCSI", "LBXHGB", "BMXBMI"]
    d = d.dropna(subset=need).reset_index(drop=True)
    d = d[(d["followup_months"] >= 120) | (d["died"] == 1)].reset_index(drop=True)
    d = d[d["year0"] <= 2007].reset_index(drop=True)   # exam years 1999-2008 all have >= 10 years of follow-up
    d["ycan"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= 120)).astype(int)
    d["ync"] = ((d["died"] == 1) & (d["ucod_leading"] != CANCER) & (d["followup_months"] <= 120)).astype(int)
    d["dec"] = ((d["died"] == 1) & (d["followup_months"] <= 120)).astype(int)
    idx = indices(d)
    ok = idx.notna().all(axis=1)
    d, idx = d[ok].reset_index(drop=True), idx[ok].reset_index(drop=True)
    tr = d["year0"] <= 2003
    te = d["year0"] >= 2005
    sm, bm = float(d.loc[tr, "smoked"].mean()), float(d.loc[tr, "BMXBMI"].mean())
    mu, sd = idx[tr].mean(), idx[tr].std()
    z = (idx - mu) / sd
    X0 = m0(d, sm, bm)
    print(f"{len(d):,} adults 40+ with a complete blood count and albumin (cycles 1999-2008); fit {int(tr.sum()):,} (1999-2004), test {int(te.sum()):,} (2005-2008)")
    print(f"ten-year cancer deaths: fit {int(d.loc[tr, 'ycan'].sum())}, test {int(d.loc[te, 'ycan'].sum())};  non-cancer deaths: fit {int(d.loc[tr, 'ync'].sum())}, test {int(d.loc[te, 'ync'].sum())}\n")
    res = {"n": int(len(d)), "n_fit": int(tr.sum()), "n_test": int(te.sum()), "cancer_deaths_fit": int(d.loc[tr, "ycan"].sum()),
           "cancer_deaths_test": int(d.loc[te, "ycan"].sum()), "noncancer_deaths_fit": int(d.loc[tr, "ync"].sum()),
           "noncancer_deaths_test": int(d.loc[te, "ync"].sum()), "indices": {}}
    ycan, ync = d["ycan"].to_numpy(), d["ync"].to_numpy()
    pbase_c = fit_pred(X0[tr], ycan[tr], X0[te])
    pbase_n = fit_pred(X0[tr], ync[tr], X0[te])
    res["baseline_auc"] = {"cancer": round(float(roc_auc_score(ycan[te], pbase_c)), 4), "noncancer": round(float(roc_auc_score(ync[te], pbase_n)), 4)}
    print(f"baseline M0 AUC: cancer death {res['baseline_auc']['cancer']:.3f}, non-cancer death {res['baseline_auc']['noncancer']:.3f}\n")

    decm = d["dec"].to_numpy() == 1
    ycd = ycan[decm]
    pvals = []
    rows = {}
    for name in NAMES:
        zi = z[name].to_numpy()
        Xi = np.column_stack([X0, zi])
        # T1 and T3: out-of-era added discrimination, cancer and non-cancer death
        pc = fit_pred(Xi[tr], ycan[tr], Xi[te])
        pn = fit_pred(Xi[tr], ync[tr], Xi[te])
        g1 = float(roc_auc_score(ycan[te], pc) - roc_auc_score(ycan[te], pbase_c))
        g3 = float(roc_auc_score(ync[te], pn) - roc_auc_score(ync[te], pbase_n))
        ci1 = boot_gain(ycan[te], pbase_c, pc)
        ci3 = boot_gain(ync[te], pbase_n, pn)
        # T2: among decedents, cancer versus other cause, per SD
        cyc = (d["year0"].to_numpy() - 2003.0) / 4.0
        Xd = np.column_stack([np.ones(decm.sum()), X0[decm], zi[decm], cyc[decm]])
        b, se = irls(Xd, ycd.astype(float))
        j = X0.shape[1] + 1
        orr = float(np.exp(b[j]))
        ci2 = [float(np.exp(b[j] - 1.96 * se[j])), float(np.exp(b[j] + 1.96 * se[j]))]
        p2 = float(2 * (1 - norm.cdf(abs(b[j] / se[j]))))
        # per-SD odds against survivors, pooled, adjusted for M0 and cycle
        Xa = np.column_stack([np.ones(len(d)), X0, zi, cyc])
        bc, sc = irls(Xa, ycan.astype(float))
        bn, sn = irls(Xa, ync.astype(float))
        rows[name] = {"T1_gain": round(g1, 4), "T1_ci": ci1, "T3_gain": round(g3, 4), "T3_ci": ci3,
                      "T2_or": round(orr, 3), "T2_ci": [round(ci2[0], 3), round(ci2[1], 3)], "T2_p": p2,
                      "or_cancer_death": round(float(np.exp(bc[j])), 3), "or_cancer_ci": [round(float(np.exp(bc[j] - 1.96 * sc[j])), 3), round(float(np.exp(bc[j] + 1.96 * sc[j])), 3)],
                      "or_noncancer_death": round(float(np.exp(bn[j])), 3), "or_noncancer_ci": [round(float(np.exp(bn[j] - 1.96 * sn[j])), 3), round(float(np.exp(bn[j] + 1.96 * sn[j])), 3)]}
        pvals.append(p2)
    adj = holm(np.array(pvals))
    for k, name in enumerate(NAMES):
        r = rows[name]
        r["T2_p_holm"] = float(adj[k])
        r["passes_T1"] = bool(r["T1_gain"] >= 0.01 and r["T1_ci"][0] > 0)
        r["passes_T2"] = bool(r["T2_or"] > 1 and adj[k] < 0.05)
        r["qualifies"] = bool(r["passes_T1"] and r["passes_T2"])
        r["frailty_marker"] = bool(r["or_noncancer_death"] > r["or_cancer_death"])
        res["indices"][name] = r
    print(f"{'index':<12}{'T1 gain (cancer)':>20}{'T3 gain (non-cancer)':>24}{'T2 OR cancer vs other':>26}{'Holm p':>9}   OR cancer / OR non-cancer vs alive")
    for name in NAMES:
        r = res["indices"][name]
        print(f"{name:<12}{r['T1_gain']:>+9.4f} [{r['T1_ci'][0]:+.3f},{r['T1_ci'][1]:+.3f}]  {r['T3_gain']:>+8.4f} [{r['T3_ci'][0]:+.3f},{r['T3_ci'][1]:+.3f}]  "
              f"{r['T2_or']:>6.2f} [{r['T2_ci'][0]:.2f},{r['T2_ci'][1]:.2f}]  {r['T2_p_holm']:>8.3f}   {r['or_cancer_death']:.2f} / {r['or_noncancer_death']:.2f}"
              f"   {'QUALIFIES' if r['qualifies'] else ''}{' (frailty)' if r['frailty_marker'] and not r['qualifies'] else ''}")
    res["n_qualify"] = int(sum(v["qualifies"] for v in res["indices"].values()))
    res["n_frailty"] = int(sum(v["frailty_marker"] for v in res["indices"].values()))
    print(f"\nindices that qualify (pass T1 and T2): {res['n_qualify']} of 12;  more strongly linked to non-cancer than cancer death: {res['n_frailty']} of 12")

    # T4 all twelve together
    Xall = np.column_stack([X0, z.to_numpy()])
    pa = fit_pred(Xall[tr], ycan[tr], Xall[te], C=0.05)
    pb = fit_pred(X0[tr], ycan[tr], X0[te], C=0.05)
    res["T4"] = {"auc_baseline": round(float(roc_auc_score(ycan[te], pb)), 4), "auc_all": round(float(roc_auc_score(ycan[te], pa)), 4),
                 "gain": round(float(roc_auc_score(ycan[te], pa) - roc_auc_score(ycan[te], pb)), 4), "ci": boot_gain(ycan[te], pb, pa)}
    res["T4"]["passes"] = bool(res["T4"]["gain"] >= 0.01 and res["T4"]["ci"][0] > 0)
    print(f"T4 all twelve with baseline: AUC {res['T4']['auc_all']:.3f} vs {res['T4']['auc_baseline']:.3f}, gain {res['T4']['gain']:+.4f} {res['T4']['ci']}  -> {'passes' if res['T4']['passes'] else 'does not pass'}")

    # C-reactive protein, cycles with CRP
    dc = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"), usecols=["cycle", "age", "gender", "SMQ020", "BMXBMI", "LBXCRP", "died", "ucod_leading", "followup_months"])
    dc["year0"] = dc["cycle"].str[:4].astype(int)
    dc["smoked"] = np.where(dc["SMQ020"] == 1, 1.0, np.where(dc["SMQ020"] == 2, 0.0, np.nan))
    dc = dc[(dc["age"] >= 40) & (dc["year0"] <= 2007)].dropna(subset=["LBXCRP", "BMXBMI"])
    dc = dc[(dc["followup_months"] >= 120) | (dc["died"] == 1)].reset_index(drop=True)
    dc["ycan"] = ((dc["died"] == 1) & (dc["ucod_leading"] == CANCER) & (dc["followup_months"] <= 120)).astype(int)
    dc["ync"] = ((dc["died"] == 1) & (dc["ucod_leading"] != CANCER) & (dc["followup_months"] <= 120)).astype(int)
    lc = np.log(dc["LBXCRP"].clip(lower=0.01))
    trc, tec = dc["year0"] <= 2003, dc["year0"] >= 2005
    zc = ((lc - lc[trc].mean()) / lc[trc].std()).to_numpy()
    sm2, bm2 = float(dc.loc[trc, "smoked"].mean()), float(dc.loc[trc, "BMXBMI"].mean())
    Xc0 = m0(dc, sm2, bm2)
    Xc1 = np.column_stack([Xc0, zc])
    yc, yn = dc["ycan"].to_numpy(), dc["ync"].to_numpy()
    p0, p1 = fit_pred(Xc0[trc], yc[trc], Xc0[tec]), fit_pred(Xc1[trc], yc[trc], Xc1[tec])
    q0, q1 = fit_pred(Xc0[trc], yn[trc], Xc0[tec]), fit_pred(Xc1[trc], yn[trc], Xc1[tec])
    dm = ((dc["ycan"] + dc["ync"]) == 1).to_numpy()
    Xd = np.column_stack([np.ones(dm.sum()), Xc0[dm], zc[dm], (dc["year0"].to_numpy()[dm] - 2003.0) / 4.0])
    b, se = irls(Xd, yc[dm].astype(float))
    j = Xc0.shape[1] + 1
    res["CRP"] = {"n": int(len(dc)), "T1_gain": round(float(roc_auc_score(yc[tec], p1) - roc_auc_score(yc[tec], p0)), 4), "T1_ci": boot_gain(yc[tec], p0, p1),
                  "T3_gain": round(float(roc_auc_score(yn[tec], q1) - roc_auc_score(yn[tec], q0)), 4),
                  "T2_or": round(float(np.exp(b[j])), 3), "T2_ci": [round(float(np.exp(b[j] - 1.96 * se[j])), 3), round(float(np.exp(b[j] + 1.96 * se[j])), 3)]}
    print(f"CRP (n={len(dc):,}): T1 gain {res['CRP']['T1_gain']:+.4f} {res['CRP']['T1_ci']}, T3 gain {res['CRP']['T3_gain']:+.4f}, T2 OR {res['CRP']['T2_or']} {res['CRP']['T2_ci']}")

    # T5 the literature's design: self-reported cancer history at the same visit
    x = pd.read_csv(os.path.join(ROOT, "data", "nhanes_xs_8c.csv.gz"))
    x["year0"] = x["cycle"].str[:4].astype(int)
    x["smoked"] = np.where(x["SMQ020"] == 1, 1.0, np.where(x["SMQ020"] == 2, 0.0, np.nan)) if "SMQ020" in x else np.nan
    x = x[x["age"] >= 40].dropna(subset=need).reset_index(drop=True)
    ix = indices(x)
    okx = ix.notna().all(axis=1)
    x, ix = x[okx].reset_index(drop=True), ix[okx].reset_index(drop=True)
    xs_tr = x["year0"] <= 2005
    xs_te = x["year0"] >= 2007
    mu2, sd2 = ix[xs_tr].mean(), ix[xs_tr].std()
    zx = (ix - mu2) / sd2
    smx, bmx = float(x.loc[xs_tr, "smoked"].mean()), float(x.loc[xs_tr, "BMXBMI"].mean())
    X0x = m0(x, smx, bmx)
    yx = x["cancer"].to_numpy()
    base_p = fit_pred(X0x[xs_tr], yx[xs_tr], X0x[xs_te])
    res["T5"] = {"n": int(len(x)), "cases": int(yx.sum()), "baseline_auc": round(float(roc_auc_score(yx[xs_te], base_p)), 4), "indices": {}}
    print(f"\nT5 self-reported cancer history, {len(x):,} adults 40+, {int(yx.sum()):,} cases; baseline AUC (later cycles) {res['T5']['baseline_auc']:.3f}")
    for name in NAMES:
        zi = zx[name].to_numpy()
        Xa = np.column_stack([np.ones(len(x)), X0x, zi, (x["year0"].to_numpy() - 2006.0) / 4.0])
        b, se = irls(Xa, yx.astype(float))
        j = X0x.shape[1] + 1
        pi = fit_pred(np.column_stack([X0x, zi])[xs_tr], yx[xs_tr], np.column_stack([X0x, zi])[xs_te])
        gain = float(roc_auc_score(yx[xs_te], pi) - roc_auc_score(yx[xs_te], base_p))
        res["T5"]["indices"][name] = {"or": round(float(np.exp(b[j])), 3), "ci": [round(float(np.exp(b[j] - 1.96 * se[j])), 3), round(float(np.exp(b[j] + 1.96 * se[j])), 3)],
                                      "auc_gain_later_cycles": round(gain, 4)}
        r = res["T5"]["indices"][name]
        print(f"   {name:<12} OR per SD {r['or']:.2f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]   AUC gain in later cycles {r['auc_gain_later_cycles']:+.4f}")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
