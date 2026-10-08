"""
Independent replication (R) and decision analysis (D) for the cheap-markers paper.
Both are fixed in the addendum of docs/CHEAP_MARKERS_PREREG.md, committed before this was run.

R  Tests 1 to 3 in NHANES III (1988-1994), a separate survey, using granulocyte analogues
   of the neutrophil-based markers. Test 1 is fully external: models are fitted on the
   1999-2008 cohort and applied to NHANES III.
D  What a marker-based triage rule would do among adults 60 and over, against flagging the
   oldest people by age alone at the same flag rate, in the 2005-2008 test cycles and in NHANES III.

Run:  python experiments/cheap_markers_replication.py
Needs data/nhanes3_markers.csv.gz (python fetch_nhanes3_markers.py)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cheap_markers import CANCER, NAMES, boot_gain, fit_pred, holm, indices, irls, m0

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cheap_markers_replication_result.json")
RNG = np.random.default_rng(20261009)
NEED = ["LBDNENO", "LBDLYMNO", "LBDMONO", "LBXPLTSI", "LBXNEPCT", "LBXSAL", "LBXRDW", "LBXWBCSI", "LBXHGB", "BMXBMI"]


def original():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan))
    d = d[(d["age"] >= 40) & (d["year0"] <= 2007)].dropna(subset=NEED).reset_index(drop=True)
    d = d[(d["followup_months"] >= 120) | (d["died"] == 1)].reset_index(drop=True)
    d["ycan"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= 120)).astype(int)
    d["ync"] = ((d["died"] == 1) & (d["ucod_leading"] != CANCER) & (d["followup_months"] <= 120)).astype(int)
    idx = indices(d)
    ok = idx.notna().all(axis=1)
    return d[ok].reset_index(drop=True), idx[ok].reset_index(drop=True)


def n3_indices(x):
    n, l, m, p, alb = x["gran_n"], x["lym_n"], x["mono_n"], x["platelets"], x["albumin"]
    glr = n / l
    out = pd.DataFrame({"NLR": glr, "PLR": p / l, "MLR": m / l, "SII": p * n / l, "SIRI": n * m / l, "NPAR": x["gran_pct"] / alb,
                        "ALI": x["bmi"] * alb / glr, "PNI": alb * 10 + 5 * l, "RDW": x["rdw"], "WBC": x["wbc"], "Albumin": alb, "Haemoglobin": x["hemoglobin"]})
    return np.log(out.where(out > 0))


def third():
    x = pd.read_csv(os.path.join(ROOT, "data", "nhanes3_markers.csv.gz"))
    for c in ("months", "died", "ucod", "age", "male", "smoked", "bmi", "prior_cancer"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["months", "died", "age"]).reset_index(drop=True)
    x = x[(x["age"] >= 40) & (x["prior_cancer"] == 0)].dropna(subset=["wbc", "lym_n", "mono_n", "gran_n", "gran_pct", "platelets", "albumin", "rdw", "hemoglobin", "bmi"]).reset_index(drop=True)
    x = x[(x["months"] >= 120) | (x["died"] == 1)].reset_index(drop=True)
    x["ycan"] = ((x["died"] == 1) & (x["ucod"] == CANCER) & (x["months"] <= 120)).astype(int)
    x["ync"] = ((x["died"] == 1) & (x["ucod"] != CANCER) & (x["months"] <= 120)).astype(int)
    idx = n3_indices(x)
    ok = idx.notna().all(axis=1)
    return x[ok].reset_index(drop=True), idx[ok].reset_index(drop=True)


def m0_n3(x, sm, bm):
    a = x["age"].to_numpy() - 60.0
    return np.column_stack([a, a ** 2 / 20.0, x["male"].to_numpy(), x["smoked"].fillna(sm).to_numpy(), (x["bmi"].fillna(bm).to_numpy() - 27.0) / 5.0])


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.96
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return [round(float(max(0, c - h)), 4), round(float(min(1, c + h)), 4)]


def rule_stats(y, flag, age, n_boot=1000):
    k = int(flag.sum())
    oldest = np.argsort(-age, kind="stable")[:k]
    agef = np.zeros(len(y), dtype=bool)
    agef[oldest] = True
    out = {"flagged_share": round(k / len(y), 4), "n": int(len(y)), "cancer_deaths": int(y.sum())}
    for lab, f in (("marker", flag), ("age_only", agef)):
        tp = int((f & (y == 1)).sum())
        out[lab] = {"sensitivity": round(tp / max(int(y.sum()), 1), 4), "sens_ci": wilson(tp, int(y.sum())), "ppv": round(tp / max(int(f.sum()), 1), 4), "tp": tp}
    diffs = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(y), len(y))
        yy, ff, aa = y[i], flag[i], age[i]
        kk = int(ff.sum())
        if kk == 0 or yy.sum() == 0:
            continue
        og = np.zeros(len(yy), dtype=bool)
        og[np.argsort(-aa, kind="stable")[:kk]] = True
        diffs.append(((ff & (yy == 1)).sum() - (og & (yy == 1)).sum()) / yy.sum())
    out["sens_diff_marker_minus_age"] = round(out["marker"]["sensitivity"] - out["age_only"]["sensitivity"], 4)
    out["sens_diff_ci"] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    return out, agef


def other_cause(yo, flag, age):
    k = int(flag.sum())
    og = np.zeros(len(yo), dtype=bool)
    og[np.argsort(-age, kind="stable")[:k]] = True
    n = max(int(yo.sum()), 1)
    return {"deaths": int(yo.sum()), "marker_sensitivity": round(float((flag & (yo == 1)).sum() / n), 4), "age_only_sensitivity": round(float((og & (yo == 1)).sum() / n), 4)}


def net_benefit(y, flag, pt):
    n = len(y)
    tp = (flag & (y == 1)).sum()
    fp = (flag & (y == 0)).sum()
    return float(tp / n - fp / n * pt / (1 - pt))


def main():
    d, idx = original()
    tr, te = d["year0"] <= 2007, d["year0"] >= 2005
    sm, bm = float(d["smoked"].mean()), float(d["BMXBMI"].mean())
    mu, sd = idx.mean(), idx.std()
    z0 = (idx - mu) / sd
    X0 = m0(d, sm, bm)
    x, ix = third()
    muN, sdN = ix.mean(), ix.std()
    zN = (ix - muN) / sdN
    X0N = m0_n3(x, sm, bm)
    ycN, ynN = x["ycan"].to_numpy(), x["ync"].to_numpy()
    print(f"NHANES III: {len(x):,} adults 40+, complete analogue markers; ten-year cancer deaths {int(ycN.sum())}, other-cause deaths {int(ynN.sum())}; smoking known for {x['smoked'].notna().mean():.0%}")
    res = {"n3": {"n": int(len(x)), "cancer_deaths": int(ycN.sum()), "noncancer_deaths": int(ynN.sum())}, "indices": {}}
    yc, yn = d["ycan"].to_numpy(), d["ync"].to_numpy()
    base_c = fit_pred(X0, yc, X0N)
    base_n = fit_pred(X0, yn, X0N)
    res["baseline_auc"] = {"cancer": round(float(roc_auc_score(ycN, base_c)), 4), "noncancer": round(float(roc_auc_score(ynN, base_n)), 4)}
    print(f"baseline M0 fitted on 1999-2008, applied to NHANES III: AUC cancer death {res['baseline_auc']['cancer']:.3f}, other-cause death {res['baseline_auc']['noncancer']:.3f}\n")
    decm = (ycN + ynN) == 1
    rows, pv = {}, []
    cyc = np.zeros(len(x))
    for name in NAMES:
        Xi_tr = np.column_stack([X0, z0[name].to_numpy()])
        Xi_te = np.column_stack([X0N, zN[name].to_numpy()])
        pc, pn = fit_pred(Xi_tr, yc, Xi_te), fit_pred(Xi_tr, yn, Xi_te)
        g1 = float(roc_auc_score(ycN, pc) - roc_auc_score(ycN, base_c))
        g3 = float(roc_auc_score(ynN, pn) - roc_auc_score(ynN, base_n))
        ci1, ci3 = boot_gain(ycN, base_c, pc), boot_gain(ynN, base_n, pn)
        zi = zN[name].to_numpy()
        Xd = np.column_stack([np.ones(decm.sum()), X0N[decm], zi[decm]])
        b, se = irls(Xd, ycN[decm].astype(float))
        j = X0N.shape[1] + 1
        Xa = np.column_stack([np.ones(len(x)), X0N, zi])
        bc, sc = irls(Xa, ycN.astype(float))
        bn, sn = irls(Xa, ynN.astype(float))
        p2 = float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(b[j] / se[j]))))
        rows[name] = {"T1_gain": round(g1, 4), "T1_ci": ci1, "T3_gain": round(g3, 4), "T3_ci": ci3,
                      "T2_or": round(float(np.exp(b[j])), 3), "T2_ci": [round(float(np.exp(b[j] - 1.96 * se[j])), 3), round(float(np.exp(b[j] + 1.96 * se[j])), 3)], "T2_p": p2,
                      "or_cancer_death": round(float(np.exp(bc[j])), 3), "or_cancer_ci": [round(float(np.exp(bc[j] - 1.96 * sc[j])), 3), round(float(np.exp(bc[j] + 1.96 * sc[j])), 3)],
                      "or_noncancer_death": round(float(np.exp(bn[j])), 3), "or_noncancer_ci": [round(float(np.exp(bn[j] - 1.96 * sn[j])), 3), round(float(np.exp(bn[j] + 1.96 * sn[j])), 3)]}
        pv.append(p2)
    adj = holm(np.array(pv))
    for k, name in enumerate(NAMES):
        r = rows[name]
        r["T2_p_holm"] = float(adj[k])
        r["passes_T1"] = bool(r["T1_gain"] >= 0.01 and r["T1_ci"][0] > 0)
        r["passes_T2"] = bool(r["T2_or"] > 1 and adj[k] < 0.05)
        r["qualifies"] = bool(r["passes_T1"] and r["passes_T2"])
        r["frailty_pattern"] = bool(r["T2_or"] < 1 and r["or_noncancer_death"] > r["or_cancer_death"])
        res["indices"][name] = r
    print(f"{'index':<12}{'T1 gain (cancer)':>22}{'T3 gain (other)':>20}{'T2 OR':>18}{'Holm p':>9}   OR cancer / OR other")
    for name in NAMES:
        r = res["indices"][name]
        print(f"{name:<12}{r['T1_gain']:>+9.4f} [{r['T1_ci'][0]:+.3f},{r['T1_ci'][1]:+.3f}]  {r['T3_gain']:>+8.4f}  {r['T2_or']:>6.2f} [{r['T2_ci'][0]:.2f},{r['T2_ci'][1]:.2f}]  {r['T2_p_holm']:>7.3f}   {r['or_cancer_death']:.2f} / {r['or_noncancer_death']:.2f}"
              f"{'  QUALIFIES' if r['qualifies'] else ''}{'  (frailty pattern)' if r['frailty_pattern'] else ''}")
    infl = ["NLR", "PLR", "MLR", "SII", "SIRI", "NPAR"]
    res["n_qualify"] = int(sum(v["qualifies"] for v in res["indices"].values()))
    res["frailty_pattern_inflammation"] = int(sum(res["indices"][n]["frailty_pattern"] for n in infl))
    res["replicated"] = bool(res["n_qualify"] == 0 and res["frailty_pattern_inflammation"] >= 5)
    print(f"\nqualify in NHANES III: {res['n_qualify']} of 12; inflammation analogues with the frailty pattern (decedent OR < 1 and other-cause OR > cancer OR): {res['frailty_pattern_inflammation']} of 6")
    print(f"REPLICATION BAR (none qualify and at least 5 of 6 show the pattern): {'MET' if res['replicated'] else 'NOT MET'}")

    # ---------------- D: decision analysis, adults 60 and over
    print("\nD. Decision analysis among adults 60 and over")
    dd = {}
    ted = d[te & (d["age"] >= 60)]
    ixd = idx[te & (d["age"] >= 60)]
    nlr_t = float(np.exp(3.0 and np.log(3.0)))
    rules = {}
    # test cycles 2005-2008
    nlr_raw = np.exp(ixd["NLR"].to_numpy())
    flag_nlr3 = nlr_raw >= 3.0
    q = {"NLR >= 3": flag_nlr3, "top fifth of NLR": ixd["NLR"].to_numpy() >= np.quantile(ixd["NLR"], 0.8),
         "top fifth of SII": ixd["SII"].to_numpy() >= np.quantile(ixd["SII"], 0.8), "top fifth of RDW": ixd["RDW"].to_numpy() >= np.quantile(ixd["RDW"], 0.8)}
    share_nlr3 = float(flag_nlr3.mean())
    y60, a60 = ted["ycan"].to_numpy(), ted["age"].to_numpy()
    dd["test_cycles_2005_2008"] = {"n": int(len(ted)), "cancer_deaths": int(y60.sum()), "rules": {}}
    for k, f in q.items():
        s, _ = rule_stats(y60, f, a60)
        dd["test_cycles_2005_2008"]["rules"][k] = s
        s["other_cause"] = other_cause(ted["ync"].to_numpy(), f, a60)
        s["net_benefit"] = {str(pt): {"rule": round(net_benefit(y60, f, pt), 5), "treat_all": round(net_benefit(y60, np.ones(len(y60), dtype=bool), pt), 5)} for pt in (0.02, 0.05, 0.10)}
    # NHANES III, quantile-matched cutoff for the NLR >= 3 rule
    n60 = x[x["age"] >= 60].reset_index(drop=True)
    in60 = ix[x["age"] >= 60].reset_index(drop=True)
    y3, a3 = n60["ycan"].to_numpy(), n60["age"].to_numpy()
    cut = np.quantile(in60["NLR"], 1 - share_nlr3)
    q3 = {"NLR-analogue at the cutoff that flags the same share as NLR >= 3": in60["NLR"].to_numpy() >= cut,
          "top fifth of GLR": in60["NLR"].to_numpy() >= np.quantile(in60["NLR"], 0.8), "top fifth of SII-analogue": in60["SII"].to_numpy() >= np.quantile(in60["SII"], 0.8),
          "top fifth of RDW": in60["RDW"].to_numpy() >= np.quantile(in60["RDW"], 0.8)}
    dd["nhanes3"] = {"n": int(len(n60)), "cancer_deaths": int(y3.sum()), "rules": {}}
    for k, f in q3.items():
        s, _ = rule_stats(y3, f, a3)
        s["net_benefit"] = {str(pt): {"rule": round(net_benefit(y3, f, pt), 5), "treat_all": round(net_benefit(y3, np.ones(len(y3), dtype=bool), pt), 5)} for pt in (0.02, 0.05, 0.10)}
        dd["nhanes3"]["rules"][k] = s
        s["other_cause"] = other_cause(n60["ync"].to_numpy(), f, a3)
    res["decision"] = dd
    base = float(y60.mean())
    print(f"  2005-2008, age 60+: n={len(ted):,}, ten-year cancer death rate {base:.1%}")
    for k, s in dd["test_cycles_2005_2008"]["rules"].items():
        print(f"   {k:<22} flags {s['flagged_share']:.0%}: sens {s['marker']['sensitivity']:.1%} PPV {s['marker']['ppv']:.1%} | oldest-same-share sens {s['age_only']['sensitivity']:.1%} PPV {s['age_only']['ppv']:.1%} | diff {s['sens_diff_marker_minus_age']:+.3f} {s['sens_diff_ci']}")
    print(f"  NHANES III, age 60+: n={len(n60):,}, ten-year cancer death rate {float(y3.mean()):.1%}")
    for k, s in dd["nhanes3"]["rules"].items():
        print(f"   {k[:48]:<48} flags {s['flagged_share']:.0%}: sens {s['marker']['sensitivity']:.1%} PPV {s['marker']['ppv']:.1%} | age-only sens {s['age_only']['sensitivity']:.1%} | diff {s['sens_diff_marker_minus_age']:+.3f} {s['sens_diff_ci']}")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
