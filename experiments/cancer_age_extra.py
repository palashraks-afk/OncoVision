"""
Exploratory follow-ups to experiments/cancer_age.py. Everything here is labelled exploratory in the
paper; none carries a bar.

  1  who benefits? discrimination, calibration and the cost-matched gain within race and ethnicity,
     schooling and sex
  2  does the fairness shortfall come from the model or from the outcome? age-and-sex alone by group
  3  robustness: first 24 months removed; a five-year horizon
  4  NHANES and NHANES III pooled for the cost-matched comparison (each alone is underpowered)

Run:  python experiments/cancer_age_extra.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cancer_age import CANCER, Fit, coverage, load_nhanes, load_nhanes3, load_nhis, slope_intercept

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cancer_age_extra_result.json")
RNG = np.random.default_rng(20261011)


def gain_at(y, age, p, share, n_boot=400):
    k = max(int(share * len(y)), 1)
    key = age + RNG.random(len(y)) * 0.01
    ca, cb = coverage(y, key, [k])[0], coverage(y, p, [k])[0]
    d = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        d.append(coverage(y[i], p[i], [k])[0] - coverage(y[i], key[i], [k])[0])
    return {"age_only": round(float(ca), 4), "risk_based": round(float(cb), 4), "diff": round(float(cb - ca), 4), "ci": [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]}


def main():
    d = load_nhis()
    tr, te = d[d["year"] <= 2002].reset_index(drop=True), d[d["year"] >= 2005].reset_index(drop=True)
    f0, f2, f4 = Fit(tr, 0), Fit(tr, 2), Fit(tr, 4)
    y = te["y"].to_numpy()
    p0, p2, p4 = f0.predict(te), f2.predict(te), f4.predict(te)
    inv = te["age"].between(40, 74).to_numpy()
    res = {"groups": {}}

    print("1 and 2. Within groups (temporal test, NHIS 2005-2009)\n")
    print(f"{'group':<26}{'n':>8}{'cancer':>8}{'AUC F0':>8}{'AUC F2':>8}{'slope F2':>9}{'O/E':>6}   cost-matched gain at 20% invited (ages 40-74)")
    labs = {"race": ((0, "non-Hispanic White"), (1, "non-Hispanic Black"), (2, "Hispanic"), (3, "other")),
            "educ": ((0, "less than high school"), (1, "high school or GED"), (2, "some college"), (3, "college or more")),
            "male": ((0.0, "women"), (1.0, "men"))}
    for col, items in labs.items():
        key = "race4" if col == "race" else col
        for code, name in items:
            m = (te[key] == code).to_numpy()
            if int(y[m].sum()) < 30:
                continue
            sl, ic = slope_intercept(y[m], p2[m])
            mi = m & inv
            g = gain_at(y[mi], te["age"].to_numpy()[mi], p2[mi], 0.20) if int(y[mi].sum()) >= 30 else None
            r = {"n": int(m.sum()), "cancer_deaths": int(y[m].sum()), "auc_F0": round(float(roc_auc_score(y[m], p0[m])), 4), "auc_F2": round(float(roc_auc_score(y[m], p2[m])), 4),
                 "auc_F4": round(float(roc_auc_score(y[m], p4[m])), 4), "slope_F2": round(sl, 3), "observed_over_expected": round(float(y[m].sum() / p2[m].sum()), 3), "gain_at_20pct": g,
                 "current_smoker_share": round(float((te["smoke_status"][m] == 2).mean()), 3), "mean_age": round(float(te["age"][m].mean()), 1)}
            res["groups"][name] = r
            gs = f"{g['age_only']:.0%} -> {g['risk_based']:.0%} ({g['diff']:+.1%} [{g['ci'][0]:+.1%}, {g['ci'][1]:+.1%}])" if g else "n/a"
            print(f"{name:<26}{r['n']:>8,}{r['cancer_deaths']:>8}{r['auc_F0']:>8.3f}{r['auc_F2']:>8.3f}{sl:>9.2f}{r['observed_over_expected']:>6.2f}   {gs}")

    # robustness
    print("\n3. Robustness")
    rob = {}
    lag = te[te["months"] > 24].reset_index(drop=True)
    pl = f2.predict(lag)
    rob["first_24_months_removed"] = {"n": int(len(lag)), "cancer_deaths": int(lag["y"].sum()), "auc_F2": round(float(roc_auc_score(lag["y"], pl)), 4),
                                      "auc_F0": round(float(roc_auc_score(lag["y"], f0.predict(lag))), 4)}
    il = lag["age"].between(40, 74).to_numpy()
    rob["first_24_months_removed"]["gain_at_20pct"] = gain_at(lag["y"].to_numpy()[il], lag["age"].to_numpy()[il], pl[il], 0.20)
    print("  first 24 months removed:", rob["first_24_months_removed"])
    d5 = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    d5 = d5[(d5["age"] >= 35) & (d5["age"] <= 84) & (d5["prior_cancer"] == 0) & (d5["year"] <= 2013)]
    d5["y"] = ((d5["mortstat"] == 1) & (d5["ucod"] == CANCER) & (d5["months"] <= 60)).astype(int)
    rob["five_year_note"] = "five-year horizon uses the same fitted ten-year model scaled; see paper"
    # five-year: refit on the same fitting years with a five-year outcome
    d5 = d[(d["months"] >= 60) | (d["mortstat"] == 1)].copy()
    d5["y"] = ((d5["mortstat"] == 1) & (d5["ucod"] == CANCER) & (d5["months"] <= 60)).astype(int)
    t5, e5 = d5[d5["year"] <= 2002].reset_index(drop=True), d5[d5["year"] >= 2005].reset_index(drop=True)
    g0, g2 = Fit(t5, 0), Fit(t5, 2)
    q0, q2 = g0.predict(e5), g2.predict(e5)
    i5 = e5["age"].between(40, 74).to_numpy()
    rob["five_year"] = {"n": int(len(e5)), "cancer_deaths": int(e5["y"].sum()), "auc_F0": round(float(roc_auc_score(e5["y"], q0)), 4), "auc_F2": round(float(roc_auc_score(e5["y"], q2)), 4),
                        "gain_at_20pct": gain_at(e5["y"].to_numpy()[i5], e5["age"].to_numpy()[i5], q2[i5], 0.20)}
    print("  five-year horizon:", rob["five_year"])
    res["robustness"] = rob

    # pooled external cost-matched comparison
    print("\n4. NHANES and NHANES III pooled, cost-matched, ages 40-74")
    tr_e = tr.copy()
    tr_e["smoke_status"] = (tr_e["smoke_status"] > 0).astype(float)
    tr_e["cigs_per_day"], tr_e["quit_years"] = 0.0, 0.0
    e2 = Fit(tr_e, 2)
    xs = []
    for loader in (load_nhanes, load_nhanes3):
        x = loader()
        x = x[x["age"].between(40, 74)].copy()
        x["p"] = e2.predict(x)
        xs.append(x[["age", "y", "p"]])
    pooled = pd.concat(xs, ignore_index=True)
    res["pooled_external"] = {"n": int(len(pooled)), "cancer_deaths": int(pooled["y"].sum()), "shares": {}}
    for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        g = gain_at(pooled["y"].to_numpy(), pooled["age"].to_numpy(), pooled["p"].to_numpy(), share)
        res["pooled_external"]["shares"][str(share)] = g
        print(f"  invite {share:.0%}: age-only {g['age_only']:.1%}, risk-based {g['risk_based']:.1%}, difference {g['diff']:+.1%} [{g['ci'][0]:+.1%}, {g['ci'][1]:+.1%}]")
    res["pooled_external"]["interval_above_zero_everywhere"] = bool(all(v["ci"][0] > 0 for v in res["pooled_external"]["shares"].values()))
    print("  interval above zero at every share:", res["pooled_external"]["interval_above_zero_everywhere"])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
