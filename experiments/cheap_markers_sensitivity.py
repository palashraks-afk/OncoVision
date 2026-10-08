"""
Sensitivity analyses for experiments/cheap_markers.py. No bars; exploratory.

Does the picture change if we
  - drop the first 24 months of follow-up (undiagnosed cancer could raise a marker)?
  - look at women and men separately?
  - look at ages 40-59 and 60 and over?
  - use a five-year horizon in all eight cycles, fitted on 1999-2006 and tested on 2007-2014?

Run:  python experiments/cheap_markers_sensitivity.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cheap_markers import CANCER, NAMES, boot_gain, fit_pred, indices, irls, m0

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cheap_markers_sensitivity_result.json")
FOCUS = ["NLR", "SII", "SIRI", "NPAR", "MLR", "PLR", "ALI", "PNI", "RDW"]
NEED = ["LBDNENO", "LBDLYMNO", "LBDMONO", "LBXPLTSI", "LBXNEPCT", "LBXSAL", "LBXRDW", "LBXWBCSI", "LBXHGB", "BMXBMI"]


def load(max_year, horizon):
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan))
    d = d[(d["age"] >= 40) & (d["year0"] <= max_year)].dropna(subset=NEED).reset_index(drop=True)
    d = d[(d["followup_months"] >= horizon) | (d["died"] == 1)].reset_index(drop=True)
    d["ycan"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= horizon)).astype(int)
    d["ync"] = ((d["died"] == 1) & (d["ucod_leading"] != CANCER) & (d["followup_months"] <= horizon)).astype(int)
    idx = indices(d)
    ok = idx.notna().all(axis=1)
    return d[ok].reset_index(drop=True), idx[ok].reset_index(drop=True)


def decedent_or(d, z, mask, lag=0, horizon=120):
    m = mask & (((d["ycan"] + d["ync"]) == 1).to_numpy()) & (d["followup_months"].to_numpy() > lag)
    out = {}
    sm, bm = float(d["smoked"].mean()), float(d["BMXBMI"].mean())
    X0 = m0(d, sm, bm)
    cyc = (d["year0"].to_numpy() - 2003.0) / 4.0
    for name in FOCUS:
        zi = z[name].to_numpy()
        X = np.column_stack([np.ones(m.sum()), X0[m], zi[m], cyc[m]])
        b, se = irls(X, d["ycan"].to_numpy()[m].astype(float))
        j = X0.shape[1] + 1
        out[name] = {"or": round(float(np.exp(b[j])), 3), "ci": [round(float(np.exp(b[j] - 1.96 * se[j])), 3), round(float(np.exp(b[j] + 1.96 * se[j])), 3)]}
    return out, int(m.sum()), int(d["ycan"].to_numpy()[m].sum())


def main():
    d, idx = load(2007, 120)
    tr = d["year0"] <= 2003
    z = (idx - idx[tr].mean()) / idx[tr].std()
    res = {}
    allm = np.ones(len(d), dtype=bool)
    plans = {"all (the pre-registered analysis)": (allm, 0), "first 24 months removed": (allm, 24),
             "women": ((d["gender"] == 0).to_numpy(), 0), "men": ((d["gender"] == 1).to_numpy(), 0),
             "ages 40-59": ((d["age"] < 60).to_numpy(), 0), "ages 60 and over": ((d["age"] >= 60).to_numpy(), 0)}
    print("Decedents within ten years: odds of dying of cancer rather than another cause, per SD (adjusted for baseline and cycle)\n")
    print(f"{'subset':<34}{'decedents':>10}{'cancer':>8}  " + "  ".join(f"{n:>11}" for n in FOCUS))
    for label, (mask, lag) in plans.items():
        o, n, k = decedent_or(d, z, mask, lag)
        res[label] = {"decedents": n, "cancer_deaths": k, "or": o}
        print(f"{label:<34}{n:>10}{k:>8}  " + "  ".join(f"{o[x]['or']:>5.2f}({o[x]['ci'][0]:.2f})" if True else "" for x in FOCUS))

    # five-year horizon, all eight cycles, fit 1999-2006, test 2007-2014
    d5, idx5 = load(2013, 60)
    tr5, te5 = d5["year0"] <= 2005, d5["year0"] >= 2007
    z5 = (idx5 - idx5[tr5].mean()) / idx5[tr5].std()
    sm, bm = float(d5.loc[tr5, "smoked"].mean()), float(d5.loc[tr5, "BMXBMI"].mean())
    X0 = m0(d5, sm, bm)
    yc, yn = d5["ycan"].to_numpy(), d5["ync"].to_numpy()
    p0c, p0n = fit_pred(X0[tr5], yc[tr5], X0[te5]), fit_pred(X0[tr5], yn[tr5], X0[te5])
    five = {"n": int(len(d5)), "cancer_deaths_test": int(yc[te5].sum()), "baseline_cancer_auc": round(float(roc_auc_score(yc[te5], p0c)), 4), "indices": {}}
    print(f"\nFive-year horizon, fit 1999-2006, test 2007-2014 (n={len(d5):,}, test cancer deaths {int(yc[te5].sum())}): baseline AUC {five['baseline_cancer_auc']:.3f}")
    for name in NAMES:
        Xi = np.column_stack([X0, z5[name].to_numpy()])
        pc, pn = fit_pred(Xi[tr5], yc[tr5], Xi[te5]), fit_pred(Xi[tr5], yn[tr5], Xi[te5])
        gc = float(roc_auc_score(yc[te5], pc) - roc_auc_score(yc[te5], p0c))
        gn = float(roc_auc_score(yn[te5], pn) - roc_auc_score(yn[te5], p0n))
        five["indices"][name] = {"gain_cancer": round(gc, 4), "ci_cancer": boot_gain(yc[te5], p0c, pc, 500), "gain_noncancer": round(gn, 4)}
        r = five["indices"][name]
        print(f"   {name:<12} cancer {r['gain_cancer']:+.4f} {r['ci_cancer']}   non-cancer {r['gain_noncancer']:+.4f}")
    res["five_year"] = five
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
