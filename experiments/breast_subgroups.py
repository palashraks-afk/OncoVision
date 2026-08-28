"""
Subgroup measurement for the breast panel, which has no demographics at all.

The gap this closes
-------------------
Every other panel reports AUC broken out by sex and age band. The breast panel
reports nothing, because the Wisconsin cohort carries no age, no sex and no race
columns. That has been listed as "zero subgroup measurement" for a long time and
treated as unfixable, which is only half true: the demographic breakdown is
genuinely impossible, but "does this panel work equally well across the cases it
will actually see" is answerable from what the cohort does contain.

What is measured instead
------------------------
Lesion size, using nuclear area, split at its tertiles. This is the clinically
meaningful axis here. A panel that only separates malignant from benign when the
nuclei are already large is far less useful than one that works on small
lesions, because the small ones are the ones where an earlier answer changes
anything.

Nuclear texture is added as a second axis, since it is the feature least
correlated with size, so it tests a different kind of hard case.

Honest about what this is not
-----------------------------
This does not tell anyone whether the panel works equally well across race or
age. It cannot, and no amount of analysis of this cohort will, because the
columns do not exist. That limitation stands and is stated on the panel. What
this replaces is the previous situation of having no subgroup evidence of any
kind.

Run:  python experiments/breast_subgroups.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUT = "experiments/breast_subgroups_result.json"


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == "breast")
    X, y, _ = tm.prepare(cfg)
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X = X.fillna(X.median())
    y = pd.Series(y).astype(int).reset_index(drop=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]

    overall = roc_auc_score(y, p)
    print(f"breast panel, {len(y)} aspirates, {int(y.sum())} malignant")
    print(f"overall out-of-fold AUC {overall:.3f}\n")
    print("The Wisconsin cohort has no age, sex or race column, so the usual")
    print("demographic breakdown is impossible. These are the axes it does have.\n")

    results = {"overall_auc": round(float(overall), 3), "n": int(len(y)),
               "cases": int(y.sum()), "subgroups": {}}

    for feat, label in [("area_mean", "lesion size (nuclear area)"),
                        ("texture_mean", "nuclear texture")]:
        lo, hi = X[feat].quantile([1 / 3, 2 / 3])
        bands = [
            (f"{label}: smallest third", X[feat] <= lo),
            (f"{label}: middle third", (X[feat] > lo) & (X[feat] <= hi)),
            (f"{label}: largest third", X[feat] > hi),
        ]
        print(f"  by {label}:")
        for name, mask in bands:
            yy, pp = y[mask], p[mask.values]
            if yy.nunique() < 2:
                print(f"    {name:<44} only one class present, cannot score")
                results["subgroups"][name] = {"n": int(mask.sum()), "auc": None,
                                              "note": "only one class present"}
                continue
            a = roc_auc_score(yy, pp)
            ci = bootstrap_ci(np.asarray(yy), pp, roc_auc_score)
            print(f"    {name:<44} n={int(mask.sum()):<4} malignant={int(yy.sum()):<4} "
                  f"AUC {a:.3f}  (95% CI {ci[0]} to {ci[1]})")
            results["subgroups"][name] = {"n": int(mask.sum()), "cases": int(yy.sum()),
                                          "auc": round(float(a), 3), "auc_ci": ci}
        print()

    scored = [v["auc"] for v in results["subgroups"].values() if v.get("auc") is not None]
    if scored:
        spread = max(scored) - min(scored)
        results["worst_subgroup_auc"] = min(scored)
        results["subgroup_spread"] = round(float(spread), 3)
        print(f"worst subgroup {min(scored):.3f}, best {max(scored):.3f}, "
              f"spread {spread:.3f}")
        if min(scored) < 0.8:
            print("The panel is materially weaker in at least one subgroup, which belongs")
            print("on the panel rather than in a footnote.")
        else:
            print("No subgroup collapses. The panel holds across lesion size and texture.")

    results["cannot_measure"] = ("Race, age and sex. The Wisconsin cohort does not record "
                                 "them, so accuracy across those groups is unmeasured "
                                 "rather than acceptable, and no analysis of this cohort "
                                 "can change that.")
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
