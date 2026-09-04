"""
Does the liver panel improve if it reads the rest of the same blood test?

The gap
-------
The liver panel reads bilirubin, alkaline phosphatase, ALT, AST, total protein
and albumin, plus age, sex and risk history. Six chemistry values.

NHANES draws a full comprehensive metabolic panel on the same sample, and four
more of its analytes were simply never pulled:

    GGT         the one a hepatologist asks for first, because it confirms that
                a raised alkaline phosphatase came from the liver and not bone
    globulin    total protein minus albumin, and the albumin-globulin ratio is
                a standard read on chronic liver disease
    LDH         released by damaged tissue generally
    uric acid   handled by the liver, and raised in several liver conditions

None of these costs the patient anything extra. They are already on the
requisition. Leaving them out was an omission rather than a decision, and this
measures whether correcting it buys anything.

Arms
----
    A  what ships today
    B  + GGT
    C  + GGT and globulin
    D  + all four

Judged on repeated paired folds on identical splits, which is the arbiter used
everywhere else in this project. The German cohort is not used as a check here
because it records none of these four analytes, so it cannot speak to them.

Run:  python experiments/liver_extra_analytes.py
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

warnings.filterwarnings("ignore")

REPEATS = 5
OUT = "experiments/liver_extra_analytes_result.json"

SHIPPED = ["age", "gender", "bilirubin", "alkaline_phosphatase", "alt", "ast",
           "protein_total", "albumin", "diabetes", "hepatitis_b", "hepatitis_c"]

ARMS = {
    "A shipped today": SHIPPED,
    "B + GGT": SHIPPED + ["ggt"],
    "C + GGT, globulin": SHIPPED + ["ggt", "globulin"],
    "D + all four": SHIPPED + ["ggt", "globulin", "ldh", "uric_acid"],
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    df = pd.read_csv("data/nhanes_liver_multicycle.csv")
    y = df["liver_disease"].astype(int).reset_index(drop=True)
    print(f"{len(y)} NHANES adults, {int(y.sum())} with a liver condition\n")

    scores = {}
    for name, feats in ARMS.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        arr = np.array([oof(X, y, s) for s in range(REPEATS)])
        scores[name] = arr
        print(f"  {name:<22} {len(feats):>2} features   AUC {arr.mean():.3f}  "
              f"(range {arr.min():.3f} to {arr.max():.3f})", flush=True)

    base = scores["A shipped today"]
    results = {}
    print()
    for name, arr in scores.items():
        if name == "A shipped today":
            continue
        d = arr - base
        wins = int((d > 0).sum())
        good = d.mean() > 0.005 and wins >= REPEATS * 0.75 and d.min() > 0
        print(f"  {name} vs shipped: {d.mean():+.3f}  wins {wins}/{REPEATS}"
              f"  -> {'helps' if good else 'not enough'}")
        results[name] = {"auc": round(float(arr.mean()), 3),
                         "gain": round(float(d.mean()), 3),
                         "wins": wins, "repeats": REPEATS, "helps": bool(good)}

    best = max(results, key=lambda k: results[k]["gain"])
    adopt = results[best]["helps"]
    print(f"\nbest arm: {best} ({results[best]['gain']:+.3f})")
    print(f"-> {'ADOPT' if adopt else 'nothing worth adding'}")

    with open(OUT, "w") as f:
        json.dump({"shipped_auc": round(float(base.mean()), 3),
                   "arms": results, "best_arm": best, "adopt": bool(adopt)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
