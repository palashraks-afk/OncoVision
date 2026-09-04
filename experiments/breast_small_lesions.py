"""
Can the breast panel be made to work on small lesions?

The weakness
------------
The breast panel scores 0.954 overall and 0.739 on the smallest third of lesions
by nuclear area, where only 6 of 190 are malignant. It is weakest exactly where
an earlier answer would change something, because a large obvious mass is not
the case anybody needs help with.

Why it might be fixable
-----------------------
The panel currently reads four of the thirty Wisconsin measurements: the mean
radius, texture, perimeter and area. That restriction came from this project's
rule that a model may only use what the application can collect, and at the time
the application asked for four numbers.

That rule has since been sharpened. Breast is now labelled an INTERPRETATION
panel: it requires a fine needle aspirate that has already been taken and
imaged. Someone holding that report is not holding four numbers. A real
pathology workup carries the shape descriptors too, and the "worst" variants,
which are the largest values across the nuclei rather than the average.

Small lesions are precisely where an average should struggle and a worst-case
descriptor should not: a small mass with one very irregular nucleus is the hard
case, and averaging hides it.

Arms
----
    A  the four shipped means
    B  all ten mean measurements
    C  means plus the ten "worst" measurements
    D  all thirty

Judged on the smallest-third subgroup, not on the overall figure, because the
overall figure is already 0.954 and improving it is not the point. Any arm that
wins overall while leaving small lesions where they are has not fixed anything.

Run:  python experiments/breast_small_lesions.py
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

REPEATS = 10
OUT = "experiments/breast_small_lesions_result.json"

MEANS = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
         "smoothness_mean", "compactness_mean", "concavity_mean",
         "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]
WORST = [c.replace("_mean", "_worst") for c in MEANS]
SE = [c.replace("_mean", "_se") for c in MEANS]

ARMS = {
    "A four shipped means": MEANS[:4],
    "B all ten means": MEANS,
    "C means + worst": MEANS + WORST,
    "D all thirty": MEANS + WORST + SE,
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]


def main():
    d = pd.read_csv("data/data.csv")
    y = (d["diagnosis"].astype(str).str.upper() == "M").astype(int).reset_index(drop=True)

    # Same subgroup definition as experiments/breast_subgroups.py.
    area = pd.to_numeric(d["area_mean"], errors="coerce").reset_index(drop=True)
    lo = area.quantile(1 / 3)
    small = (area <= lo).values

    print(f"{len(y)} aspirates, {int(y.sum())} malignant")
    print(f"smallest third by nuclear area: {int(small.sum())} lesions, "
          f"{int(y[small].sum())} malignant\n")

    results = {}
    for name, feats in ARMS.items():
        X = d[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        overall, small_auc = [], []
        last = None
        for s in range(REPEATS):
            p = oof(X, y, s)
            overall.append(roc_auc_score(y, p))
            small_auc.append(roc_auc_score(y[small], p[small]))
            if s == 0:
                last = p
        o, sm = float(np.mean(overall)), float(np.mean(small_auc))
        ci = bootstrap_ci(np.asarray(y[small]), last[small], roc_auc_score)
        results[name] = {"features": len(feats), "overall_auc": round(o, 3),
                         "small_lesion_auc": round(sm, 3), "small_lesion_ci": ci}
        print(f"  {name:<24} {len(feats):>2} features   overall {o:.3f}   "
              f"SMALL LESIONS {sm:.3f}  (CI {ci[0]} to {ci[1]})", flush=True)

    base = results["A four shipped means"]
    best = max(results, key=lambda k: results[k]["small_lesion_auc"])
    gain = results[best]["small_lesion_auc"] - base["small_lesion_auc"]
    print("\n" + "=" * 82)
    print(f"shipped today       small lesions {base['small_lesion_auc']}")
    print(f"best arm            small lesions {results[best]['small_lesion_auc']}   {best}")
    print(f"improvement         {gain:+.3f}")
    adopt = gain >= 0.02
    print(f"-> {'ADOPT, and the panel asks for more of the report' if adopt else 'no real improvement available'}")

    with open(OUT, "w") as f:
        json.dump({"arms": results, "best_arm": best,
                   "small_lesion_gain": round(float(gain), 3),
                   "adopt": bool(adopt)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
