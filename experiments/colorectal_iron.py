"""
Can the bowel panel see what KIND of anaemia it is looking at?

The mechanism
-------------
The classic way a right-sided colon cancer announces itself is iron-deficiency
anaemia from slow occult bleeding. That is the textbook presentation, and it is
often the only sign before obstruction.

But "anaemia" on its own is nearly useless as a signal, because most anaemia is
not cancer. What distinguishes iron deficiency is the shape of the red cells:

    iron deficiency          low haemoglobin, LOW MCV, HIGH RDW
    chronic disease          low haemoglobin, normal MCV, normal RDW
    B12 or folate deficiency low haemoglobin, HIGH MCV

The shipped bowel panel read haemoglobin and red cell count and nothing else
from the blood count. It could tell that someone was anaemic. It could not tell
which of those three they were, which is the entire discriminating step.

MCV, RDW, haematocrit, MCH, MPV and neutrophil percentage are printed on the
same complete blood count, on the same sample, at no extra cost. They had simply
never been pulled.

Arms
----
    A  what ships today                     haemoglobin and red cell count
    B  + MCV and RDW                        the two that name the anaemia
    C  + all six red cell and count indices everything on the same CBC

B exists separately from C because if the mechanism above is real then two
variables should carry most of it, and a gain that only appears with six added
is more likely to be capacity than mechanism.

This panel has 96 events in 23,794 people, so the confidence intervals are wide
and a single split means nothing. Judged on repeated paired folds, and a gain is
only accepted if it wins on the great majority of them.

Run:  python experiments/colorectal_iron.py
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

REPEATS = 10
OUT = "experiments/colorectal_iron_result.json"

SHIPPED = ["age", "gender", "wbc", "rbc", "hemoglobin", "platelets", "glucose",
           "calcium", "bun", "creatinine", "protein_total", "albumin", "ast",
           "alt", "bilirubin", "alkaline_phosphatase"]

ARMS = {
    "A shipped today": SHIPPED,
    "B + MCV and RDW": SHIPPED + ["mcv", "rdw"],
    "C + all CBC indices": SHIPPED + ["mcv", "rdw", "hematocrit", "mch", "mpv",
                                      "neutrophil_pct"],
}


def oof_auc(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    df = pd.read_csv("data/nhanes_colorectal.csv")
    y = df["colorectal_cancer"].astype(int).reset_index(drop=True)
    print(f"{len(y):,} adults, {int(y.sum())} colorectal cancers within 8 years "
          f"({y.mean():.2%})\n")

    missing = [c for a in ARMS.values() for c in a if c not in df.columns]
    if missing:
        print(f"cohort is missing {sorted(set(missing))} — rerun "
              f"fetch_nhanes_colorectal.py")
        return

    scores = {}
    for name, feats in ARMS.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        arr = np.array([oof_auc(X, y, s) for s in range(REPEATS)])
        scores[name] = arr
        print(f"  {name:<22} {len(feats):>2} features   AUC {arr.mean():.3f}  "
              f"(range {arr.min():.3f} to {arr.max():.3f})", flush=True)

    base = scores["A shipped today"]
    results, print_lines = {}, []
    for name, arr in scores.items():
        if name == "A shipped today":
            continue
        d = arr - base
        wins = int((d > 0).sum())
        helps = d.mean() > 0.005 and wins >= REPEATS * 0.8
        results[name] = {"auc": round(float(arr.mean()), 3),
                         "gain": round(float(d.mean()), 3),
                         "wins": wins, "repeats": REPEATS, "helps": bool(helps)}
        print_lines.append(f"  {name} vs shipped: {d.mean():+.3f}  "
                           f"wins {wins}/{REPEATS}  -> "
                           f"{'helps' if helps else 'not enough'}")

    print()
    for line in print_lines:
        print(line)

    winners = [k for k, v in results.items() if v["helps"]]
    best = max(results, key=lambda k: results[k]["gain"]) if results else None
    print(f"\n-> {'ADOPT ' + best if winners else 'the red cell indices do not help this panel'}")

    with open(OUT, "w") as f:
        json.dump({"shipped_auc": round(float(base.mean()), 3),
                   "arms": results, "best_arm": best,
                   "adopt": bool(winners)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
