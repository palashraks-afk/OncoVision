"""
Does the colorectal panel actually beat age and sex, or does it just look like it?

The conflict this resolves
--------------------------
Two measurements disagree.

    cross-validation, all 96 events   model 0.804, age and sex 0.774   +0.030
    single held-out split, ~19 events model 0.793, age and sex 0.817   -0.024

One of those says ship and the other says the baseline wins. They cannot both
be the answer, and picking whichever one flatters the panel is exactly the
behaviour this project exists to avoid.

The held-out split is the weaker evidence here, and not because it is
inconvenient. Colorectal has 96 events in total, so a 20 percent test split
holds roughly 19 of them. An AUC estimated on 19 events has an interval running
0.708 to 0.867, which is wide enough to contain both claims at once. It cannot
settle the question.

So settle it the way the question deserves: repeat the whole comparison many
times, on identical folds, and look at how often the model actually wins.

    paired    both arms see exactly the same fold assignment every repeat, so
              the difference is the model and not the shuffle
    repeats   many, because one number from one split is what caused this mess
    reported  the distribution of the difference, not a single figure

If the model wins on most repeats and the mean difference is positive, it
ships. If it does not, colorectal is an age model and gets withdrawn, the same
way prostate did.

Run:  python experiments/colorectal_vs_age.py
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

REPEATS = 20
OUT = "experiments/colorectal_vs_age_result.json"

BASELINE = ["age", "gender"]


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    config = next(c for c in tm.DATASETS if c["name"] == "colorectal")
    X, y, _ = tm.prepare(config)
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X = X.fillna(X.median())
    y = pd.Series(y).astype(int).reset_index(drop=True)

    print(f"{len(y)} adults, {int(y.sum())} colorectal cases ({y.mean():.2%})")
    print(f"{REPEATS} paired repeats, identical folds for both arms\n")

    full, base, diff = [], [], []
    for seed in range(REPEATS):
        a_full = oof(X, y, seed)
        a_base = oof(X[BASELINE], y, seed)
        full.append(a_full)
        base.append(a_base)
        diff.append(a_full - a_base)
        print(f"  repeat {seed:>2}: model {a_full:.3f}   age+sex {a_base:.3f}   "
              f"{a_full - a_base:+.3f}", flush=True)

    d = np.array(diff)
    wins = int((d > 0).sum())
    lo, hi = np.percentile(d, [2.5, 97.5])

    print("\n" + "=" * 74)
    print(f"model mean            {np.mean(full):.3f}")
    print(f"age and sex mean      {np.mean(base):.3f}")
    print(f"mean difference       {d.mean():+.3f}   (95% range {lo:+.3f} to {hi:+.3f})")
    print(f"model wins            {wins} of {REPEATS} repeats")
    print("=" * 74)

    ships = d.mean() > 0.01 and wins >= REPEATS * 0.75 and lo > 0
    print(f"\nVERDICT: {'SHIP' if ships else 'WITHDRAW, it is an age model'}")
    if not ships:
        print("The bloodwork does not reliably add anything over knowing age and sex.")

    with open(OUT, "w") as f:
        json.dump({
            "repeats": REPEATS, "n": int(len(y)), "events": int(y.sum()),
            "model_mean": round(float(np.mean(full)), 3),
            "age_sex_mean": round(float(np.mean(base)), 3),
            "mean_difference": round(float(d.mean()), 3),
            "difference_range": [round(float(lo), 3), round(float(hi), 3)],
            "model_wins": wins,
            "verdict": "ship" if ships else "withdraw",
        }, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
