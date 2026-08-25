"""
With ten cycles pooled, do lung, colorectal, prostate or breast beat their baseline?

Pooling 1999 to 2018 doubled the event counts and put all four sites over the
roughly 96 event floor. That makes them testable. It does not make them good,
and this decides which.

Method
------
Paired repeated cross-validation, the same arbiter that settled colorectal when
its single held-out split disagreed with its cross-validation. Both arms see
identical folds every repeat, so the difference is the model and not the
shuffle. A single split is one draw and is not allowed to decide anything here.

Baselines are chosen per site to be the honest thing to beat:

    lung         age, sex and smoking. Smoking is the overwhelming cause, and a
                 model that cannot beat "does this person smoke" is worthless.
    colorectal   age and sex.
    prostate     age. Male only.
    breast       age. Female only.

The bar, applied identically to all four: the mean difference must exceed 0.01,
the model must win at least three quarters of repeats, and the 95 percent range
of the difference must exclude zero.

What a positive result would and would not mean
------------------------------------------------
The target here is a LIFETIME diagnosis, because pooling all ten cycles means
giving up the age-at-diagnosis question that only five cycles asked. So a person
treated twenty years ago counts as positive. That inflates any apparent signal
with survivorship and with the aftermath of treatment, and it is the same
weakness that made the old general panel mostly an age model.

That is why beating the baseline here is necessary but not sufficient, and why
the recency-windowed result stays the one that ships wherever it exists.

Run:  python experiments/pooled_sites.py
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
OUT = "experiments/pooled_sites_result.json"

BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "rdw", "mpv",
         "glucose", "calcium", "bun", "creatinine", "protein_total", "albumin",
         "ast", "alt", "bilirubin", "alkaline_phosphatase"]

SITES = {
    "lung":       (["age", "gender", "smoking"], None),
    "colorectal": (["age", "gender"], None),
    "prostate":   (["age"], 1.0),      # men only
    "breast":     (["age"], 0.0),      # women only
}


def oof_auc(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    df = pd.read_csv("data/nhanes_sites_pooled.csv")
    results = {}

    for site, (base, sex) in SITES.items():
        sub = df[df[site].notna()].copy()
        if sex is not None:
            sub = sub[sub["gender"] == sex]
        y = sub[site].astype(int).reset_index(drop=True)
        X = sub[base + BLOOD].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())

        print(f"=== {site} ===  {len(y)} adults, {int(y.sum())} cases "
              f"({y.mean():.2%}), baseline = {', '.join(base)}", flush=True)

        full, bl, diff = [], [], []
        for seed in range(REPEATS):
            a_full = oof_auc(X, y, seed)
            a_base = oof_auc(X[base], y, seed)
            full.append(a_full); bl.append(a_base); diff.append(a_full - a_base)
            print(f"    repeat {seed}: model {a_full:.3f}  baseline {a_base:.3f}  "
                  f"{a_full - a_base:+.3f}", flush=True)

        d = np.array(diff)
        wins = int((d > 0).sum())
        lo, hi = (float(np.min(d)), float(np.max(d))) if REPEATS < 10 else \
                 tuple(np.percentile(d, [2.5, 97.5]))
        ships = d.mean() > 0.01 and wins >= REPEATS * 0.75 and lo > 0
        print(f"  model {np.mean(full):.3f}   baseline {np.mean(bl):.3f}   "
              f"mean diff {d.mean():+.3f}  (range {lo:+.3f} to {hi:+.3f})  "
              f"wins {wins}/{REPEATS}")
        print(f"  VERDICT: {'BLOODWORK HELPS' if ships else 'bloodwork adds nothing'}\n")

        results[site] = {
            "n": int(len(y)), "cases": int(y.sum()),
            "baseline_features": base,
            "model_mean": round(float(np.mean(full)), 3),
            "baseline_mean": round(float(np.mean(bl)), 3),
            "mean_difference": round(float(d.mean()), 3),
            "difference_range": [round(lo, 3), round(hi, 3)],
            "wins": wins, "repeats": REPEATS,
            "verdict": "helps" if ships else "adds nothing",
        }

    print("=" * 92)
    print(f"{'site':<12}{'cases':>7}{'baseline':>10}{'model':>8}{'gain':>9}{'wins':>7}   verdict")
    print("=" * 92)
    for s, r in results.items():
        print(f"{s:<12}{r['cases']:>7}{r['baseline_mean']:>10}{r['model_mean']:>8}"
              f"{r['mean_difference']:>+9.3f}{r['wins']:>4}/{r['repeats']}   {r['verdict']}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
