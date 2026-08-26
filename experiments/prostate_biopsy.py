"""
Can a prostate panel be built that beats PSA on its own?

Why this cohort and not NHANES
------------------------------
NHANES settled the question the other way and settled it hard. With 738 lifetime
prostate cases, bloodwork added -0.000 over age. Doubling the events from 373 to
738 moved nothing, because prostate risk lives in PSA and NHANES excludes men
with a prostate cancer history from the PSA subsample by design.

This cohort has PSA. 212 men with suspected prostate cancer, all of whom went to
transperineal biopsy at one centre between May 2022 and November 2023, with the
pathology result attached. 121 came back adenocarcinoma.

The controls are the important part, exactly as with the ovarian panel: they are
not healthy men, they are men who were suspicious enough to be biopsied and whose
biopsy came back benign. 67 benign prostatic hyperplasia, plus 24 other benign
findings. Separating cancer from BPH in a man already referred for biopsy is the
decision a urologist actually faces. Separating a biopsied man from the general
population is not a decision at all.

Source: Zenodo 10.5281/zenodo.17623285, CC-BY 4.0.

The baseline that matters
-------------------------
Age is not the bar here. PSA is. Any man in this cohort already had his PSA
measured, so a panel that cannot beat reading that one number is worthless
regardless of what it scores against age.

    A  age alone                          the weak baseline
    B  PSA alone                          the baseline that counts
    C  age, PSA, volume, PSA density, BMI  everything from blood and ultrasound
    D  C plus PI-RADS                     adds the MRI read

D is reported separately on purpose. PI-RADS is a radiologist's score off an
MRI, not a lab value, so a panel that needs it is making a different and much
stronger demand of the user. C is the one that fits this application's premise.

Method is paired repeated cross-validation on identical folds, the same arbiter
used for colorectal and the pooled sites.

Run:  python experiments/prostate_biopsy.py
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

REPEATS = 20
OUT = "experiments/prostate_biopsy_result.json"

ARMS = {
    "A age alone":                    ["Age"],
    "B PSA alone":                    ["PSA"],
    "C blood and ultrasound":         ["Age", "PSA", "Prostate_Volume", "PSAD", "BMI"],
    "D plus PI-RADS (MRI)":           ["Age", "PSA", "Prostate_Volume", "PSAD", "BMI", "PI_RADS"],
}

# Malignancy rate among men taken to biopsy. This cohort is 57 percent, which is
# high because it is a referral series enriched by MRI. Published transperineal
# series run nearer 40 percent, and that is the prior used for precision so the
# cohort's own enrichment is not passed off as the world.
BIOPSY_MALIGNANCY = 0.40


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), p


def main():
    d = pd.read_csv("data/prostate_biopsy_zenodo.csv")
    y = (d["Pathology"] == "Adenocarcinoma").astype(int)

    print(f"{len(d)} men biopsied, {int(y.sum())} adenocarcinoma ({y.mean():.1%})")
    print(f"controls: {int((~y.astype(bool)).sum())} benign, of which "
          f"{int((d.Pathology == 'Benign Prostatic Hyperplasia').sum())} BPH\n")

    scores = {}
    for name, feats in ARMS.items():
        X = d[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        aucs = []
        last_p = None
        for seed in range(REPEATS):
            a, p = oof(X, y, seed)
            aucs.append(a)
            if seed == 0:
                last_p = p
        arr = np.array(aucs)
        ci = bootstrap_ci(np.asarray(y), last_p, roc_auc_score)
        scores[name] = arr
        print(f"  {name:<26} AUC {arr.mean():.3f}  (repeat range {arr.min():.3f} "
              f"to {arr.max():.3f})  single-run CI {ci}")

    psa = scores["B PSA alone"]
    results = {}
    print()
    for name in ("C blood and ultrasound", "D plus PI-RADS (MRI)"):
        diff = scores[name] - psa
        wins = int((diff > 0).sum())
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        beats = diff.mean() > 0.01 and wins >= REPEATS * 0.75 and lo > 0
        print(f"  {name} vs PSA alone: {diff.mean():+.3f}  "
              f"(95% range {lo:+.3f} to {hi:+.3f})  wins {wins}/{REPEATS}  "
              f"-> {'BEATS PSA' if beats else 'does not beat PSA'}")
        results[name] = {
            "mean_auc": round(float(scores[name].mean()), 3),
            "gain_over_psa": round(float(diff.mean()), 3),
            "range": [round(lo, 3), round(hi, 3)],
            "wins": wins, "repeats": REPEATS,
            "beats_psa": bool(beats),
        }

    out = {
        "n": int(len(d)), "cases": int(y.sum()),
        "cohort_prevalence": round(float(y.mean()), 3),
        "biopsy_prevalence_used": BIOPSY_MALIGNANCY,
        "age_alone": round(float(scores["A age alone"].mean()), 3),
        "psa_alone": round(float(psa.mean()), 3),
        "arms": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
