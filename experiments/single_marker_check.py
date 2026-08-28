"""
Does each panel beat its own single best marker?

Why this exists
---------------
The prostate panel was rejected partly because it could not beat PSA on its own.
That is a fair bar, but it was only ever applied to prostate, and a standard
applied to one panel and not the others is not a standard. If the pancreatic
panel turns out to be CA 19-9 wearing a hat, that has to be said out loud in the
same breath.

So every shipped panel that has an obvious dominant marker is tested the same
way: the full feature set against that one number, on identical folds.

    pancreatic   CA 19-9
    ovarian      CA 125, and CA 125 with HE4, the clinical ROMA pair
    liver        AFP is not in the shipped liver feature set, so ALT and AST
                 stand in as the obvious single reads
    breast       nuclear area, the strongest single Wisconsin separator
    colorectal   haemoglobin, the marker the ColonFlag idea rests on

A panel that merely matches its best marker is not necessarily worthless, but
the user could get the same answer by reading one line of their lab report, and
they deserve to know that.

Run:  python experiments/single_marker_check.py
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
OUT = "experiments/single_marker_check_result.json"

# panel -> list of (label, feature subset) to test against the full panel
MARKERS = {
    "pancreatic": [("CA 19-9 alone", ["plasma_ca19_9"])],
    "ovarian":    [("CA 125 alone", ["ca125"]),
                   ("CA 125 + HE4 (ROMA pair)", ["ca125", "he4"])],
    "liver":      [("ALT + AST alone", ["alt", "ast"])],
    "breast":     [("nuclear area alone", ["area_mean"])],
    "colorectal": [("haemoglobin alone", ["hemoglobin"])],
    "prostate":   [("PSA alone", ["psa"]), ("PI-RADS alone", ["pi_rads"])],
    "lung":       [("serum cotinine alone", ["cotinine"]),
                   ("age, sex, smoking", ["age", "gender", "smoking"])],
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN or name not in MARKERS:
            continue
        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        full = np.array([oof(X, y, s) for s in range(REPEATS)])
        print(f"\n=== {name} ===  {len(y)} rows, {int(y.sum())} cases")
        print(f"  full panel ({X.shape[1]} features)   AUC {full.mean():.3f}", flush=True)

        entry = {"full_auc": round(float(full.mean()), 3), "comparisons": {}}
        for label, feats in MARKERS[name]:
            feats = [f for f in feats if f in X.columns]
            if not feats:
                print(f"  {label}: features not in this panel, skipped")
                continue
            one = np.array([oof(X[feats], y, s) for s in range(REPEATS)])
            d = full - one
            wins = int((d > 0).sum())
            beats = d.mean() > 0.01 and wins >= REPEATS * 0.75 and d.min() > 0
            print(f"  {label:<28} AUC {one.mean():.3f}   full beats it by "
                  f"{d.mean():+.3f}  wins {wins}/{REPEATS}  "
                  f"-> {'panel adds real value' if beats else 'PANEL MATCHES ONE MARKER'}")
            entry["comparisons"][label] = {
                "marker_auc": round(float(one.mean()), 3),
                "panel_gain": round(float(d.mean()), 3),
                "wins": wins, "repeats": REPEATS,
                "panel_beats_marker": bool(beats),
            }
        results[name] = entry

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
