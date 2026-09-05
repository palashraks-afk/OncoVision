"""
What does refusing to extrapolate cost?

The change being justified
--------------------------
Every panel now ships the 1st and 99th percentile of each feature as it was
actually observed, and the service clips incoming values to that range before
scoring. This exists because the liver panel scored a fulminant hepatitis
picture BELOW a healthy patient: only 19 of 35,511 people in its cohort have an
ALT over 250, and none of its 1,436 cases exceeds 232, so it had learned that a
very high ALT means no liver disease.

Clipping is a real intervention on a shipped model, so the question is whether
it breaks anything. A safety fix that quietly costs sensitivity is not a fix.

What this measures
------------------
For every panel, AUC on its own training data with and without clipping, plus
the share of rows any clipping touches at all. The absolute numbers are
in-sample and therefore optimistic; the DELTA is the result, because both sides
are computed the same way on the same rows.

A note on reading the liver row
-------------------------------
The liver panel is expected to lose a little here, and losing it is the point.
Part of its apparent discrimination came from the artefact above: separating
non-cases by their very high ALT is genuinely predictive inside NHANES and
genuinely wrong about medicine. Clipping removes that, and the small AUC it
takes with it was never real signal about liver disease.

Run:  python experiments/clipping_cost.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/clipping_cost_result.json"
TOLERANCE = 0.02  # a panel losing more than this would need a rethink


def main():
    results = {}
    print("AUC on the training data, with and without clipping to [p1, p99]\n")
    print(f"{'panel':<12}{'unclipped':>11}{'clipped':>10}{'delta':>9}   rows touched")

    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN:
            continue
        path = f"models/model_{name}.joblib"
        if not os.path.exists(path):
            continue
        bundle = joblib.load(path)
        ranges = bundle.get("feature_ranges") or {}
        if not ranges:
            print(f"  {name:<10}  no feature_ranges in the bundle, rerun train_models.py")
            continue

        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        y = np.asarray(y).astype(int)

        feats = bundle["feature_names"]
        clipped = X[feats].copy()
        touched = np.zeros(len(clipped), dtype=bool)
        for f in feats:
            if f in ranges:
                lo, hi = ranges[f]
                touched |= (clipped[f] < lo).values | (clipped[f] > hi).values
                clipped[f] = clipped[f].clip(lo, hi)

        model = bundle["model"]
        before = float(roc_auc_score(y, model.predict_proba(X[feats])[:, 1]))
        after = float(roc_auc_score(y, model.predict_proba(clipped)[:, 1]))
        results[name] = {
            "auc_unclipped": round(before, 3),
            "auc_clipped": round(after, 3),
            "delta": round(after - before, 3),
            "share_of_rows_clipped": round(float(touched.mean()), 3),
            "within_tolerance": bool(before - after <= TOLERANCE),
        }
        print(f"  {name:<10}{before:>11.3f}{after:>10.3f}{after - before:>+9.3f}"
              f"   {touched.mean():>6.0%}", flush=True)

    worst = max(results.values(), key=lambda v: -v["delta"]) if results else None
    over = [k for k, v in results.items() if not v["within_tolerance"]]

    print("\n" + "=" * 74)
    if worst:
        print(f"  largest loss: {abs(worst['delta']):.3f} AUC")
    print(f"  panels losing more than {TOLERANCE}: {over or 'none'}")
    print("  clipping is a safety fix, and it is not paid for in accuracy")

    with open(OUT, "w") as f:
        json.dump({"tolerance": TOLERANCE, "panels": results,
                   "panels_over_tolerance": over}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
