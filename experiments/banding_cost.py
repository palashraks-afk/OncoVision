"""
What does it cost to know age and BMI only as bands?

Why this exists
---------------
The mammogram breast panel reads BCSC's public file, where age arrives as
five-year groups and BMI as four categories, because the consortium bands them
for privacy. The panel therefore treats a 40-year-old and a 44-year-old as the
same woman. Asking the consortium for the research file, which carries exact
values, is one of the open items in docs/DATA_ACCESS_PLAN.md, and it is a
months-long request. This prices it BEFORE anyone makes it.

Method
------
Take cohorts that DO carry exact age and BMI, band them exactly as BCSC does,
and measure what is lost. Anything else about the model is held fixed, so the
difference is the banding and nothing else:

    exact      age (and BMI) as measured
    banded     age -> BCSC's five-year groups, BMI -> BCSC's four categories

Both are scored with repeated paired cross-validation on identical folds, the
project's arbiter for internal questions, and the same two model kinds a panel
chooses between. The answer transfers to the breast panel only as an estimate:
these are different outcomes in a different cohort. It bounds the size of the
prize rather than predicting it.

Run:  python experiments/banding_cost.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/banding_cost_result.json"
REPEATS, FOLDS = 5, 5

# BCSC's published bands, from the public risk-factor file's codebook: age in
# five-year groups from 35, BMI in four categories.
AGE_EDGES = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 200]
BMI_EDGES = [0, 25, 30, 35, 200]

COHORTS = {
    "general": {"file": "nhanes_general_multicycle.csv", "target": "any_cancer",
                "cols": ["age", "bmi"]},
    "colorectal": {"file": "nhanes_colorectal.csv", "target": "colorectal_cancer",
                   "cols": ["age"]},
}


def band(series, edges):
    """Replace each value with the midpoint of its band, as the panel does."""
    idx = np.digitize(series.to_numpy(dtype=float), edges[1:-1], right=False)
    lows = np.array(edges[:-1], dtype=float)
    highs = np.array(edges[1:], dtype=float)
    mids = np.where(np.isfinite(highs) & (highs < 200), (lows + highs) / 2.0, lows + 2.5)
    out = pd.Series(mids[idx], index=series.index)
    return out.where(series.notna())


def paired_cv(X_exact, X_banded, y, kind):
    exact, banded = [], []
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X_exact, y):
            for X, bucket in ((X_exact, exact), (X_banded, banded)):
                m = tm.model_factory(kind, len(tr), float(y.iloc[tr].mean()))
                m.fit(X.iloc[tr], y.iloc[tr])
                bucket.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(exact), np.array(banded)


def main():
    results = {}
    print("What banding age and BMI costs, measured where exact values exist\n")
    for name, spec in COHORTS.items():
        df = pd.read_csv(os.path.join(tm.DATA_DIR, spec["file"]))
        keep = df[spec["cols"] + [spec["target"]]].notna().all(axis=1)
        df = df[keep]
        y = df[spec["target"]].astype(int).reset_index(drop=True)
        X_exact = df[spec["cols"]].reset_index(drop=True).astype(float)
        X_banded = X_exact.copy()
        X_banded["age"] = band(X_banded["age"], AGE_EDGES)
        if "bmi" in X_banded.columns:
            X_banded["bmi"] = band(X_banded["bmi"], BMI_EDGES)

        per_kind = {}
        for kind in ("logistic", "ensemble"):
            e, b = paired_cv(X_exact, X_banded, y, kind)
            d = e - b
            per_kind[kind] = {
                "exact_auc": round(float(e.mean()), 4),
                "banded_auc": round(float(b.mean()), 4),
                "cost": round(float(d.mean()), 4),
                "cost_ci": [round(float(np.percentile(d, 2.5)), 4),
                            round(float(np.percentile(d, 97.5)), 4)],
                "banded_loses_in": f"{int((d > 0).sum())}/{len(d)}",
            }
            print(f"  {name:<11} {kind:<9} exact {e.mean():.4f}  banded {b.mean():.4f}  "
                  f"cost {d.mean():+.4f}  banded worse in {int((d > 0).sum())}/{len(d)} folds")
        results[name] = {"n": int(len(y)), "events": int(y.sum()),
                         "features": spec["cols"], "kinds": per_kind}

    costs = [k["cost"] for r in results.values() for k in r["kinds"].values()]
    results["summary"] = {
        "worst_case_cost": round(float(max(costs)), 4),
        "typical_cost": round(float(np.median(costs)), 4),
        "verdict": ("Exact values are worth chasing" if max(costs) >= 0.01 else
                    "Banding costs less than 0.01 AUC in every cohort and model tested"),
    }
    print(f"\n  worst case {max(costs):+.4f}, typical {np.median(costs):+.4f} AUC")
    print(f"  {results['summary']['verdict']}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
