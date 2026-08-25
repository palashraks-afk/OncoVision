"""
How much of each panel's published AUC is the split rather than the model?

The problem this exists to fix
------------------------------
The cervical panel reported a held-out AUC of 0.725 while cross-validating at
0.587. A gap of 0.138 on 55 positive biopsies does not mean the model is good
and the CV is pessimistic. It means one arbitrary 80/20 split happened to be
favourable, and that split was deciding a number published on the interface.

Every other panel agreed with itself to within about 0.03, so the problem was
specific, but the METHOD was wrong everywhere: a single split is a single draw
from a distribution, and reporting it as though it were the estimate hides how
wide that distribution is.

What this does instead
----------------------
Repeated stratified hold-out. The same 80/20 protocol the project already uses,
drawn many times with different seeds, refitting from scratch each time. That
gives three things a single split cannot:

    mean        the estimate that does not depend on a lucky seed
    spread      how much the split alone moves the number
    shipped     where the seed-42 split the project publishes actually falls

If the shipped split sits near the mean, the published number is honest. If it
sits in the tail, the published number is an artifact and has to be replaced.

Repeat counts are scaled to cohort size because liver and general are large and
slow, and because large cohorts have far less split-to-split variance anyway,
which is exactly the effect being measured.

Run:  python experiments/split_stability.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

SHIPPED_SEED = 42
OUT = "experiments/split_stability_result.json"

# Large cohorts are slow and, being large, vary far less between splits.
def repeats_for(n_rows: int) -> int:
    if n_rows > 20_000:
        return 5
    if n_rows > 5_000:
        return 10
    return 30


def fit(X, y, seed):
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    ).fit(X, y)


def one_split(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    med = X_tr.median()
    X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
    model = fit(X_tr, y_tr, seed)
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def main():
    results = {}
    for config in tm.DATASETS:
        name = config["name"]
        if name in tm.WITHDRAWN:
            continue

        X, y, _ = tm.prepare(config)
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.Series(y).astype(int).reset_index(drop=True)
        X = X.reset_index(drop=True)

        n = repeats_for(len(y))
        print(f"{name}: {len(y)} rows, {int(y.sum())} positive, {n} repeated splits ...",
              flush=True)

        aucs = []
        for seed in range(n):
            try:
                aucs.append(one_split(X, y, seed))
            except Exception as exc:
                print(f"    seed {seed} failed: {str(exc)[:60]}")
        if not aucs:
            continue

        shipped = one_split(X, y, SHIPPED_SEED)
        arr = np.array(aucs)
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        # Where the shipped split falls inside the distribution of splits.
        pct = float((arr < shipped).mean() * 100)

        results[name] = {
            "n_rows": int(len(y)), "n_positive": int(y.sum()), "n_splits": len(aucs),
            "mean_auc": round(float(arr.mean()), 3),
            "median_auc": round(float(np.median(arr)), 3),
            "std_auc": round(float(arr.std()), 3),
            "min_auc": round(float(arr.min()), 3),
            "max_auc": round(float(arr.max()), 3),
            "split_range_2p5_97p5": [round(lo, 3), round(hi, 3)],
            "shipped_split_auc": round(shipped, 3),
            "shipped_split_percentile": round(pct, 1),
            "shipped_minus_mean": round(shipped - float(arr.mean()), 3),
        }
        r = results[name]
        print(f"    mean {r['mean_auc']}  sd {r['std_auc']}  "
              f"range {r['min_auc']} to {r['max_auc']}  "
              f"shipped {r['shipped_split_auc']} at the "
              f"{r['shipped_split_percentile']:.0f}th percentile "
              f"({r['shipped_minus_mean']:+.3f} vs mean)", flush=True)

    print("\n" + "=" * 100)
    print(f"{'panel':<12}{'rows':>7}{'pos':>7}{'mean':>8}{'sd':>7}"
          f"{'split spread':>16}{'shipped':>9}{'pctile':>8}{'verdict':>16}")
    print("=" * 100)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_auc"]):
        # A shipped split above the 90th percentile of its own split
        # distribution is a lucky draw, not an estimate.
        verdict = "LUCKY SPLIT" if r["shipped_split_percentile"] >= 90 else "representative"
        spread = f"{r['min_auc']} to {r['max_auc']}"
        print(f"{name:<12}{r['n_rows']:>7}{r['n_positive']:>7}{r['mean_auc']:>8}"
              f"{r['std_auc']:>7}{spread:>16}{r['shipped_split_auc']:>9}"
              f"{r['shipped_split_percentile']:>7.0f}%{verdict:>16}")

    with open(OUT, "w") as f:
        json.dump({"shipped_seed": SHIPPED_SEED, "panels": results}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
