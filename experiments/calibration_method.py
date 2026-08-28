"""
Is isotonic calibration hurting the small panels?

The symptom
-----------
On the held-out split the prostate panel scores 0.840 while plain logistic
regression scores 0.876. The shipped model is already logistic, so the gap is
not the algorithm, it is what happens to it afterwards: isotonic calibration
fitted inside 169 training rows.

Isotonic regression is non-parametric and needs data. On small samples it
overfits the calibration curve and can flatten genuine ranking differences into
ties, which costs AUC. Platt scaling, a single sigmoid with two parameters, is
the standard choice when events are few.

Every panel is calibrated the same way today, so this checks all of them rather
than special-casing prostate after the fact.

    none       raw model output, no calibration
    isotonic   what ships today
    sigmoid    Platt scaling

Judged on out-of-fold AUC over repeated splits, plus Brier score, because
calibration exists to make the printed percentage mean something and it would
be a poor trade to win AUC and lose that.

Run:  python experiments/calibration_method.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

REPEATS = 5
OUT = "experiments/calibration_method_result.json"


def score(X, y, method, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    base = tm.build_ensemble(len(y), float(y.mean()))
    if method == "none":
        est = base
    else:
        est = CalibratedClassifierCV(base, method=method, cv=cv)
    p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), float(brier_score_loss(y, p))


def main():
    # Only the small panels. Isotonic regression needs data to fit its step
    # function, so the risk of it overfitting is a small-sample risk. The NHANES
    # panels carry 20,000 rows and more and are not in question here.
    SMALL = {"prostate", "ovarian", "breast", "pancreatic", "colorectal"}
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN or name not in SMALL:
            continue
        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        print(f"\n=== {name} ===  {len(y)} rows, {int(y.sum())} events", flush=True)
        entry = {"n": int(len(y)), "events": int(y.sum()), "methods": {}}
        for method in ("none", "isotonic", "sigmoid"):
            aucs, briers = [], []
            for s in range(REPEATS):
                a, b = score(X, y, method, s)
                aucs.append(a); briers.append(b)
            a, b = float(np.mean(aucs)), float(np.mean(briers))
            entry["methods"][method] = {"auc": round(a, 3), "brier": round(b, 4)}
            print(f"  {method:<10} AUC {a:.3f}   Brier {b:.4f}", flush=True)

        best = max(entry["methods"], key=lambda m: entry["methods"][m]["auc"])
        cur = entry["methods"]["isotonic"]["auc"]
        gain = entry["methods"][best]["auc"] - cur
        entry["best_by_auc"] = best
        entry["gain_over_isotonic"] = round(gain, 3)
        # Only worth switching if it is a real margin, not noise.
        entry["switch"] = bool(best != "isotonic" and gain >= 0.01)
        print(f"  best: {best}  ({gain:+.3f} vs isotonic)"
              f"{'   -> SWITCH' if entry['switch'] else ''}")
        results[name] = entry

    print("\n" + "=" * 78)
    switches = [k for k, v in results.items() if v["switch"]]
    print("panels that should change calibration:", switches or "none")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
