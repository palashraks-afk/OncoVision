"""
How much does each panel add over knowing age and sex, measured stably?

Why not just use the held-out number
------------------------------------
The held-out split says the bowel panel scores 0.793 against an 0.817 age-and-sex
baseline, so it LOSES to demographics by 0.024. Twenty paired repeats on
identical folds say it WINS by 0.038. Both cannot inform a warning shown to a
user, and the single split is the one already shown to be unrepresentative: it
sits at the 40th percentile on the panel and its baseline happened to draw well.

Labelling bowel "barely beats age and sex" off that split would repeat exactly
the error that got the cervical panel withdrawn, which is treating one draw as
an estimate.

So the gain that drives the warning is measured the same way everything else in
this project now is: repeated, paired, on identical folds.

    for each panel, for each repeat
        fit the full feature set          -> AUC
        fit age and sex only              -> AUC
        difference is that repeat's gain

Panels whose cohorts carry no usable age-and-sex baseline are reported as such
rather than given a number. Breast is the case: the Wisconsin cohort has neither
age nor sex, so "how much do you add over demographics" has no answer there and
saying so is better than inventing one.

Run:  python experiments/demographic_gain.py
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

OUT = "experiments/demographic_gain_result.json"


def repeats_for(n_rows: int) -> int:
    if n_rows > 20_000:
        return 5
    return 20


def oof_auc(X, y, seed):
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
        if name in tm.WITHDRAWN:
            continue
        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        base_feats = [f for f in ("age", "gender") if f in X.columns]
        if not base_feats:
            print(f"=== {name} ===  no age or sex in this cohort, gain is undefined")
            results[name] = {"gain": None, "reason": "cohort records neither age nor sex"}
            continue

        n = repeats_for(len(y))
        full, base = [], []
        for s in range(n):
            full.append(oof_auc(X, y, s))
            base.append(oof_auc(X[base_feats], y, s))
        d = np.array(full) - np.array(base)
        wins = int((d > 0).sum())
        entry = {
            "n_repeats": n,
            "panel_auc": round(float(np.mean(full)), 3),
            "age_sex_auc": round(float(np.mean(base)), 3),
            "gain": round(float(d.mean()), 3),
            "gain_range": [round(float(d.min()), 3), round(float(d.max()), 3)],
            "wins": wins,
            "baseline_features": base_feats,
        }
        entry["barely_beats_demographics"] = bool(
            entry["gain"] < tm.BARELY_BEATS_DEMOGRAPHICS)
        results[name] = entry
        print(f"=== {name} ===  panel {entry['panel_auc']}  age/sex {entry['age_sex_auc']}  "
              f"gain {entry['gain']:+.3f}  wins {wins}/{n}"
              f"{'   <-- barely beats demographics' if entry['barely_beats_demographics'] else ''}",
              flush=True)

    print("\n" + "=" * 80)
    weak = [k for k, v in results.items() if v.get("barely_beats_demographics")]
    print("panels that barely beat age and sex:", weak or "none")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
