"""
Does the external "gain over age and sex" survive a baseline that can use age?

colorectal_external.py and general_external.py both fit a calibrated tree
ensemble on the age-and-sex arm as well as the full panel. Given two features,
one of them binary, a tree ensemble ranks people in coarse steps, so that arm
understates what age and sex alone can do. On the bowel cohort a logistic
age-and-sex model scores 0.822 where the ensemble scored 0.765 -- a larger
difference than the entire gain the panel was credited with.

So the external transfer is repeated here with both model kinds on both arms,
trained on the continuous survey and tested on NHANES III, and the gain is
judged against the STRONGER of the two baselines. A paired bootstrap over test
patients gives the interval, since both scores rank the same people.

Nothing from NHANES III touches fitting, calibration or imputation: training
medians fill its gaps, exactly as before.

Run:  python experiments/external_baseline_strength.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/external_baseline_strength_result.json"
BOOT = 2000

PANELS = {
    "colorectal": {
        "train": "data/nhanes_colorectal.csv",
        "test": "data/nhanes3_colorectal.csv",
        "target": "colorectal_cancer",
        "features": ["age", "gender", "wbc", "rbc", "hemoglobin", "platelets",
                     "glucose", "calcium", "bun", "creatinine", "protein_total",
                     "albumin", "ast", "alt", "bilirubin", "alkaline_phosphatase"],
    },
    "general": {
        "train": "data/nhanes_screening_general.csv",
        "test": "data/nhanes3_general.csv",
        "target": "recent_cancer",
        "features": ["age", "gender", "bmi", "smoking", "alcohol_intake"],
    },
}
BASE = ["age", "gender"]


def fit_predict(tr, te, feats, target, kind):
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    ytr = tr[target].astype(int)
    model = CalibratedClassifierCV(
        tm.model_factory(kind, len(ytr), float(ytr.mean())), method="isotonic",
        cv=StratifiedKFold(5, shuffle=True, random_state=0))
    model.fit(Xtr.fillna(med), ytr)
    Xte = pd.DataFrame({f: (pd.to_numeric(te[f], errors="coerce") if f in te.columns
                            else np.nan) for f in feats}).fillna(med)
    return model.predict_proba(Xte)[:, 1]


def main():
    results = {}
    for name, cfg in PANELS.items():
        tr, te = pd.read_csv(cfg["train"]), pd.read_csv(cfg["test"])
        y = te[cfg["target"]].astype(int).to_numpy()
        p = {}
        for kind in ("logistic", "ensemble"):
            p[f"full_{kind}"] = fit_predict(tr, te, cfg["features"], cfg["target"], kind)
            p[f"base_{kind}"] = fit_predict(tr, te, BASE, cfg["target"], kind)
        auc = {k: float(roc_auc_score(y, v)) for k, v in p.items()}

        best_base = max(("base_logistic", "base_ensemble"), key=auc.get)
        best_full = max(("full_logistic", "full_ensemble"), key=auc.get)

        rng = np.random.default_rng(0)
        gains_old, gains_honest = [], []
        for _ in range(BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0 or y[i].sum() == len(i):
                continue
            a = {k: roc_auc_score(y[i], v[i]) for k, v in p.items()}
            gains_old.append(a["full_ensemble"] - a["base_ensemble"])
            gains_honest.append(max(a["full_logistic"], a["full_ensemble"])
                                - max(a["base_logistic"], a["base_ensemble"]))

        def ci(g):
            return [round(float(np.percentile(g, 2.5)), 3),
                    round(float(np.percentile(g, 97.5)), 3)]

        old = auc["full_ensemble"] - auc["base_ensemble"]
        honest = auc[best_full] - auc[best_base]
        results[name] = {
            "test_n": int(len(y)), "test_events": int(y.sum()),
            "auc": {k: round(v, 3) for k, v in auc.items()},
            "old_gain_ensemble_vs_ensemble": round(old, 3), "old_gain_ci": ci(gains_old),
            "honest_gain_best_vs_best": round(honest, 3), "honest_gain_ci": ci(gains_honest),
            "strongest_baseline": best_base, "best_panel": best_full,
            "gain_survives_honest_baseline": bool(ci(gains_honest)[0] > 0),
        }
        r = results[name]
        print(f"=== {name} ===  NHANES III n={r['test_n']:,} events={r['test_events']}")
        for k, v in r["auc"].items():
            print(f"    {k:<15} {v:.3f}")
        print(f"    old gain  (ensemble vs ensemble)  {old:+.3f}  CI {r['old_gain_ci']}")
        print(f"    honest gain (best vs best)        {honest:+.3f}  CI {r['honest_gain_ci']}"
              f"  -> {'survives' if r['gain_survives_honest_baseline'] else 'DOES NOT SURVIVE'}",
              flush=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
