"""
Does reading the blood beat asking whether they smoke?

The bar
-------
Lung cancer is overwhelmingly caused by tobacco, so the baseline is not age. It
is age, sex and the smoking question itself. A lung panel only justifies its
existence if the LAB VALUES add something on top of what the patient already
told the intake form, because "you are old and you smoke" is not a product.

The previous attempt added a complete blood count and gained 0.017. This adds
the measurements that attempt missed:

    A  age, sex, self-reported smoking          the questionnaire, the baseline
    B  A plus pack-years                        a better questionnaire
    C  A plus serum cotinine                    one lab value: measured tobacco
    D  A plus cotinine, CRP, blood count, chem  the whole lab report
    E  D plus spirometry                        only where FEV1/FVC exists

C is the interesting arm. Cotinine is the nicotine metabolite and the objective
standard for tobacco exposure, and it sits on a lab report. If C beats A by a
wide margin then the panel is genuinely reading the blood rather than restating
the questionnaire, which is the entire claim this application makes.

E is reported separately and on its own subset, because spirometry exists for
only 26 percent of the cohort and restricting to it costs most of the events.

Method: paired repeated cross-validation on identical folds, the arbiter used
throughout this project.

Run:  python experiments/lung_panel.py
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
OUT = "experiments/lung_panel_result.json"

QUESTIONNAIRE = ["age", "gender", "smoking"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "rdw", "mpv",
         "glucose", "calcium", "bun", "creatinine", "protein_total", "albumin",
         "ast", "alt", "bilirubin", "alkaline_phosphatase"]
SPIRO = ["fev1", "fvc", "fev1_fvc_ratio"]

ARMS = {
    "A questionnaire only":        QUESTIONNAIRE,
    "B plus pack-years":           QUESTIONNAIRE + ["smoking_packyears"],
    "C plus cotinine":             QUESTIONNAIRE + ["cotinine"],
    "D plus whole lab report":     QUESTIONNAIRE + ["cotinine", "crp"] + BLOOD,
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def run(df, arms, label, repeats=REPEATS):
    y = df["lung_cancer"].astype(int).reset_index(drop=True)
    print(f"\n{label}: {len(y)} adults, {int(y.sum())} lung cancer ({y.mean():.3%})")
    scores = {}
    for name, feats in arms.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        aucs = [oof(X, y, s) for s in range(repeats)]
        arr = np.array(aucs)
        scores[name] = arr
        print(f"  {name:<28} AUC {arr.mean():.3f}  (range {arr.min():.3f} to {arr.max():.3f})",
              flush=True)
    return scores


def compare(scores, base_name, repeats=REPEATS):
    base = scores[base_name]
    out = {}
    print()
    for name, arr in scores.items():
        if name == base_name:
            continue
        d = arr - base
        wins = int((d > 0).sum())
        lo, hi = float(d.min()), float(d.max())
        good = d.mean() > 0.01 and wins >= repeats * 0.75 and lo > 0
        print(f"  {name} vs baseline: {d.mean():+.3f}  (range {lo:+.3f} to {hi:+.3f})  "
              f"wins {wins}/{repeats}  -> {'HELPS' if good else 'not enough'}")
        out[name] = {"mean_auc": round(float(arr.mean()), 3),
                     "gain": round(float(d.mean()), 3),
                     "range": [round(lo, 3), round(hi, 3)],
                     "wins": wins, "repeats": repeats, "helps": bool(good)}
    return out


def main():
    df = pd.read_csv("data/nhanes_lung.csv")

    scores = run(df, ARMS, "FULL COHORT")
    results = compare(scores, "A questionnaire only")

    # Spirometry subset, reported separately because it costs most of the events.
    sub = df[df["fev1"].notna()].copy()
    spiro_arms = {
        "A questionnaire only": QUESTIONNAIRE,
        "D whole lab report": QUESTIONNAIRE + ["cotinine", "crp"] + BLOOD,
        "E plus spirometry": QUESTIONNAIRE + ["cotinine", "crp"] + BLOOD + SPIRO,
    }
    s2 = run(sub, spiro_arms, "SPIROMETRY SUBSET")
    spiro_results = compare(s2, "A questionnaire only")

    payload = {
        "full_cohort": {"n": int(len(df)), "cases": int(df.lung_cancer.sum()),
                        "baseline_auc": round(float(scores["A questionnaire only"].mean()), 3),
                        "arms": results},
        "spirometry_subset": {"n": int(len(sub)), "cases": int(sub.lung_cancer.sum()),
                              "baseline_auc": round(float(s2["A questionnaire only"].mean()), 3),
                              "arms": spiro_results},
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
