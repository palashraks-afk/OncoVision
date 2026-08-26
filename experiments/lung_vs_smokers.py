"""
Lung cancer against smokers, not against the general population.

The confound
------------
Every lung result so far has been flattered by an easy comparison. Almost all
lung cancer cases smoked, and most of the 45,000 controls did not, so "does this
person smoke" separates the groups nearly on its own. That is why the
questionnaire baseline sits at 0.836 and why the entire lab report only adds
0.022 on top of it. The model is not being asked a hard question.

The clinically real question is different. Nobody needs a tool to tell a
lifelong non-smoker apart from a smoker with lung cancer. The decision that
matters is which of the smokers has the tumour, because that is the population
that actually gets screened.

The Shanghai cohort this project tried to obtain does exactly that by using COPD
patients as controls. That file is behind a proof-of-work bot challenge, which
is not something to work around. But the same comparison can be built from data
already in hand, by simply throwing away the easy controls.

    controls    ever-smokers with no cancer, or anyone with a raised cotinine
    cases       ever-smokers with lung cancer

Now age, sex and smoking status can no longer do the work, and any separation
has to come from the blood. If the lab values still discriminate here, the panel
is genuinely reading the lab report. If they do not, the earlier +0.022 was the
smoking question wearing a lab coat, and lung does not ship.

Arms, on the smoker-restricted cohort:

    A  age, sex, smoking status        the questionnaire, now nearly useless
    B  A plus pack-years               dose, which does carry real information
    C  B plus cotinine                 measured exposure
    D  C plus CRP, blood count, chem   the whole lab report

Run:  python experiments/lung_vs_smokers.py
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
OUT = "experiments/lung_vs_smokers_result.json"

# Serum cotinine above 3 ng/mL is the standard cut for active tobacco exposure.
COTININE_ACTIVE = 3.0

QUEST = ["age", "gender", "smoking"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "rdw", "mpv",
         "glucose", "calcium", "bun", "creatinine", "protein_total", "albumin",
         "ast", "alt", "bilirubin", "alkaline_phosphatase"]

ARMS = {
    "A questionnaire only":      QUEST,
    "B plus pack-years":         QUEST + ["smoking_packyears"],
    "C plus cotinine":           QUEST + ["smoking_packyears", "cotinine"],
    "D plus whole lab report":   QUEST + ["smoking_packyears", "cotinine", "crp"] + BLOOD,
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    df = pd.read_csv("data/nhanes_lung.csv")

    smoker = (df["smoking"] > 0) | (df["cotinine"] >= COTININE_ACTIVE)
    sub = df[smoker].copy().reset_index(drop=True)
    y = sub["lung_cancer"].astype(int)

    print(f"full cohort      {len(df)} adults, {int(df.lung_cancer.sum())} lung cancer")
    print(f"smokers only     {len(sub)} adults, {int(y.sum())} lung cancer ({y.mean():.3%})")
    print(f"                 controls are ever-smokers or cotinine >= {COTININE_ACTIVE} ng/mL\n")

    scores = {}
    for name, feats in ARMS.items():
        X = sub[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        arr = np.array([oof(X, y, s) for s in range(REPEATS)])
        scores[name] = arr
        print(f"  {name:<28} AUC {arr.mean():.3f}  "
              f"(range {arr.min():.3f} to {arr.max():.3f})", flush=True)

    base = scores["A questionnaire only"]
    results = {}
    print()
    for name, arr in scores.items():
        if name == "A questionnaire only":
            continue
        d = arr - base
        wins = int((d > 0).sum())
        good = d.mean() > 0.01 and wins >= REPEATS * 0.75 and d.min() > 0
        print(f"  {name} vs questionnaire: {d.mean():+.3f}  "
              f"(range {d.min():+.3f} to {d.max():+.3f})  wins {wins}/{REPEATS}  "
              f"-> {'HELPS' if good else 'not enough'}")
        results[name] = {"mean_auc": round(float(arr.mean()), 3),
                         "gain": round(float(d.mean()), 3),
                         "range": [round(float(d.min()), 3), round(float(d.max()), 3)],
                         "wins": wins, "repeats": REPEATS, "helps": bool(good)}

    lab = scores["D plus whole lab report"]
    dose = scores["B plus pack-years"]
    d2 = lab - dose
    print(f"\n  lab report over pack-years alone: {d2.mean():+.3f}  "
          f"wins {int((d2>0).sum())}/{REPEATS}")

    with open(OUT, "w") as f:
        json.dump({"n": int(len(sub)), "cases": int(y.sum()),
                   "cotinine_cut": COTININE_ACTIVE,
                   "baseline_auc": round(float(base.mean()), 3),
                   "arms": results,
                   "lab_over_packyears": round(float(d2.mean()), 3)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
