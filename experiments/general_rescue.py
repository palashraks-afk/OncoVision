"""
Can the general panel be made worth shipping?

The problem
-----------
The general panel is the weakest thing in this project. It reaches 0.757 where
age and sex alone reach 0.727, a gain of 0.005 on the held-out split. It is
close to a demographic lookup wearing a lab coat.

An earlier experiment concluded that routine bloodwork actively hurts it: 14
blood values scored 0.663 against 0.729 for age and sex. That result stands, but
it was narrower than the conclusion drawn from it. What was tested was a
complete blood count and a metabolic panel. Two lab values that plausibly track
occult malignancy were never offered to it at all:

    SERUM COTININE   objective tobacco exposure, and tobacco causes a large
                     share of all cancers. The panel currently gets a
                     three-level self-reported smoking answer instead.
    CRP              chronic inflammation, which accompanies tumour biology and
                     is one of the few cheap systemic markers of it.

So the honest version of the earlier claim is "a blood count and a metabolic
panel do not help". Whether these two help is a separate question and is
answered here.

Arms
----
    A  age and sex                              the bar
    B  what ships today: + BMI, smoking, alcohol
    C  B + serum cotinine
    D  B + cotinine + CRP
    E  D + the full blood count and chemistry   the combination previously
                                                found harmful, retested with
                                                the two new values present

CRP is missing for 40 percent of the cohort, which matters: a feature present
for only some people can look useful by proxying for which cycle someone was in.
So D and E are also run on the CRP-complete subset, and if the gain only exists
on the full cohort it is an artefact rather than a finding.

Run:  python experiments/general_rescue.py
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

REPEATS = 5
OUT = "experiments/general_rescue_result.json"

SHIPPED = ["age", "gender", "bmi", "smoking", "alcohol_intake"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "glucose", "calcium", "bun",
         "creatinine", "protein_total", "albumin", "ast", "alt", "bilirubin",
         "alkaline_phosphatase"]

ARMS = {
    "A age and sex":                 ["age", "gender"],
    "B shipped today":               SHIPPED,
    "C + serum cotinine":            SHIPPED + ["cotinine"],
    "D + cotinine and CRP":          SHIPPED + ["cotinine", "crp"],
    "E + whole lab report":          SHIPPED + ["cotinine", "crp"] + BLOOD,
}


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), p


def run(df, label):
    y = df["recent_cancer"].astype(int).reset_index(drop=True)
    print(f"\n{label}: {len(y)} adults, {int(y.sum())} recent cancers "
          f"({y.mean():.2%})")
    scores, last = {}, {}
    for name, feats in ARMS.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        aucs = []
        for s in range(REPEATS):
            a, p = oof(X, y, s)
            aucs.append(a)
            if s == 0:
                last[name] = p
        arr = np.array(aucs)
        scores[name] = arr
        ci = bootstrap_ci(np.asarray(y), last[name], roc_auc_score)
        print(f"  {name:<26} AUC {arr.mean():.3f}  (CI {ci[0]} to {ci[1]})", flush=True)
    return scores, y


def compare(scores, base="A age and sex"):
    out, b = {}, scores[base]
    print()
    for name, arr in scores.items():
        if name == base:
            continue
        d = arr - b
        wins = int((d > 0).sum())
        good = d.mean() > 0.01 and wins >= len(d) * 0.75 and d.min() > 0
        print(f"  {name} vs age and sex: {d.mean():+.3f}  wins {wins}/{len(d)}"
              f"  -> {'real' if good else 'not enough'}")
        out[name] = {"auc": round(float(arr.mean()), 3),
                     "gain_over_age_sex": round(float(d.mean()), 3),
                     "wins": wins, "repeats": len(d), "meaningful": bool(good)}
    return out


def main():
    df = pd.read_csv("data/nhanes_screening_general.csv")
    full_scores, _ = run(df, "FULL COHORT")
    full = compare(full_scores)

    sub = df[df["crp"].notna()].copy()
    sub_scores, _ = run(sub, "CRP-COMPLETE SUBSET")
    subset = compare(sub_scores)

    shipped_gain = full["B shipped today"]["gain_over_age_sex"]
    best_name = max(full, key=lambda k: full[k]["gain_over_age_sex"])
    best = full[best_name]
    print("\n" + "=" * 84)
    print(f"shipped today          {full['B shipped today']['auc']}  "
          f"({shipped_gain:+.3f} over age and sex)")
    print(f"best arm               {best['auc']}  ({best['gain_over_age_sex']:+.3f})  {best_name}")
    improve = best["gain_over_age_sex"] - shipped_gain
    print(f"improvement available  {improve:+.3f}")
    if improve >= 0.01 and best["meaningful"]:
        print("-> worth adopting; the general panel can be made to read the lab report")
    else:
        print("-> nothing here rescues it. The general panel stays weak and says so.")

    with open(OUT, "w") as f:
        json.dump({"full_cohort": full, "crp_complete_subset": subset,
                   "shipped_gain": shipped_gain,
                   "best_arm": best_name,
                   "improvement_available": round(improve, 3)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
