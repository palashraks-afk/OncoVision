"""
Does reading the bloodwork beat knowing someone's age?

The old general panel reached 0.781 while age and sex alone reached 0.777. It
was adding 0.004, because it predicted a lifetime diagnosis from lifestyle and
never looked at a blood value.

This tests the rebuilt version against the only baseline that matters. Four
feature sets, one target, one held-out split, no reshuffling:

    A  age and sex                    the baseline to beat
    B  age, sex and lifestyle         what the old panel used
    C  bloodwork only                 14 values, no demographics at all
    D  bloodwork, demographics and lifestyle

Target is cancer diagnosed within four years of the exam, with long-ago
survivors excluded rather than counted as positive.

Split is temporal: trained on 2005 to 2012, tested on 2013 to 2014. A later
cycle is a different sample and different assay lots, so it is a real test.

Run:  python experiments/screening_vs_age.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LATE = ("2013-2014",)

DEMOG = ["age", "gender"]
LIFESTYLE = ["bmi", "smoking", "alcohol_intake"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "glucose", "calcium", "bun",
         "creatinine", "protein_total", "albumin", "ast", "alt", "bilirubin",
         "alkaline_phosphatase"]

ARMS = {
    "A age and sex only": DEMOG,
    "B age, sex, lifestyle": DEMOG + LIFESTYLE,
    "C bloodwork only": BLOOD,
    "D bloodwork + demographics + lifestyle": DEMOG + LIFESTYLE + BLOOD,
}

# Recent-diagnosis prevalence measured in this very cohort, so the precision
# projection is self consistent rather than borrowed.
PREVALENCE = 0.0314


def fit(X, y):
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    ).fit(X, y)


def run(name, feats, tr, te):
    X_tr = tr[feats].apply(pd.to_numeric, errors="coerce")
    X_te = te[feats].apply(pd.to_numeric, errors="coerce")
    med = X_tr.median()
    X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
    y_tr = tr["recent_cancer"].astype(int)
    y_te = te["recent_cancer"].astype(int)

    model = fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    ci = bootstrap_ci(np.asarray(y_te), p, roc_auc_score)

    # Operating point from the training data only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y_tr), float(y_tr.mean())),
                               method="isotonic", cv=cv),
        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, cuts = roc_curve(y_tr, oof)
    thr = cuts[int(np.argmax(tpr - fpr))]
    thr = float(np.clip(thr, 0.005, 0.99)) if np.isfinite(thr) else 0.5

    sens = sens_at(thr)(np.asarray(y_te), p)
    spec = spec_at(thr)(np.asarray(y_te), p)
    ppv, _npv, nnt = projected_ppv_npv(sens, spec, PREVALENCE)

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, class_weight="balanced")).fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    print(f"  {name:<40} AUC {auc:.3f} (95% CI {ci[0]} to {ci[1]})  "
          f"logistic {lr_auc:.3f}  sens {sens:.2f} spec {spec:.2f}  "
          f"flagged/case {nnt:.1f}")
    return {
        "arm": name, "features": feats, "n_features": len(feats),
        "auc": round(float(auc), 3), "auc_ci": ci,
        "logistic_auc": round(float(lr_auc), 3),
        "threshold": round(thr, 4),
        "sensitivity": round(float(sens), 3), "specificity": round(float(spec), 3),
        "ppv_at_prevalence": round(float(ppv), 4),
        "flagged_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
    }


def main():
    df = pd.read_csv("data/nhanes_screening_general.csv")
    late = df["cycle"].isin(LATE)
    tr, te = df[~late], df[late]
    print(f"train {len(tr)} ({int(tr.recent_cancer.sum())} cases)   "
          f"test {len(te)} ({int(te.recent_cancer.sum())} cases), held-out cycle\n")

    results = [run(name, feats, tr, te) for name, feats in ARMS.items()]

    base = results[0]["auc"]
    best = max(results, key=lambda r: r["auc"])
    print("\n" + "=" * 78)
    print(f"age and sex alone      {base}")
    print(f"best feature set       {best['auc']}  ({best['arm']})")
    print(f"gain from the bloodwork {best['auc'] - base:+.3f}")
    print("=" * 78)

    with open("experiments/screening_vs_age_result.json", "w") as f:
        json.dump({"prevalence": PREVALENCE, "arms": results,
                   "gain_over_age_sex": round(best["auc"] - base, 3)}, f, indent=2)
    print("\nwrote experiments/screening_vs_age_result.json")


if __name__ == "__main__":
    main()
