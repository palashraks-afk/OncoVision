"""
Should the general and liver panels be retrained on NHANES instead of their
original cohorts?

The general panel is the clearest failure in this project. It scores 0.966 on a
held-out slice of the 1,500-record risk-factor cohort it trains on, and 0.596 on
a representative sample of US adults. A model that good on its own data and that
poor on the public is not measuring cancer risk, it is measuring its cohort.

NHANES now supplies 37,564 adults with 3,536 cancer diagnoses across seven
cycles, and 35,511 adults with 1,436 liver conditions. So the question is direct:
train on the real population instead, and test on cycles the model never saw.

Three arms per panel, all tested on the same held-out late cycles (2015 to 2018):

    A  original cohort          the status quo
    B  NHANES early cycles      2005 to 2014
    C  NHANES early + original  pooled

Temporal validation is the point. A later cycle is a different sample, a
different decade and different assay lots, so it is a genuine external test
rather than a reshuffle.

Run:  python experiments/retrain_on_nhanes.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LATE = ("2015-2016", "2017-2018")

GENERAL_FEATURES = ["age", "gender", "bmi", "smoking", "alcohol_intake", "physical_activity"]
LIVER_FEATURES = ["age", "gender", "bilirubin", "alkaline_phosphatase",
                  "alt", "ast", "protein_total", "albumin"]

GENERAL_PREV = 450.7 / 100_000
LIVER_PREV = 3100.0 / 100_000


def fit(X, y):
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    ).fit(X, y)


def logistic():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def measure(y, p, prevalence):
    sens = sens_at(0.5)(np.asarray(y), np.asarray(p))
    spec = spec_at(0.5)(np.asarray(y), np.asarray(p))
    ppv, _npv, nnt = projected_ppv_npv(sens, spec, prevalence)
    return {
        "auc": round(float(roc_auc_score(y, p)), 3),
        "auc_ci": bootstrap_ci(np.asarray(y), np.asarray(p), roc_auc_score),
        "sensitivity": round(float(sens), 3),
        "specificity": round(float(spec), 3),
        "ppv_at_population_prevalence": round(float(ppv), 5),
        "people_flagged_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
    }


def arm(name, X_tr, y_tr, X_te, y_te, prevalence):
    ens = fit(X_tr, y_tr)
    res = measure(y_te, ens.predict_proba(X_te)[:, 1], prevalence)
    lr = logistic().fit(X_tr, y_tr)
    res["logistic_auc"] = round(float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])), 3)
    res["arm"] = name
    res["n_train"] = int(len(y_tr))
    res["cases_train"] = int(np.sum(y_tr))
    print(f"  {name:<34} n={len(y_tr):<7} AUC {res['auc']:.3f} "
          f"(95% CI {res['auc_ci'][0]} to {res['auc_ci'][1]})  "
          f"logistic {res['logistic_auc']:.3f}  "
          f"flagged/case {res['people_flagged_per_true_case']}")
    return res


def run_general():
    print("\nGENERAL PANEL, tested on NHANES 2015-2018")
    nh = pd.read_csv("data/nhanes_general_multicycle.csv")
    nh = nh.dropna(subset=GENERAL_FEATURES + ["any_cancer"])
    late = nh["cycle"].isin(LATE)
    te_X, te_y = nh.loc[late, GENERAL_FEATURES], nh.loc[late, "any_cancer"].astype(int)
    early_X, early_y = nh.loc[~late, GENERAL_FEATURES], nh.loc[~late, "any_cancer"].astype(int)
    print(f"  test set: {len(te_y)} adults, {int(te_y.sum())} cancers ({te_y.mean():.1%})")

    cfg = next(c for c in tm.DATASETS if c["name"] == "general")
    Xo, yo, med = tm.prepare(cfg)
    Xo_aligned = pd.DataFrame({f: Xo[f] if f in Xo.columns else med.get(f, 0.0)
                               for f in GENERAL_FEATURES})[GENERAL_FEATURES]

    out = [
        arm("A original risk-factor cohort", Xo_aligned, yo, te_X, te_y, GENERAL_PREV),
        arm("B NHANES 2005-2014", early_X, early_y, te_X, te_y, GENERAL_PREV),
        arm("C NHANES early + original",
            pd.concat([early_X, Xo_aligned], ignore_index=True),
            pd.concat([early_y, yo], ignore_index=True), te_X, te_y, GENERAL_PREV),
    ]
    return out


def run_liver():
    print("\nLIVER PANEL, tested on NHANES 2015-2018")
    nh = pd.read_csv("data/nhanes_liver_multicycle.csv").dropna(subset=LIVER_FEATURES + ["liver_disease"])
    late = nh["cycle"].isin(LATE)
    te_X, te_y = nh.loc[late, LIVER_FEATURES], nh.loc[late, "liver_disease"].astype(int)
    early_X, early_y = nh.loc[~late, LIVER_FEATURES], nh.loc[~late, "liver_disease"].astype(int)
    print(f"  test set: {len(te_y)} adults, {int(te_y.sum())} liver conditions ({te_y.mean():.1%})")

    clin = pd.concat([
        pd.read_csv("data/ilpd_liver_india.csv")[LIVER_FEATURES + ["liver_disease"]],
        pd.read_csv("data/hcv_liver_germany.csv")[LIVER_FEATURES + ["liver_disease"]],
    ], ignore_index=True)

    out = [
        arm("A India + Germany only", clin[LIVER_FEATURES], clin["liver_disease"].astype(int),
            te_X, te_y, LIVER_PREV),
        arm("B NHANES 2005-2014", early_X, early_y, te_X, te_y, LIVER_PREV),
        arm("C NHANES early + India + Germany",
            pd.concat([early_X, clin[LIVER_FEATURES]], ignore_index=True),
            pd.concat([early_y, clin["liver_disease"].astype(int)], ignore_index=True),
            te_X, te_y, LIVER_PREV),
    ]
    return out


def main():
    results = {"general": run_general(), "liver": run_liver()}

    print("\n" + "=" * 78)
    for panel, arms in results.items():
        best = max(arms, key=lambda a: a["auc"])
        baseline = arms[0]
        print(f"{panel}: best is '{best['arm']}' at {best['auc']}, "
              f"against the status quo at {baseline['auc']} "
              f"({best['auc'] - baseline['auc']:+.3f})")
    print("=" * 78)

    with open("experiments/retrain_on_nhanes_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/retrain_on_nhanes_result.json")


if __name__ == "__main__":
    main()
