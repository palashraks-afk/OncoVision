"""
External validation of the liver panel across two independent real cohorts.

The question every held-out split fails to answer is whether a model works on
patients it has never seen from a source it has never seen. A held-out slice of
one dataset shares that dataset's hospital, its assay machines, its referral
patterns and its population. Splitting it does not test generalisation, it
tests memorisation.

Two independent cohorts make the real test possible:

  India    ILPD, 583 patients, Andhra Pradesh. 71% liver disease.
  Germany  HCV data, 589 patients. 9.5% liver disease, the rest blood donors.

Different continent, hospital, protocol, population and disease prevalence, and
eight shared liver chemistry measurements after unit harmonisation.

Trained on one, tested on the other, in both directions. Nothing from the test
cohort touches training, tuning or calibration.

Run:  python external_validation.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv, calibration_slope_intercept

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES = ["age", "gender", "bilirubin", "alkaline_phosphatase",
            "alt", "ast", "protein_total", "albumin"]
TARGET = "liver_disease"

COHORTS = {
    "India": ("data/ilpd_liver_india.csv", "ILPD, Andhra Pradesh, UCI 225"),
    "Germany": ("data/hcv_liver_germany.csv", "HCV data, Germany, UCI 571"),
}

# SEER liver and intrahepatic bile duct incidence, used only to show what the
# precision would look like at a screening prevalence. Chronic liver disease is
# far more common than liver cancer, so this is a floor, not an estimate of
# liver disease prevalence.
LIVER_INCIDENCE = 9.5 / 100_000


def load(name):
    path, _ = COHORTS[name]
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET].astype(int)


def metrics(y, p, label):
    auc = roc_auc_score(y, p)
    sens = sens_at(0.5)(np.asarray(y), np.asarray(p))
    spec = spec_at(0.5)(np.asarray(y), np.asarray(p))
    slope, intercept = calibration_slope_intercept(np.asarray(y), np.asarray(p))
    ppv, npv, nnt = projected_ppv_npv(sens, spec, LIVER_INCIDENCE)
    return {
        "setting": label,
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": round(float(np.mean(y)), 3),
        "auc": round(float(auc), 3),
        "auc_ci": bootstrap_ci(np.asarray(y), np.asarray(p), roc_auc_score),
        "sensitivity": round(float(sens), 3),
        "sensitivity_ci": bootstrap_ci(np.asarray(y), np.asarray(p), sens_at(0.5)),
        "specificity": round(float(spec), 3),
        "specificity_ci": bootstrap_ci(np.asarray(y), np.asarray(p), spec_at(0.5)),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "calibration_slope": slope,
    }


def fit(X, y):
    """Same ensemble the application ships, calibrated the same way."""
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    )
    model.fit(X, y)
    return model


def run_direction(train_name, test_name):
    Xtr, ytr = load(train_name)
    Xte, yte = load(test_name)

    model = fit(Xtr, ytr)

    # Internal reference: held-out split of the training cohort itself.
    Xa, Xb, ya, yb = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=RANDOM_STATE, stratify=ytr
    )
    internal_model = fit(Xa, ya)
    internal = metrics(yb, internal_model.predict_proba(Xb)[:, 1],
                       f"internal held-out split of {train_name}")

    # The real test: a cohort from another country.
    external = metrics(yte, model.predict_proba(Xte)[:, 1],
                       f"external, trained on {train_name}, tested on {test_name}")

    # Logistic regression, same splits, as a floor.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    lr_external = round(float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])), 3)

    return {
        "train": train_name,
        "test": test_name,
        "internal": internal,
        "external": external,
        "external_logistic_auc": lr_external,
        "auc_drop": round(internal["auc"] - external["auc"], 3),
    }


def main():
    print("Loading cohorts")
    for name, (path, desc) in COHORTS.items():
        X, y = load(name)
        print(f"  {name:<9} {len(y):>4} patients, {int(y.sum()):>4} with liver disease "
              f"({y.mean():.1%}) - {desc}")

    results = {}
    for a, b in [("India", "Germany"), ("Germany", "India")]:
        print(f"\nTraining on {a}, testing on {b}")
        r = run_direction(a, b)
        results[f"{a}_to_{b}"] = r
        print(f"  internal held-out AUC  {r['internal']['auc']}  "
              f"(95% CI {r['internal']['auc_ci'][0]} to {r['internal']['auc_ci'][1]})")
        print(f"  EXTERNAL AUC           {r['external']['auc']}  "
              f"(95% CI {r['external']['auc_ci'][0]} to {r['external']['auc_ci'][1]})")
        print(f"  drop                   {r['auc_drop']}")
        print(f"  logistic on external   {r['external_logistic_auc']}")

    with open("external_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 88)
    print(f"{'direction':<22}{'internal':<12}{'external':<12}{'drop':<10}{'external 95% CI'}")
    print("=" * 88)
    for k, r in results.items():
        ci = r["external"]["auc_ci"]
        print(f"{k.replace('_', ' '):<22}{r['internal']['auc']:<12}{r['external']['auc']:<12}"
              f"{r['auc_drop']:<10}{ci[0]} to {ci[1]}")
    print("\nwrote external_validation.json")


if __name__ == "__main__":
    main()
