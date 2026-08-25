"""
Is there a legitimate fifth cancer panel in the cervical risk-factor cohort?

Prostate was withdrawn and cannot be rescued: no public tabular cohort pairs PSA
with a biopsy outcome at usable scale, and NHANES excludes already-diagnosed men
from PSA testing by design. So if a fifth panel is going to exist it has to be a
different disease.

Cervical is the strongest candidate on paper. It is one of only four cancers with
a USPSTF-recommended screening test, and UCI 383 carries 858 women from Caracas
with a **biopsy-confirmed** outcome, which is a harder endpoint than most public
cohorts offer.

Two things have to be checked before building anything.

LEAKAGE. The cohort includes Dx:Cancer, Dx:CIN and Dx:HPV, which record a prior
diagnosis. Training on those would be predicting a biopsy result from the fact
that someone already has the disease. They are dropped.

POWER. 55 positives is below the roughly 96 events needed to estimate a
proportion tightly, so the interval will be wide, and a wide interval that
contains chance is a reason not to ship.

Run:  python experiments/cervical_panel.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Risk factors a patient can answer. Everything beginning Dx: is a prior
# diagnosis and is excluded as leakage.
FEATURES = [
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)",
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
    "IUD", "IUD (years)", "STDs", "STDs (number)", "STDs:HPV",
    "STDs: Number of diagnosis",
]

# Which prior to project precision onto.
#
# The first version of this script used SEER population incidence, 7.8 per
# 100,000 women per year, and reported 7,383 women flagged per true case. That
# number was arithmetically right and conceptually wrong.
#
# These 858 women were not a general population. They attended a clinic in
# Caracas and were assessed for colposcopy, and 6.4 percent had a positive
# biopsy. A panel that runs after referral has to be judged against the
# prevalence in the referred group, not against the prevalence in the street,
# exactly as the ovarian panel is judged against the malignancy rate among
# women taken to surgery rather than against SEER.
#
# Projecting onto population incidence would only be correct if this were
# offered as a population screen, which it is not.
REFERRAL_PREVALENCE = 0.064

# Kept for the record: what the population-incidence projection says.
CERVICAL_INCIDENCE = 7.8 / 100_000


def fit(X, y):
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    ).fit(X, y)


def main():
    raw = fetch_ucirepo(id=383).data.original
    df = raw.copy()

    leaked = [c for c in df.columns if c.startswith("Dx")]
    print(f"Dropped as leakage (prior diagnosis): {leaked}\n")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Biopsy"], errors="coerce")
    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)
    X = X.fillna(X.median())

    print(f"Cohort: {len(y)} women, {int(y.sum())} biopsy-positive ({y.mean():.1%})")
    print(f"Features: {len(FEATURES)}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y_tr), float(y_tr.mean())),
                               method="isotonic", cv=cv),
        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, cuts = roc_curve(y_tr, oof)
    thr = cuts[int(np.argmax(tpr - fpr))]
    thr = float(np.clip(thr, 0.005, 0.99)) if np.isfinite(thr) else 0.5

    model = fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    ci = bootstrap_ci(np.asarray(y_te), p, roc_auc_score)

    # Baselines on the same split.
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, class_weight="balanced")).fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    age_only = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, class_weight="balanced"))
    age_only.fit(X_tr[["Age"]], y_tr)
    age_auc = roc_auc_score(y_te, age_only.predict_proba(X_te[["Age"]])[:, 1])

    sens = sens_at(thr)(np.asarray(y_te), p)
    spec = spec_at(thr)(np.asarray(y_te), p)
    ppv, npv, nnt = projected_ppv_npv(sens, spec, REFERRAL_PREVALENCE)
    pop_ppv, _, pop_nnt = projected_ppv_npv(sens, spec, CERVICAL_INCIDENCE)

    print(f"  ensemble          AUC {auc:.3f}  (95% CI {ci[0]} to {ci[1]})")
    print(f"  logistic          AUC {lr_auc:.3f}")
    print(f"  age alone         AUC {age_auc:.3f}")
    print(f"  threshold {thr:.4f}  sens {sens:.2f}  spec {spec:.2f}")
    print(f"  at referral prevalence {REFERRAL_PREVALENCE*100:.1f}%: PPV {ppv*100:.1f}%, "
          f"NPV {npv*100:.1f}%, {nnt:.1f} referred per true case")
    print(f"  for contrast, at SEER population incidence {CERVICAL_INCIDENCE*100:.4f}%: "
          f"PPV {pop_ppv*100:.2f}%, {pop_nnt:.0f} flagged per true case, which is why "
          f"this is not offered as a population screen")

    beats_chance = ci[0] > 0.5
    beats_age = auc > age_auc + 0.02

    print()
    if beats_chance and beats_age:
        print("VERDICT: the interval excludes chance and it beats age alone. "
              "Worth building as a fifth panel, with the small case count stated.")
    elif beats_chance:
        print("VERDICT: it beats chance but not age alone by a useful margin. "
              "Same problem the general panel has. Not worth a new panel.")
    else:
        print("VERDICT: the interval includes chance. Not built.")

    out = {
        "question": "Can a cervical panel replace the withdrawn prostate one?",
        "n": int(len(y)), "cases": int(y.sum()),
        "features": FEATURES, "dropped_as_leakage": leaked,
        "auc": round(float(auc), 3), "auc_ci": ci,
        "logistic_auc": round(float(lr_auc), 3),
        "age_only_auc": round(float(age_auc), 3),
        "threshold": round(thr, 4),
        "sensitivity": round(float(sens), 3), "specificity": round(float(spec), 3),
        "referral_prevalence": REFERRAL_PREVALENCE,
        "ppv_at_referral_prevalence": round(float(ppv), 4),
        "npv_at_referral_prevalence": round(float(npv), 4),
        "referred_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
        "ppv_at_population_incidence": round(float(pop_ppv), 6),
        "flagged_per_true_case_population": round(float(pop_nnt), 1) if np.isfinite(pop_nnt) else None,
        "beats_chance": bool(beats_chance), "beats_age_alone": bool(beats_age),
    }
    with open("experiments/cervical_panel_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/cervical_panel_result.json")


if __name__ == "__main__":
    main()
