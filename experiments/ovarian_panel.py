"""
Is the ovarian cohort a real panel, or is it just measuring age?

Median age is 53 in the cancer group and 36 in the benign group. That is a
seventeen year gap, and it is exactly the trap the general panel fell into:
a model can look excellent while doing nothing except noticing that cancer
patients are older. So age gets its own arm here and has to be beaten.

Five feature sets, one target, one held-out split, no reshuffling:

    A  age and menopausal status        the baseline to beat
    B  routine bloodwork only           no age, no tumour markers
    C  routine bloodwork plus age
    D  tumour markers only              CA125, HE4, CEA, AFP, CA19-9
    E  everything

The target is histology from a resected specimen: ovarian cancer against
benign ovarian tumour, in women who all presented with an ovarian mass.

A note on the precision projection. This is a triage panel, not a population
screen, so projecting onto SEER ovarian incidence would be meaningless. The
relevant prior is the malignancy rate among women referred for surgery on an
adnexal mass, which runs about 20 percent. The cohort itself is 49 percent
malignant because it is a surgical series, so the projection corrects for that
enrichment rather than reporting the cohort rate as if it were the world.

Run:  python experiments/ovarian_panel.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

AGE = ["age", "menopause"]
ROUTINE = [
    "albumin", "protein_total", "glucose", "calcium", "creatinine", "bun",
    "bilirubin", "alt", "ast", "alkaline_phosphatase", "ggt",
    "hemoglobin", "rbc", "platelets", "hematocrit", "mcv", "mch", "rdw",
    "mpv", "neutrophil_pct",
]
MARKERS = ["ca125", "he4", "cea", "alpha_fetoprotein_level", "plasma_ca19_9"]

ARMS = {
    "A age and menopause only": AGE,
    "B routine bloodwork only": ROUTINE,
    "C routine bloodwork + age": AGE + ROUTINE,
    "D tumour markers only": MARKERS,
    "E everything": AGE + ROUTINE + MARKERS,
}

# Malignancy rate among women taken to surgery for an adnexal mass.
REFERRAL_MALIGNANCY = 0.20


def fit(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    ).fit(X, y)


def run(name, feats, tr, te):
    X_tr = tr[feats].apply(pd.to_numeric, errors="coerce")
    X_te = te[feats].apply(pd.to_numeric, errors="coerce")
    med = X_tr.median()
    X_tr, X_te = X_tr.fillna(med), X_te.fillna(med)
    y_tr = tr["ovarian_cancer"].astype(int)
    y_te = te["ovarian_cancer"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y_tr), float(y_tr.mean())),
                               method="isotonic", cv=cv),
        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, cuts = roc_curve(y_tr, oof)
    thr = cuts[int(np.argmax(tpr - fpr))]
    thr = float(np.clip(thr, 0.01, 0.99)) if np.isfinite(thr) else 0.5

    model = fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    ci = bootstrap_ci(np.asarray(y_te), p, roc_auc_score)

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, class_weight="balanced")).fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    sens = sens_at(thr)(np.asarray(y_te), p)
    spec = spec_at(thr)(np.asarray(y_te), p)
    ppv, npv, nnt = projected_ppv_npv(sens, spec, REFERRAL_MALIGNANCY)

    print(f"  {name:<28} AUC {auc:.3f} (95% CI {ci[0]} to {ci[1]})  "
          f"logistic {lr_auc:.3f}  sens {sens:.2f} spec {spec:.2f}  "
          f"PPV {ppv:.2f}  referred/case {nnt:.1f}")
    return {
        "arm": name, "features": feats, "n_features": len(feats),
        "auc": round(float(auc), 3), "auc_ci": ci,
        "logistic_auc": round(float(lr_auc), 3),
        "threshold": round(thr, 4),
        "sensitivity": round(float(sens), 3), "specificity": round(float(spec), 3),
        "ppv_at_referral_prevalence": round(float(ppv), 4),
        "npv_at_referral_prevalence": round(float(npv), 4),
        "referred_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
    }


def main():
    df = pd.read_csv("data/ovarian_soochow.csv")
    y = df["ovarian_cancer"].astype(int)
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"cohort {len(df)} women, {int(y.sum())} malignant ({y.mean():.1%})")
    print(f"train {len(tr)}   test {len(te)} ({int(te.ovarian_cancer.sum())} malignant)\n")

    results = [run(name, feats, tr, te) for name, feats in ARMS.items()]

    base = results[0]["auc"]
    blood_only = results[1]["auc"]
    best = max(results, key=lambda r: r["auc"])

    print("\n" + "=" * 92)
    print(f"age and menopause alone   {base}")
    print(f"routine bloodwork alone   {blood_only}   (no age, no tumour markers)")
    print(f"best feature set          {best['auc']}  ({best['arm']})")
    print(f"gain over age alone       {best['auc'] - base:+.3f}")
    print("=" * 92)

    verdict = (
        "SHIP" if (best["auc_ci"][0] > 0.5 and best["auc"] > base + 0.02)
        else "DO NOT SHIP"
    )
    print(f"\nVERDICT: {verdict}")
    if verdict == "SHIP":
        print("The interval excludes chance and it beats age alone by a real margin.")
    else:
        print("It is measuring age. Same failure as the general panel. Not built.")

    out = {
        "question": "Does the ovarian panel beat age alone?",
        "n": int(len(df)), "cases": int(y.sum()),
        "cohort_prevalence": round(float(y.mean()), 4),
        "referral_prevalence_used": REFERRAL_MALIGNANCY,
        "arms": results,
        "age_only_auc": base,
        "bloodwork_only_auc": blood_only,
        "gain_over_age": round(best["auc"] - base, 3),
        "verdict": verdict,
    }
    with open("experiments/ovarian_panel_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/ovarian_panel_result.json")


if __name__ == "__main__":
    main()
