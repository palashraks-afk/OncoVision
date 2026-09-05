"""
Does the prospective panel transfer to a different decade?

The question
------------
Repeated cross-validation tells you whether a result is stable inside one
dataset. It cannot tell you whether the model learned physiology or learned the
survey. For most panels in this project that distinction has never been tested,
and it is the one that decides whether a model is a finding.

NHANES III ran from 1988 to 1994 and is linked to the National Death Index by
the same agency using the same method. So:

    train    NHANES 1999-2014   33,834 adults, 339 cancer deaths in 60 months
    test     NHANES III         14,630 adults, 254 cancer deaths in 60 months

Five to twenty years earlier. Different analysers, different field staff, a
population with substantially higher smoking prevalence, and a cancer mortality
rate of 1.74% against 1.00%. Nothing from the test cohort touches fitting,
feature selection, calibration or thresholding.

Feature set
-----------
Age, sex, the complete blood count and the chemistry panel, on both sides. BMI,
smoking and alcohol are excluded from BOTH cohorts rather than imputed into one,
because an external test with different features on each side is not an external
test. This also isolates the actual claim, which is about blood work rather than
about questionnaires.

Arms
----
    age and sex only     the baseline, transferred
    full blood work      the panel, transferred

Reporting the baseline's transfer too matters. If both transfer equally well,
the panel has carried nothing across; what is being asked is whether the GAIN
survives, not whether the AUC does. Age predicts cancer death in any decade.

Run:  python experiments/prospective_external.py
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
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/prospective_external_result.json"
TRAIN_CSV = "data/nhanes_cancer_mortality.csv"
TEST_CSV = "data/nhanes3_cancer_mortality.csv"

DEMO = ["age", "gender"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "mch",
         "rdw", "mpv", "glucose", "calcium", "bun", "creatinine",
         "protein_total", "albumin", "ast", "alt", "bilirubin",
         "alkaline_phosphatase", "ggt"]

ARMS = {
    "age and sex only": DEMO,
    "full blood work": DEMO + BLOOD,
}


def fit_and_transfer(tr, te, feats, seed=0):
    """Fit on the training cohort, score the test cohort. Medians from train."""
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    ytr = tr["cancer_death"].astype(int)

    # The test cohort is filled with the TRAINING medians, never its own. Using
    # the test cohort's medians would leak its distribution into the imputation
    # and quietly flatter the transfer.
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
    yte = te["cancer_death"].astype(int)

    model = CalibratedClassifierCV(
        tm.build_ensemble(len(ytr), float(ytr.mean())),
        method="isotonic",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), np.asarray(yte), p


def main():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    missing = [c for c in DEMO + BLOOD if c not in te.columns]
    if missing:
        print(f"test cohort lacks {missing} — rerun fetch_nhanes3_external.py")
        return

    print(f"train  NHANES 1999-2014   n={len(tr):,}  deaths="
          f"{int(tr.cancer_death.sum())}  ({tr.cancer_death.mean():.2%})")
    print(f"test   NHANES III         n={len(te):,}  deaths="
          f"{int(te.cancer_death.sum())}  ({te.cancer_death.mean():.2%})\n")

    results = {}
    for name, feats in ARMS.items():
        auc, y, p = fit_and_transfer(tr, te, feats)
        ci = bootstrap_ci(y, p, roc_auc_score)
        results[name] = {"n_features": len(feats), "external_auc": round(auc, 3),
                         "external_auc_ci": ci}
        print(f"  {name:<20} {len(feats):>2} features   external AUC {auc:.3f}  "
              f"(95% CI {ci[0]} to {ci[1]})", flush=True)

    gain = (results["full blood work"]["external_auc"]
            - results["age and sex only"]["external_auc"])

    # How the same comparison looked inside the training survey, for reference.
    internal = None
    p_int = "experiments/prospective_mortality_result.json"
    if os.path.exists(p_int):
        d = json.load(open(p_int))
        arms = d.get("arms", {})
        if "E everything" in arms and "A age and sex" in arms:
            internal = round(arms["E everything"]["gain_over_age_sex"], 3)

    print("\n" + "=" * 78)
    print(f"  gain over age and sex, transferred to a different decade: {gain:+.3f}")
    if internal is not None:
        print(f"  the same gain measured inside the training survey:       {internal:+.3f}")
    survives = gain >= 0.01
    print(f"  -> {'the gain survives the transfer' if survives else 'the gain does NOT survive the transfer'}")

    with open(OUT, "w") as f:
        json.dump({"train_n": int(len(tr)), "train_events": int(tr.cancer_death.sum()),
                   "test_n": int(len(te)), "test_events": int(te.cancer_death.sum()),
                   "arms": results,
                   "external_gain_over_age_sex": round(float(gain), 3),
                   "internal_gain_for_reference": internal,
                   "gain_survives_transfer": bool(survives)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
