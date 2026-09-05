"""
Does the bowel panel work outside the survey it was fitted on?

Why this is the panel to ask about
----------------------------------
Bowel is one of only two panels here that screen for a named cancer from a
routine lab report alone, so it carries more of the application's actual claim
than the high-AUC case-control panels do. It has never been tested outside
NHANES 2005-2014.

The prospective mortality analysis has just demonstrated what that can hide. A
panel that gained +0.013 over age and sex inside its own survey, winning every
paired repeat and scoring 0.837 on leave-one-cycle-out, lost 0.013 on a cohort
from a different decade. Internal validation could not see it. Neither could
leave-one-cycle-out, because cycles of one survey share a protocol, a laboratory
contract and an instrument calibration procedure.

    train    NHANES 2005-2014   23,794 adults, 96 colorectal cancers (0.40%)
    test     NHANES III         14,499 adults, 56 colorectal cancers (0.39%)

The prevalences match to within a hundredth of a percent, which is a check that
the eight-year window was reconstructed the same way on both sides rather than a
coincidence worth celebrating.

Arms
----
    age and sex only     the baseline, transferred
    full panel           the sixteen features that ship

Reporting the baseline's transfer matters as much as the panel's. Age predicts
colorectal cancer in any decade. What is being asked is whether the GAIN over
age survives, not whether the AUC does.

Run:  python experiments/colorectal_external.py
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

OUT = "experiments/colorectal_external_result.json"
TRAIN_CSV = "data/nhanes_colorectal.csv"
TEST_CSV = "data/nhanes3_colorectal.csv"
TARGET = "colorectal_cancer"

DEMO = ["age", "gender"]
PANEL = ["wbc", "rbc", "hemoglobin", "platelets", "glucose", "calcium", "bun",
         "creatinine", "protein_total", "albumin", "ast", "alt", "bilirubin",
         "alkaline_phosphatase"]

ARMS = {
    "age and sex only": DEMO,
    "full panel": DEMO + PANEL,
}


def fit_and_transfer(tr, te, feats, seed=0):
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    ytr = tr[TARGET].astype(int)

    # Training medians, never the test cohort's own. Using the test cohort's
    # medians would leak its distribution into the imputation and flatter the
    # transfer.
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(med)
    yte = te[TARGET].astype(int)

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
    missing = [c for c in DEMO + PANEL if c not in te.columns]
    if missing:
        print(f"test cohort lacks {missing} — rerun fetch_nhanes3_colorectal.py")
        return

    print(f"train  NHANES 2005-2014   n={len(tr):,}  cases={int(tr[TARGET].sum())}"
          f"  ({tr[TARGET].mean():.2%})")
    print(f"test   NHANES III         n={len(te):,}  cases={int(te[TARGET].sum())}"
          f"  ({te[TARGET].mean():.2%})\n")

    results = {}
    for name, feats in ARMS.items():
        auc, y, p = fit_and_transfer(tr, te, feats)
        ci = bootstrap_ci(y, p, roc_auc_score)
        results[name] = {"n_features": len(feats), "external_auc": round(auc, 3),
                         "external_auc_ci": ci}
        print(f"  {name:<20} {len(feats):>2} features   external AUC {auc:.3f}  "
              f"(95% CI {ci[0]} to {ci[1]})", flush=True)

    gain = results["full panel"]["external_auc"] - results["age and sex only"]["external_auc"]

    internal = None
    p_int = "experiments/demographic_gain_result.json"
    if os.path.exists(p_int):
        internal = (json.load(open(p_int)).get("colorectal") or {}).get("gain")

    print("\n" + "=" * 78)
    print(f"  gain over age and sex, transferred to a different decade: {gain:+.3f}")
    if internal is not None:
        print(f"  the same gain measured inside the training survey:       {internal:+.3f}")
    survives = gain >= 0.01
    print(f"  -> {'the gain survives the transfer' if survives else 'the gain does NOT survive the transfer'}")

    with open(OUT, "w") as f:
        json.dump({"train_n": int(len(tr)), "train_events": int(tr[TARGET].sum()),
                   "test_n": int(len(te)), "test_events": int(te[TARGET].sum()),
                   "arms": results,
                   "external_gain_over_age_sex": round(float(gain), 3),
                   "internal_gain_for_reference": internal,
                   "gain_survives_transfer": bool(survives)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
