"""
Does the general panel's rule-out cut keep its promise outside its own survey?

The last untested one
---------------------
Of the panels that ship a rule-out call, the bowel cut has been applied
unchanged to NHANES III and kept its promise inside tolerance. Liver ships no
cut, because the cost model says everyone in that group should have the
confirmatory test anyway. Lung ships no cut, because catching every case there
means excluding almost nobody. That leaves the general panel, and it is the one
that most needs asking.

The general panel tells 22% of everyone who runs it that they are unlikely to
have cancer, and it does that on five questions with no blood test at all. It
is also the panel that barely beats age and sex. A cut that over-promises here
reassures more people, on thinner evidence, than any other cut on the site.

    train    NHANES 2005-2016   28,711 adults, 897 cancers within 4 years
    test     NHANES III         15,896 adults, 252 cancers within 4 years

What this measures
------------------
Discrimination: does the gain over age and sex survive a different decade.

The rule-out cut is the other half of the question and it lives in
experiments/rule_out_external.py, which holds every shipped cut to the same
tolerance. A panel can lose AUC and keep a safe cut, or hold its AUC and have
the cut land somewhere else entirely, because a threshold is a point on a
distribution and the distribution moves.

The alcohol question
--------------------
NHANES III never asked it in a form comparable to ALQ130, so alcohol is filled
with the training median here -- exactly what the service does for a patient
who leaves it blank. This is therefore a test of the panel answered four
questions out of five, and a floor rather than an estimate: the complete panel
cannot do worse than this. See fetch_nhanes3_general.py.

Run:  python experiments/general_external.py
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

OUT = "experiments/general_external_result.json"
TRAIN_CSV = "data/nhanes_screening_general.csv"
TEST_CSV = "data/nhanes3_general.csv"
TARGET = "recent_cancer"   # the same column name on both sides

DEMO = ["age", "gender"]
PANEL = ["age", "gender", "bmi", "smoking", "alcohol_intake"]

# The same 5-point tolerance the bowel cut was held to on NHANES III, set
# before the numbers were known.
TOLERANCE = 0.05


def fit(tr, feats, seed=0):
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    ytr = tr[TARGET].astype(int)
    model = CalibratedClassifierCV(
        tm.build_ensemble(len(ytr), float(ytr.mean())),
        method="isotonic",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    model.fit(Xtr, ytr)
    return model, med


def transfer(model, med, te, feats):
    # Training medians, never the test cohort's own. The test cohort's medians
    # would leak its distribution into the imputation and flatter the transfer.
    Xte = pd.DataFrame({f: (pd.to_numeric(te[f], errors="coerce")
                            if f in te.columns else np.nan)
                        for f in feats}).fillna(med)
    return model.predict_proba(Xte)[:, 1]


def main():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)

    absent = [f for f in PANEL if f not in te.columns]
    print(f"train  NHANES 2005-2016   n={len(tr):,}  cases={int(tr[TARGET].sum())}"
          f"  ({tr[TARGET].mean():.2%})")
    print(f"test   NHANES III         n={len(te):,}  cases={int(te[TARGET].sum())}"
          f"  ({te[TARGET].mean():.2%})")
    if absent:
        print(f"  not asked in NHANES III, filled with the training median: {absent}")
    print()

    yte = te[TARGET].astype(int).to_numpy()
    results = {}
    for name, feats in (("age and sex only", DEMO), ("full panel", PANEL)):
        model, med = fit(tr, feats)
        p = transfer(model, med, te, feats)
        auc = float(roc_auc_score(yte, p))
        ci = bootstrap_ci(yte, p, roc_auc_score)
        results[name] = {"n_features": len(feats), "external_auc": round(auc, 3),
                         "external_auc_ci": ci}
        print(f"  {name:<20} {len(feats)} features   external AUC {auc:.3f}  "
              f"(95% CI {ci[0]} to {ci[1]})", flush=True)

    gain = results["full panel"]["external_auc"] - results["age and sex only"]["external_auc"]

    # The cut itself is tested in experiments/rule_out_external.py, which owns
    # every rule-out promise on the site and holds them all to the same
    # tolerance. Repeating the calculation here would create a second place for
    # the same number to be computed, and eventually to disagree.

    internal = None
    p_int = "experiments/demographic_gain_result.json"
    if os.path.exists(p_int):
        internal = (json.load(open(p_int)).get("general") or {}).get("gain")

    print("\n" + "=" * 78)
    print(f"  gain over age and sex, transferred to a different decade: {gain:+.3f}")
    if internal is not None:
        print(f"  the same gain measured inside the training survey:       {internal:+.3f}")

    with open(OUT, "w") as f:
        json.dump({"train_n": int(len(tr)), "train_events": int(tr[TARGET].sum()),
                   "test_n": int(len(te)), "test_events": int(yte.sum()),
                   "median_filled_features": absent,
                   "arms": results,
                   "external_gain_over_age_sex": round(float(gain), 3),
                   "internal_gain_for_reference": internal,
                   "rule_out_tested_in": "experiments/rule_out_external.py"},
                  f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
