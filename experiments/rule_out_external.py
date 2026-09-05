"""
Does the rule-out threshold still rule out on a cohort from another decade?

Why this has to be asked
------------------------
The bowel panel ships a rule-out cut chosen to catch 95 percent of cases while
excluding 40 percent of people from consideration for a colonoscopy. That cut
was chosen on out-of-fold predictions inside NHANES 2005-2014.

A threshold is a promise about a rate, and a rate estimated on one survey is
exactly the kind of quantity this project has already watched fail to transfer.
The prospective mortality panel gained +0.013 inside its own survey, scored
0.837 on leave-one-cycle-out, and lost 0.013 on a cohort from a different
decade. If the rule-out cut behaves the same way, then telling a patient "this
would leave you out of further testing" is a promise the panel cannot keep.

The consequence of being wrong here is worse than an inflated AUC. A rule-out
that silently catches 80 percent instead of 95 sends one case in five home.

    train    NHANES 2005-2014, 23,794 adults, 96 colorectal cancers
    test     NHANES III 1988-1994, 14,499 adults, 56 colorectal cancers

Both the model and the threshold come from the training cohort. Nothing from the
test cohort touches fitting, calibration, imputation or the cut.

Read the result as: at the cut the panel promises, how many cases does it
actually catch on people it has never seen, and how many people does it actually
exclude?

Run:  python experiments/rule_out_external.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/rule_out_external_result.json"

PANELS = {
    "colorectal": {
        "test_csv": "data/nhanes3_colorectal.csv",
        "target": "colorectal_cancer",
    },
}

# How far the caught-case rate may fall before the promise is broken. A cut sold
# as catching 95 in 100 that catches 88 is a different product.
TOLERANCE = 0.05


def main():
    results = {}
    for panel, cfg in PANELS.items():
        bundle = joblib.load(f"models/model_{panel}.joblib")
        ro = (bundle.get("metrics") or {}).get("rule_out")
        if not ro:
            print(f"{panel}: no rule-out point in the bundle, run train_models.py")
            continue
        if not os.path.exists(cfg["test_csv"]):
            print(f"{panel}: {cfg['test_csv']} missing, run its fetcher")
            continue

        te = pd.read_csv(cfg["test_csv"])
        feats = bundle["feature_names"]
        missing = [f for f in feats if f not in te.columns]
        if missing:
            print(f"{panel}: test cohort lacks {missing}")
            continue

        med = bundle["feature_medians"]
        ranges = bundle.get("feature_ranges") or {}
        X = te[feats].apply(pd.to_numeric, errors="coerce")
        # Training medians and training ranges, never the test cohort's own.
        X = X.fillna(pd.Series(med))
        for f in feats:
            if f in ranges:
                X[f] = X[f].clip(ranges[f][0], ranges[f][1])
        y = te[cfg["target"]].astype(int).to_numpy()

        p = bundle["model"].predict_proba(X)[:, 1]
        cut = float(ro["threshold"])
        flagged = p >= cut

        caught = float(flagged[y == 1].mean()) if y.sum() else float("nan")
        excluded = float((~flagged).mean())
        missed = int((~flagged & (y == 1)).sum())

        promised = ro["sensitivity"]
        kept = bool(caught >= promised - TOLERANCE)

        results[panel] = {
            "cut": round(cut, 6),
            "promised_catch_rate": promised,
            "actual_catch_rate": round(caught, 3),
            "promised_share_excluded": ro["share_ruled_out"],
            "actual_share_excluded": round(excluded, 3),
            "cases_in_test_cohort": int(y.sum()),
            "cases_ruled_out_wrongly": missed,
            "promise_kept": kept,
            "tolerance": TOLERANCE,
        }

        print(f"=== {panel} ===")
        print(f"  cut chosen on NHANES 2005-2014: {cut * 100:.3f}%\n")
        print(f"  {'':<22}{'promised':>12}{'actual on NHANES III':>24}")
        print(f"  {'catches':<22}{promised:>12.3f}{caught:>24.3f}")
        print(f"  {'excludes':<22}{ro['share_ruled_out']:>12.3f}{excluded:>24.3f}")
        print(f"\n  {missed} of {int(y.sum())} cancers were ruled out when they "
              f"should not have been")
        print(f"  -> the promise {'HOLDS' if kept else 'DOES NOT HOLD'} "
              f"(tolerance {TOLERANCE})")

    print("\n" + "=" * 78)
    broken = [k for k, v in results.items() if not v["promise_kept"]]
    if broken:
        print(f"  rule-out promises that do not transfer: {broken}")
        print("  a cut sold as catching 95 in 100 that catches fewer is a different")
        print("  product, and the number on the card has to change")
    else:
        print("  every rule-out promise held on a cohort from another decade")

    with open(OUT, "w") as f:
        json.dump({"panels": results, "promises_broken": broken}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
