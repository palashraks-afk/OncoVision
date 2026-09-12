"""
The mammogram-report breast panel, judged on the split it never saw.

The random 20% that evaluate.py cuts from the training sample is an internal
number. The one that decides this panel is BCSC's own validation split: 597,859
mammograms that the consortium set aside, that were not sampled into training,
and that nothing here fitted, calibrated or thresholded on.

Measured
--------
    discrimination   weighted AUC on all 597,859 mammograms
    gain             over an age-only model fitted on the same training rows,
                     with a paired interval, because age carries most of
                     screening risk and a panel has to earn its extra questions
    rule-out         the shipped cut applied unchanged
    subgroups        AUC by race and ethnicity, which the cohort records and the
                     panel does not use

The validation rows are a frequency table, so every statistic is weighted by
count, and intervals come from a Poisson bootstrap: each row's count is redrawn
as Poisson(count), which resamples mammograms rather than covariate patterns.
Resampling the 100,195 rows directly would treat a pattern seen by 4,000 women
the same as one seen by one woman.

Run:  python experiments/bcsc_validation.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/bcsc_validation_result.json"
NAME = "breast_screening"
VALID = "data/bcsc_breast_validation.csv"
BOOT = 500
TOLERANCE = 0.05          # the same tolerance every other shipped cut meets

RACE = {1: "White", 2: "Asian or Pacific Islander", 3: "Black",
        4: "Native American", 5: "Other or mixed"}
# Below this many cancers a subgroup AUC is reported as unmeasurable rather
# than as a number that looks precise.
MIN_GROUP_EVENTS = 50


def wauc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    med = pd.Series(bundle["feature_medians"])
    ranges = bundle.get("feature_ranges") or {}

    def matrix(frame):
        X = tm.build_features(cfg, frame).fillna(med)
        for col, (lo, hi) in ranges.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X[bundle["feature_names"]]

    va = pd.read_csv(VALID)
    y = va["cancer"].astype(int).to_numpy()
    w = va["count"].to_numpy(dtype=float)
    p = bundle["model"].predict_proba(matrix(va))[:, 1]

    # Age-only comparator, fitted on the same training sample the panel saw.
    tr = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    Xtr = tm.build_features(cfg, tr)[["age"]]
    age_med = Xtr["age"].median()
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    base.fit(Xtr.fillna(age_med), tr["cancer"].astype(int))
    b = base.predict_proba(tm.build_features(cfg, va)[["age"]].fillna(age_med))[:, 1]

    auc, base_auc = wauc(y, p, w), wauc(y, b, w)

    rng = np.random.default_rng(0)
    aucs, gains = [], []
    for _ in range(BOOT):
        wb = rng.poisson(w).astype(float)
        a = wauc(y, p, wb)
        aucs.append(a)
        gains.append(a - wauc(y, b, wb))
    ci = [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]
    gci = [round(float(np.percentile(gains, 2.5)), 3), round(float(np.percentile(gains, 97.5)), 3)]

    n_mammo, n_cancer = int(w.sum()), int(w[y == 1].sum())
    print(f"BCSC validation split: {n_mammo:,} mammograms, {n_cancer:,} cancers\n")
    print(f"  panel      AUC {auc:.3f}  (95% CI {ci[0]} to {ci[1]})")
    print(f"  age only   AUC {base_auc:.3f}")
    print(f"  gain       {auc - base_auc:+.3f}  (95% CI {gci[0]:+.3f} to {gci[1]:+.3f})")

    result = {
        "n_mammograms": n_mammo, "n_cancers": n_cancer,
        "auc": round(auc, 3), "auc_ci": ci,
        "age_only_auc": round(base_auc, 3),
        "gain": round(auc - base_auc, 3), "gain_ci": gci,
        "gain_beats_age": bool(gci[0] > 0),
    }

    ro = (bundle.get("metrics") or {}).get("rule_out")
    if ro:
        out = p < ro["threshold"]
        caught = float(w[(y == 1) & ~out].sum() / w[y == 1].sum())
        excluded = float(w[out].sum() / w.sum())
        missed = int(round(w[(y == 1) & out].sum()))
        kept = caught >= ro["sensitivity"] - TOLERANCE
        result["rule_out"] = {
            "promised_catch_rate": ro["sensitivity"],
            "actual_catch_rate": round(caught, 3),
            "promised_share_excluded": ro["share_ruled_out"],
            "actual_share_excluded": round(excluded, 3),
            "cases_in_test_cohort": n_cancer,
            "cases_ruled_out_wrongly": missed,
            "promise_kept": bool(kept), "tolerance": TOLERANCE,
        }
        print(f"\n  rule-out cut, unchanged: promised catch {ro['sensitivity']:.1%} / "
              f"exclude {ro['share_ruled_out']:.1%}; delivered {caught:.1%} / "
              f"{excluded:.1%}; {missed:,} of {n_cancer:,} cancers ruled out "
              f"-> {'holds' if kept else 'BROKEN'}")
    else:
        result["rule_out"] = None
        print(f"\n  no rule-out cut ships: "
              f"{(bundle.get('metrics') or {}).get('no_rule_out_reason', '')}")

    groups = {}
    print("\n  by race and ethnicity (recorded, never a feature):")
    for code, label in RACE.items():
        m = va["race"].to_numpy() == code
        ev = int(w[m & (y == 1)].sum())
        if ev < MIN_GROUP_EVENTS or len(np.unique(y[m])) < 2:
            groups[label] = {"events": ev, "auc": None}
            print(f"    {label:<28} {ev:>5} cancers  too few to measure")
            continue
        g = wauc(y[m], p[m], w[m])
        groups[label] = {"events": ev, "auc": round(g, 3),
                         "materially_worse": bool(g < auc - 0.05)}
        print(f"    {label:<28} {ev:>5} cancers  AUC {g:.3f}"
              f"{'   <-- more than 0.05 below overall' if g < auc - 0.05 else ''}")
    result["groups"] = groups

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
