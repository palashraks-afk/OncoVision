"""
Does the mammogram breast panel beat what the clinic already has in front of it?

Why this exists
---------------
The panel's 0.623 on BCSC's validation split was called modest, and against age
alone it gains +0.028. But age alone is not what a clinician holds after a
mammogram: the report also prints a BI-RADS density grading, and density is the
single best-known risk factor after age. A panel that only matched age plus
density would be asking six extra history questions for nothing, which is the
test that withdrew the bowel and general panels.

So three comparators, each fitted on the same training rows the panel saw and
scored on BCSC's own 597,859-mammogram validation split:

    age                      what a birthday tells you
    age + density            what the mammogram report already says
    age + density + BMI      the above plus a measurement taken at any visit

Every statistic is weighted by row count and every interval is a Poisson
bootstrap, exactly as in bcsc_validation.py, because the validation rows are a
frequency table.

Adoption bar, set before running: the panel earns its extra questions only if
its gain over age AND density has a 95% interval that excludes zero.

Run:  python experiments/breast_vs_clinic.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/breast_vs_clinic_result.json"
NAME = "breast_screening"
VALID = "data/bcsc_breast_validation.csv"
BOOT = 500

COMPARATORS = {
    "age": ["age"],
    "age_density": ["age", "breast_density"],
    "age_density_bmi": ["age", "breast_density", "bmi"],
}


def wauc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    med = pd.Series(bundle["feature_medians"])
    ranges = bundle.get("feature_ranges") or {}

    tr = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    va = pd.read_csv(VALID)
    y_tr = tr["cancer"].astype(int)
    X_tr = tm.build_features(cfg, tr).fillna(med)
    X_va_raw = tm.build_features(cfg, va).fillna(med)

    X_va = X_va_raw.copy()
    for col, (lo, hi) in ranges.items():
        if col in X_va.columns:
            X_va[col] = X_va[col].clip(lo, hi)
    p = bundle["model"].predict_proba(X_va[bundle["feature_names"]])[:, 1]

    y = va["cancer"].astype(int).to_numpy()
    w = va["count"].to_numpy(dtype=float)
    rate = float(y_tr.mean())

    # Each comparator gets the same two candidate model kinds the panels choose
    # between, and the STRONGER one on the validation split is kept. Picking the
    # comparator's best on the test data can only shrink the panel's gain, which
    # is the conservative direction: a weak baseline is what invented the bowel
    # panel's gain in the first place.
    scores, kinds = {}, {}
    for label, cols in COMPARATORS.items():
        best, best_auc, best_kind = None, -1.0, None
        for kind in ("logistic", "ensemble"):
            m = tm.model_factory(kind, len(y_tr), rate)
            m.fit(X_tr[cols], y_tr)
            s = m.predict_proba(X_va_raw[cols])[:, 1]
            a = wauc(y, s, w)
            if a > best_auc:
                best, best_auc, best_kind = s, a, kind
        scores[label], kinds[label] = best, best_kind

    rng = np.random.default_rng(0)
    boots = {label: [] for label in COMPARATORS}
    panel_boot = []
    for _ in range(BOOT):
        wb = rng.poisson(w).astype(float)
        a = wauc(y, p, wb)
        panel_boot.append(a)
        for label, s in scores.items():
            boots[label].append(a - wauc(y, s, wb))

    auc = wauc(y, p, w)
    n_mammo, n_cancer = int(w.sum()), int(w[y == 1].sum())
    result = {"n_mammograms": n_mammo, "n_cancers": n_cancer, "panel_auc": round(auc, 3),
              "panel_auc_ci": [round(float(np.percentile(panel_boot, 2.5)), 3),
                               round(float(np.percentile(panel_boot, 97.5)), 3)],
              "comparators": {}}

    print(f"BCSC validation split: {n_mammo:,} mammograms, {n_cancer:,} cancers")
    print(f"  panel  AUC {auc:.3f}  ({len(bundle['feature_names'])} inputs)\n")
    for label, cols in COMPARATORS.items():
        base_auc = wauc(y, scores[label], w)
        lo = round(float(np.percentile(boots[label], 2.5)), 3)
        hi = round(float(np.percentile(boots[label], 97.5)), 3)
        beats = bool(lo > 0)
        result["comparators"][label] = {
            "features": cols, "model_kind": kinds[label], "auc": round(base_auc, 3),
            "gain": round(auc - base_auc, 3), "gain_ci": [lo, hi], "panel_beats": beats}
        print(f"  vs {label:<16} ({kinds[label]:<8}) AUC {base_auc:.3f}   "
              f"gain {auc - base_auc:+.3f}  95% CI {lo:+.3f} to {hi:+.3f}   "
              f"{'panel earns it' if beats else 'NOT SHOWN'}")

    key = result["comparators"]["age_density"]
    result["earns_its_questions"] = key["panel_beats"]
    print(f"\n  BAR: beat age and the density grading already on the report -> "
          f"{'PASSED' if key['panel_beats'] else 'FAILED'}")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
