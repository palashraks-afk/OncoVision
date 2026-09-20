"""
The revived general panel, on a cohort from another decade.

The decision this makes
-----------------------
On the continuous survey the general panel, given the rest of the checkup,
cleared both internal bars: +0.016 over the stronger age-and-sex model and a
rule-out cut excluding 4.9% more adults at the same share of cancers caught.
Section 4.2 of the paper is the reason that is not enough. The prospective
panel gained inside NHANES, held on leave-one-cycle-out, and reversed on
NHANES III, because cycles of one survey share a protocol, a laboratory
contract and a pipeline. NHANES III shares none of them: different decade,
different instruments, different people.

Pre-registered, written before this was run
-------------------------------------------
The general panel is revived only if BOTH hold on NHANES III:

  1. its gain over the stronger age-and-sex model, fitted on the training rows
     and applied here, has a 95% interval excluding zero; and
  2. at the same share of cancers caught, its rule-out cut excludes materially
     more adults than a cut on age and sex, interval excluding zero.

Failing either leaves the panel withdrawn, and the write-up says the checkup
was tried and did not survive transfer.

Two lines of the report are missing here: NHANES III measured triglycerides
only in the fasting subsample and published no urinary creatinine, so the
albumin-to-creatinine ratio cannot be built. Both are filled with the training
median, exactly as the service does for a patient who does not have them, which
makes this a test of the panel as used by someone missing two values.

Run:  python experiments/checkup_external.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/checkup_external_result.json"
TRAIN = "data/nhanes_general_checkup.csv"
EXTERNAL = "data/nhanes3_general_checkup.csv"
TARGET = "recent_cancer"
CATCH = 0.95
N_BOOT = 2000

DEMO = ["age", "gender"]
BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "mch", "rdw",
         "mpv", "glucose", "calcium", "bun", "creatinine", "protein_total", "albumin",
         "ast", "alt", "bilirubin", "alkaline_phosphatase"]
CHECKUP_EXTRA = ["hba1c", "cholesterol_total", "hdl", "triglycerides",
                 "urine_albumin_creatinine_ratio", "urine_albumin", "systolic_bp",
                 "diastolic_bp", "bmi", "waist"]


def rule_out_share(y, p, catch=CATCH):
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    total = ys.sum()
    if total == 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(ys), (1 - catch) * total, side="right"))
    return k / len(ys)


def paired_boot(y, a, b, fn, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].min() == y[i].max():
            continue
        out.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    return (round(float(np.mean(out)), 4),
            [round(float(np.percentile(out, 2.5)), 4),
             round(float(np.percentile(out, 97.5)), 4)])


def fit_best(X_tr, y_tr, X_te):
    """The stronger of the two kinds, chosen by cross-validated AUC on training."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    best = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(y_tr), float(y_tr.mean()))
        cvauc = float(np.mean(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")))
        if best is None or cvauc > best[0]:
            m2 = tm.model_factory(kind, len(y_tr), float(y_tr.mean()))
            m2.fit(X_tr, y_tr)
            best = (cvauc, kind, m2.predict_proba(X_te)[:, 1], m2)
    return best


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(EXTERNAL)
    y_tr = tr[TARGET].astype(int)
    y_te = te[TARGET].astype(int).to_numpy()
    med = tr.median(numeric_only=True)

    print(f"Train: {len(tr):,} adults, {int(y_tr.sum())} cancers within 4 years "
          f"(NHANES 2005-2016)")
    print(f"Test:  {len(te):,} adults, {int(y_te.sum())} cancers within 4 years "
          f"(NHANES III, 1988-1994)\n")

    arms = {"age_sex": DEMO, "blood_only": DEMO + BLOOD,
            "checkup": DEMO + BLOOD + CHECKUP_EXTRA}
    results = {"n_train": int(len(tr)), "events_train": int(y_tr.sum()),
               "n_test": int(len(te)), "events_test": int(y_te.sum()), "arms": {}}
    scores = {}
    missing_note = {}
    for name, cols in arms.items():
        X_tr = tr[cols].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0)
        X_te = pd.DataFrame({c: pd.to_numeric(te[c], errors="coerce")
                             if c in te.columns else np.nan for c in cols})
        missing_note[name] = [c for c in cols if c not in te.columns or X_te[c].isna().all()]
        X_te = X_te.fillna(med).fillna(0.0)
        cvauc, kind, p, _ = fit_best(X_tr, y_tr, X_te)
        scores[name] = p
        results["arms"][name] = {
            "features": len(cols), "model_kind": kind,
            "internal_cv_auc": round(cvauc, 4),
            "external_auc": round(float(roc_auc_score(y_te, p)), 4),
            "external_auc_ci": bootstrap_ci(y_te, p, roc_auc_score),
            "filled_with_training_median": missing_note[name]}
        print(f"  {name:<12} {len(cols):>2} inputs  {kind:<9} internal CV {cvauc:.4f}   "
              f"NHANES III {roc_auc_score(y_te, p):.4f}"
              + (f"   (missing there: {', '.join(missing_note[name])})"
                 if missing_note[name] else ""))

    base = scores["age_sex"]
    for name in ("blood_only", "checkup"):
        p = scores[name]
        gain, gain_ci = paired_boot(y_te, p, base, roc_auc_score)
        share, share_ci = paired_boot(y_te, p, base, rule_out_share, seed=1)
        gain_ok, rule_ok = bool(gain_ci[0] > 0), bool(share_ci[0] > 0)
        results["arms"][name].update({
            "external_gain_over_age_sex": gain, "external_gain_ci": gain_ci,
            "beats_age_sex_externally": gain_ok,
            "external_extra_share_ruled_out": share, "external_extra_share_ci": share_ci,
            "rule_out_beats_age_externally": rule_ok,
            "survives_transfer": bool(gain_ok and rule_ok)})
        print(f"\n  {name} on NHANES III, against age and sex:")
        print(f"    discrimination {gain:+.4f}  95% CI {gain_ci[0]:+.4f} to {gain_ci[1]:+.4f}"
              f"  -> {'beats it' if gain_ok else 'NOT SHOWN'}")
        print(f"    rule-out at {CATCH:.0%} caught: {share:+.1%} more adults excluded "
              f"({share_ci[0]:+.1%} to {share_ci[1]:+.1%}) -> "
              f"{'beats a cut on age' if rule_ok else 'NOT SHOWN'}")
        print(f"    SURVIVES TRANSFER: {results['arms'][name]['survives_transfer']}")

    results["revived"] = bool(results["arms"]["checkup"]["survives_transfer"])
    print(f"\n  VERDICT: the general panel is "
          + ("REVIVED: the rest of the checkup survives a change of decade."
             if results["revived"] else
             "still withdrawn. The checkup's internal gain does not survive transfer."))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
