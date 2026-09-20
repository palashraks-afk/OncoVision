"""
Does the REST of a routine checkup carry anything the blood count did not?

The question
------------
This project's panels read two tests: a blood count and a metabolic panel. Two
panels were withdrawn on the finding that those added nothing to age and sex --
bowel (colon or rectal cancer within eight years) and general (any cancer within
four years). But an ordinary checkup produces more paper than that, and none of
it had ever been pulled: an HbA1c, a lipid panel, a urine albumin-to-creatinine
ratio, a blood pressure, a waist measurement. Those are the lines on the report
a patient is actually handed.

So the withdrawals are retested on the fuller report, on exactly the same rows
and the same targets, changing nothing but the columns.

Pre-registered bar, written before the first number was produced
-----------------------------------------------------------------
A panel is revived only if BOTH hold:

  1. DISCRIMINATION. It beats the stronger of a logistic and an ensemble model
     on age and sex alone, over 5 repeated 5-fold paired splits on identical
     folds, with a 95% interval on the paired difference that excludes zero.

  2. THE RULE-OUT. At the same share of cancers caught, its cut excludes
     materially more adults than a cut on age and sex, interval excluding zero.
     This is the test that withdrew both panels, and passing 1 while failing 2
     is not a revival: the only use of a weak screening panel is telling
     someone they can safely skip a procedure.

Arms, on identical folds and identical rows:
    age_sex        the baseline every claim in this project is measured against
    blood_only     the withdrawn panel's own blood values
    checkup        those plus HbA1c, lipids, urine ACR, blood pressure, waist

Ferritin is reported separately where it exists. Iron-deficiency anaemia is the
classic presentation of a right-sided bowel cancer, but NHANES only ran ferritin
in some cycles and on some subsamples, so it is never imputed for everyone.

A negative here is worth as much as a positive and is written up the same way:
it would say the checkup, not merely the blood count, carries nothing about
these cancers beyond the patient's age.

Run:  python experiments/checkup_panels.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/checkup_panels_result.json"
REPEATS, FOLDS = 5, 5
CATCH = 0.95

COHORTS = {
    "bowel": {"file": "data/nhanes_colorectal_checkup.csv", "target": "colorectal_cancer",
              "label": "colon or rectal cancer within 8 years"},
    "general": {"file": "data/nhanes_general_checkup.csv", "target": "recent_cancer",
                "label": "any cancer within 4 years"},
}

BLOOD = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "mch", "rdw",
         "mpv", "neutrophil_pct", "glucose", "calcium", "bun", "creatinine",
         "protein_total", "albumin", "ast", "alt", "bilirubin", "alkaline_phosphatase"]
CHECKUP_EXTRA = ["hba1c", "cholesterol_total", "hdl", "triglycerides",
                 "urine_albumin_creatinine_ratio", "urine_albumin", "systolic_bp",
                 "diastolic_bp", "bmi", "waist"]
DEMO = ["age", "gender"]


def oof_scores(X, y, kind):
    preds = np.zeros((len(y), REPEATS))
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y):
            m = tm.model_factory(kind, len(tr), float(y.iloc[tr].mean()))
            m.fit(X.iloc[tr], y.iloc[tr])
            preds[te, r] = m.predict_proba(X.iloc[te])[:, 1]
    return preds


def best_of_kinds(X, y):
    best = None
    for kind in ("logistic", "ensemble"):
        preds = oof_scores(X, y, kind)
        aucs = np.array([roc_auc_score(y, preds[:, r]) for r in range(REPEATS)])
        if best is None or aucs.mean() > best[1].mean():
            best = (kind, aucs, preds)
    return best


def rule_out_share(y, p, catch=CATCH):
    """The share of people a cut excludes while still catching `catch` of cases."""
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    total = ys.sum()
    if total == 0:
        return 0.0
    allowed = (1 - catch) * total
    k = int(np.searchsorted(np.cumsum(ys), allowed, side="right"))
    return k / len(ys)


def bootstrap_diff(y, p_panel, p_base, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5:
            continue
        diffs.append(rule_out_share(y[i], p_panel[i]) - rule_out_share(y[i], p_base[i]))
    return (round(float(np.mean(diffs)), 4),
            [round(float(np.percentile(diffs, 2.5)), 4),
             round(float(np.percentile(diffs, 97.5)), 4)])


def run(name, spec):
    df = pd.read_csv(spec["file"])
    y = df[spec["target"]].astype(int).reset_index(drop=True)
    med = df.median(numeric_only=True)
    entry = {"label": spec["label"], "n": int(len(y)), "events": int(y.sum()),
             "prevalence": round(float(y.mean()), 4), "arms": {}}
    print(f"\n{name}: {spec['label']} -- {len(y):,} adults, {int(y.sum())} cases")

    fitted = {}
    for arm, cols in (("age_sex", DEMO), ("blood_only", DEMO + BLOOD),
                      ("checkup", DEMO + BLOOD + CHECKUP_EXTRA)):
        cols = [c for c in cols if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0)
        X = X.reset_index(drop=True)
        kind, aucs, preds = best_of_kinds(X, y)
        fitted[arm] = (preds, aucs)
        entry["arms"][arm] = {"features": len(cols), "model_kind": kind,
                              "auc": round(float(aucs.mean()), 4)}
        print(f"  {arm:<12} {len(cols):>2} inputs  {kind:<9} AUC {aucs.mean():.4f}")

    base_preds, base_aucs = fitted["age_sex"]
    for arm in ("blood_only", "checkup"):
        preds, aucs = fitted[arm]
        d = aucs - base_aucs
        ci = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
        share, share_ci = bootstrap_diff(y, preds[:, 0], base_preds[:, 0])
        gain_ok, rule_ok = bool(ci[0] > 0), bool(share_ci[0] > 0)
        entry["arms"][arm].update({
            "gain_over_age_sex": round(float(d.mean()), 4), "gain_ci": ci,
            "beats_age_sex": gain_ok,
            "extra_share_ruled_out_at_matched_catch": share, "extra_share_ci": share_ci,
            "rule_out_beats_age": rule_ok, "revived": bool(gain_ok and rule_ok)})
        print(f"  {arm} vs age and sex: {d.mean():+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f}) "
              f"-> {'beats it' if gain_ok else 'NOT SHOWN'};  rule-out {share:+.1%} "
              f"({share_ci[0]:+.1%} to {share_ci[1]:+.1%}) -> "
              f"{'beats age' if rule_ok else 'NOT SHOWN'};  REVIVED: "
              f"{entry['arms'][arm]['revived']}")

    if "ferritin" in df.columns:
        iron = df[df["ferritin"].notna()].reset_index(drop=True)
        yi = iron[spec["target"]].astype(int)
        if yi.sum() >= 20:
            imed = iron.median(numeric_only=True)
            got = {}
            for arm, cols in (("age_sex", DEMO),
                              ("checkup_iron", DEMO + BLOOD + CHECKUP_EXTRA + ["ferritin"])):
                cols = [c for c in cols if c in iron.columns]
                X = iron[cols].apply(pd.to_numeric, errors="coerce").fillna(imed).fillna(0.0)
                got[arm] = best_of_kinds(X, yi)[1]
            d = got["checkup_iron"] - got["age_sex"]
            entry["ferritin_subset"] = {
                "n": int(len(yi)), "events": int(yi.sum()),
                "gain": round(float(d.mean()), 4),
                "gain_ci": [round(float(np.percentile(d, 2.5)), 4),
                            round(float(np.percentile(d, 97.5)), 4)],
                "beats_age_sex": bool(np.percentile(d, 2.5) > 0)}
            print(f"  ferritin subset: {int(yi.sum())} cases, gain {d.mean():+.4f}")
        else:
            entry["ferritin_subset"] = {"events": int(yi.sum()), "too_few": True}
            print(f"  ferritin subset: {int(yi.sum())} cases, too few to score")
    entry["revived"] = bool(entry["arms"]["checkup"].get("revived"))
    return entry


def main():
    print("The rest of the checkup, tested against the two withdrawn panels")
    results = {name: run(name, spec) for name, spec in COHORTS.items()}
    results["any_revived"] = bool(any(v["revived"] for v in results.values()
                                      if isinstance(v, dict)))
    print(f"\nVERDICT: "
          + ("a withdrawn panel is revived on the fuller checkup"
             if results["any_revived"] else
             "neither panel is revived. The rest of the checkup carries nothing about "
             "these cancers beyond age and sex."))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
