"""
The central question, asked on a design that can actually answer it.

Oncovision's premise is that early detection signal lives in a combination of
routine blood values being slightly unusual together, not in one value crossing
a reference limit. Every previous test of that premise in this project used a
cohort where the blood and the answer were recorded at the same time, so a
positive result could always be read two ways: the bloodwork predicts the
cancer, or the cancer already changed the bloodwork.

The NHANES-NDI linkage separates them. The sample is drawn, years pass, and the
death certificate arrives later from a different agency. If routine chemistry
carries early-detection signal, it should show up here. If it does not, the
honest conclusion is that it does not, and that is worth knowing too.

    n            33,834 adults, 1999-2014, no cancer diagnosis at baseline
    outcome      death from malignant neoplasm within 60 months of the draw
    events       339, a 1.0% rate that is stable across all eight cycles

Arms
----
    A  age and sex                    the baseline anything must beat
    B  + BMI, smoking, alcohol        what a questionnaire gets you for free
    C  + complete blood count         the cheapest panel in medicine
    D  + metabolic and liver panel    the rest of a routine draw
    E  everything                     the full Oncovision feature set

The comparison that matters is E against A. B exists to separate "the lab report
helped" from "asking about smoking helped", which is the confound that made the
old general panel look better than it was.

Judged on repeated paired cross-validation over identical folds, which is the
arbiter used everywhere else in this project, plus a leave-one-cycle-out check:
train on seven NHANES cycles, test on the eighth. That last one is close to
external validation, because cycles differ in assay methods, field staff and
population composition.

Run:  python experiments/prospective_mortality.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

REPEATS = 5
OUT = "experiments/prospective_mortality_result.json"
CSV = "data/nhanes_cancer_mortality.csv"

DEMO = ["age", "gender"]
LIFESTYLE = ["bmi", "smoking", "alcohol_intake"]
CBC = ["wbc", "rbc", "hemoglobin", "platelets", "hematocrit", "mcv", "mch",
       "rdw", "mpv", "neutrophil_pct"]
CHEM = ["glucose", "calcium", "bun", "creatinine", "protein_total", "albumin",
        "ast", "alt", "bilirubin", "alkaline_phosphatase", "ggt"]

ARMS = {
    "A age and sex": DEMO,
    "B + lifestyle": DEMO + LIFESTYLE,
    "C + blood count": DEMO + LIFESTYLE + CBC,
    "D + chemistry": DEMO + LIFESTYLE + CHEM,
    "E everything": DEMO + LIFESTYLE + CBC + CHEM,
}


def prep(df, feats):
    X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    return X.fillna(X.median())


def oof(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return p


def main():
    df = pd.read_csv(CSV)
    y = df["cancer_death"].astype(int).reset_index(drop=True)
    print(f"{len(y):,} adults, {int(y.sum())} cancer deaths within 60 months "
          f"({y.mean():.2%})")
    print(f"cycles: {df['cycle'].nunique()}, median follow-up "
          f"{df['followup_months'].median():.0f} months\n")

    scores, last_pred = {}, {}
    for name, feats in ARMS.items():
        X = prep(df, feats)
        aucs = []
        for s in range(REPEATS):
            p = oof(X, y, s)
            aucs.append(roc_auc_score(y, p))
            if s == 0:
                last_pred[name] = p
        arr = np.array(aucs)
        scores[name] = arr
        print(f"  {name:<20} {len(feats):>2} features   AUC {arr.mean():.3f}  "
              f"(range {arr.min():.3f} to {arr.max():.3f})", flush=True)

    base = scores["A age and sex"]
    results = {}
    print()
    for name, arr in scores.items():
        d = arr - base
        ci = bootstrap_ci(np.asarray(y), last_pred[name], roc_auc_score)
        results[name] = {
            "n_features": len(ARMS[name]),
            "auc": round(float(arr.mean()), 3),
            "auc_ci": ci,
            "gain_over_age_sex": round(float(d.mean()), 3),
            "wins": int((d > 0).sum()),
            "repeats": REPEATS,
        }
        if name != "A age and sex":
            print(f"  {name:<20} gain over age and sex {d.mean():+.3f}  "
                  f"wins {int((d > 0).sum())}/{REPEATS}")

    full_gain = results["E everything"]["gain_over_age_sex"]

    # Leave one cycle out. Cycles differ in assay method, field staff and
    # population, so this is the closest thing to an external test available
    # inside one survey.
    print("\nleave one cycle out, full feature set")
    X = prep(df, ARMS["E everything"])
    cycles, loco = df["cycle"].reset_index(drop=True), {}
    for cyc in sorted(cycles.unique()):
        te = (cycles == cyc).values
        tr = ~te
        if y[te].sum() < 5:
            continue
        model = CalibratedClassifierCV(
            tm.build_ensemble(int(tr.sum()), float(y[tr].mean())),
            method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=0))
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        a = float(roc_auc_score(y[te], p))
        loco[cyc] = round(a, 3)
        print(f"  held out {cyc}   n={int(te.sum()):<5} deaths={int(y[te].sum()):<3} "
              f"AUC {a:.3f}", flush=True)

    mean_loco = float(np.mean(list(loco.values()))) if loco else float("nan")
    print(f"\n  mean across held-out cycles: {mean_loco:.3f}")

    print("\n" + "=" * 78)
    verdict = (
        "routine bloodwork carries prospective signal beyond age and sex"
        if full_gain >= 0.02 else
        "routine bloodwork adds little beyond age and sex on this endpoint"
    )
    print(f"  full panel {results['E everything']['auc']}, age and sex "
          f"{results['A age and sex']['auc']}, gain {full_gain:+.3f}")
    print(f"  -> {verdict}")

    with open(OUT, "w") as f:
        json.dump({"n": int(len(y)), "events": int(y.sum()),
                   "prevalence": round(float(y.mean()), 4),
                   "horizon_months": 60, "arms": results,
                   "leave_one_cycle_out": loco,
                   "mean_leave_one_cycle_out": round(mean_loco, 3),
                   "gain_over_age_sex": full_gain,
                   "verdict": verdict}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
