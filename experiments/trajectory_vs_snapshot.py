"""
Does a person's own trend beat a single blood draw?

The question this settles
-------------------------
Every panel in this project reads one blood draw, and three of them were
withdrawn because one draw carried nothing beyond the patient's age. The
remaining version of the idea is that the signal is not in the level but in the
MOVEMENT: a haemoglobin of 13 means little, and a haemoglobin that has walked
down from 15 over two years means something. ColonFlag, the one deployed model
of this kind, is built exactly that way.

Three arms, on identical folds and the same patients:

    age_sex      the baseline every claim in this project is measured against
    snapshot     the latest value of each analyte -- what the panels read today
    trajectory   those values plus what they DID: the change from the person's
                 own earliest reading, the slope per year, the spread and the
                 range

Pre-registered, written before this could be run on anything real
-----------------------------------------------------------------
The trajectory idea is adopted only if BOTH hold, over 5 repeated 5-fold
paired splits:

  1. it beats age and sex, 95% interval on the paired difference excluding zero
  2. it beats the snapshot arm on the same folds, interval excluding zero

Failing 2 while passing 1 means repeat measurement added nothing over a single
draw, which is the finding, not a disappointment: it would say the movement is
already visible in the level.

The floor stands. Below MIN_EVENTS cases this prints what it has and calls it
nothing, because a handful of events cannot separate three arms, and the
cervical panel was withdrawn here for exactly that mistake.

Run:  python experiments/trajectory_vs_snapshot.py
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

OUT = "experiments/trajectory_vs_snapshot_result.json"
COHORT = "data/mimic_trajectory.csv"
TARGET = "cancer"
REPEATS, FOLDS = 5, 5
MIN_EVENTS = 96          # the event floor used throughout this project

SNAPSHOT_SUFFIXES = ("_last",)
TRAJECTORY_SUFFIXES = ("_last", "_baseline", "_delta", "_rel_delta", "_min", "_max",
                       "_sd", "_slope_per_year", "_n", "_span_days")
DEMO = ["age", "gender"]


def arms(df):
    snap = [c for c in df.columns if c.endswith(SNAPSHOT_SUFFIXES)]
    traj = [c for c in df.columns if c.endswith(TRAJECTORY_SUFFIXES)]
    return {"age_sex": DEMO,
            "snapshot": DEMO + snap,
            "trajectory": DEMO + traj}


def fold_aucs(X, y, kind):
    out = []
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        preds = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            m = tm.model_factory(kind, len(tr), float(y.iloc[tr].mean()))
            m.fit(X.iloc[tr], y.iloc[tr])
            preds[te] = m.predict_proba(X.iloc[te])[:, 1]
        out.append(roc_auc_score(y, preds))
    return np.array(out)


def best_of_kinds(X, y):
    best = None
    for kind in ("logistic", "ensemble"):
        a = fold_aucs(X, y, kind)
        if best is None or a.mean() > best[1].mean():
            best = (kind, a)
    return best


def main():
    if not os.path.exists(COHORT):
        print(f"{COHORT} is missing. Run fetch_mimic_trajectory.py first.")
        return 1
    df = pd.read_csv(COHORT)
    y = df[TARGET].astype(int)
    spec = arms(df)
    events = int(y.sum())
    print(f"{len(df):,} patients, {events} first cancer diagnoses "
          f"({y.mean():.1%})")
    for name, cols in spec.items():
        print(f"  {name:<11} {len([c for c in cols if c in df.columns]):>3} inputs")

    result = {"n": int(len(df)), "events": events, "min_events": MIN_EVENTS,
              "arms": {name: len([c for c in cols if c in df.columns])
                       for name, cols in spec.items()}}

    if events < MIN_EVENTS:
        result["verdict"] = (f"{events} cases is below the {MIN_EVENTS} event floor, so no "
                             f"comparison is run. This is the pipeline, not a result.")
        print(f"\n  {result['verdict']}")
        print("  Get the full cohort with:")
        print("      python fetch_mimic_trajectory.py --dir path/to/mimiciv/hosp")
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {OUT}")
        return 0

    med = df.median(numeric_only=True)
    fitted = {}
    for name, cols in spec.items():
        cols = [c for c in cols if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0)
        kind, aucs = best_of_kinds(X, y)
        fitted[name] = aucs
        result["arms"][name] = {"features": len(cols), "model_kind": kind,
                                "auc": round(float(aucs.mean()), 4)}
        print(f"  {name:<11} {kind:<9} AUC {aucs.mean():.4f}")

    for against in ("age_sex", "snapshot"):
        d = fitted["trajectory"] - fitted[against]
        ci = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
        key = f"trajectory_vs_{against}"
        result[key] = {"gain": round(float(d.mean()), 4), "gain_ci": ci,
                       "beats": bool(ci[0] > 0)}
        print(f"  trajectory vs {against}: {d.mean():+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f}) "
              f"-> {'beats it' if ci[0] > 0 else 'NOT SHOWN'}")

    adopted = bool(result["trajectory_vs_age_sex"]["beats"]
                   and result["trajectory_vs_snapshot"]["beats"])
    result["adopted"] = adopted
    result["verdict"] = ("Reading a person's own trend beats reading one draw, and beats age "
                         "and sex." if adopted else
                         "The trend did not clear both bars, so one draw is not improved on here.")
    print(f"\n  VERDICT: {result['verdict']}")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
