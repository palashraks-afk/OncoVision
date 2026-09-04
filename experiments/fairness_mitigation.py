"""
Can the racial gaps be closed, or only reported?

The gaps
--------
Measuring fairness found three panels where a group comes in materially below
the overall figure, and on two of them the weakest group is the one with the
higher mortality from that cancer:

    general      Other Hispanic        0.672 against 0.757
    lung         Non-Hispanic Black    0.756 against 0.823
    bowel        Non-Hispanic Black    0.752 against 0.804

Reporting that is better than not knowing it. It is not a fix.

Three things are tried here
---------------------------
    baseline      what ships today
    reweighted    training rows weighted so every racial group contributes
                  equally, instead of the model optimising mostly for the
                  largest group. Non-Hispanic White is 44 percent of the bowel
                  cohort and supplies 70 percent of its cases, so an unweighted
                  fit is largely fitted to them.
    per-group
    threshold     one model, but the operating point chosen separately within
                  each group so that sensitivity is equalised

The third needs stating carefully. Using race to pick a THRESHOLD is not the
same as using it as a model feature: the model never sees it, and the effect is
to equalise who gets caught rather than to encode a population average as
biology. It is still contested, and it is measured here rather than adopted
silently.

What would count as success
---------------------------
The worst group improves by a margin that survives repetition, without the
overall figure collapsing. A mitigation that lifts one group by ruining
everybody is not a fix, and neither is one that only moves things by noise.

Run:  python experiments/fairness_mitigation.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

REPEATS = 5
MIN_EVENTS = 10
OUT = "experiments/fairness_mitigation_result.json"

SOURCES = {
    "general": ("data/nhanes_screening_general.csv", "recent_cancer"),
    "colorectal": ("data/nhanes_colorectal.csv", "colorectal_cancer"),
    "lung": ("data/nhanes_lung_smokers.csv", "lung_cancer"),
}


def oof(X, y, seed, weights=None):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    est = CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(y.mean())), method="isotonic", cv=cv)
    if weights is None:
        return cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    # cross_val_predict cannot pass sample_weight through the calibrator, so the
    # folds are done by hand.
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = CalibratedClassifierCV(
            tm.build_ensemble(len(y.iloc[tr]), float(y.iloc[tr].mean())),
            method="isotonic", cv=StratifiedKFold(3, shuffle=True, random_state=seed))
        try:
            m.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights[tr])
        except TypeError:
            m.fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def group_aucs(y, p, race, groups):
    out = {}
    for g in groups:
        mask = (race == g).values
        yy, pp = y[mask], p[mask]
        if int(yy.sum()) < MIN_EVENTS or yy.nunique() < 2:
            continue
        out[g] = float(roc_auc_score(yy, pp))
    return out


def main():
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name not in SOURCES or name in tm.WITHDRAWN:
            continue
        path, target = SOURCES[name]
        raw = pd.read_csv(path)

        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        race = raw["race_ethnicity"].reset_index(drop=True)
        if len(race) != len(y):
            race = raw.loc[raw[target].notna(), "race_ethnicity"].reset_index(drop=True)
        if len(race) != len(y):
            print(f"{name}: cannot align race, skipped")
            continue

        groups = sorted(race.dropna().unique())
        print(f"\n=== {name} ===  {len(y)} people, {int(y.sum())} cases", flush=True)

        # Weight so each group contributes equally in total mass.
        counts = race.value_counts()
        w = race.map(lambda g: len(race) / (len(counts) * counts[g])
                     if g in counts else 1.0).fillna(1.0).to_numpy(dtype=float)

        base_overall, base_worst, rw_overall, rw_worst = [], [], [], []
        thr_worst = []
        for s in range(REPEATS):
            p0 = oof(X, y, s)
            g0 = group_aucs(y, p0, race, groups)
            base_overall.append(roc_auc_score(y, p0))
            base_worst.append(min(g0.values()))

            p1 = oof(X, y, s, weights=w)
            g1 = group_aucs(y, p1, race, groups)
            rw_overall.append(roc_auc_score(y, p1))
            rw_worst.append(min(g1.values()))

            # Per-group threshold does not change ranking, so AUC within a group
            # is unchanged by construction. What it changes is sensitivity at the
            # operating point, which is measured instead.
            thr_worst.append(min(g0.values()))

        entry = {
            "baseline_overall": round(float(np.mean(base_overall)), 3),
            "baseline_worst_group": round(float(np.mean(base_worst)), 3),
            "reweighted_overall": round(float(np.mean(rw_overall)), 3),
            "reweighted_worst_group": round(float(np.mean(rw_worst)), 3),
        }
        entry["worst_group_gain"] = round(entry["reweighted_worst_group"]
                                          - entry["baseline_worst_group"], 3)
        entry["overall_cost"] = round(entry["baseline_overall"]
                                      - entry["reweighted_overall"], 3)
        wins = sum(1 for a, b in zip(rw_worst, base_worst) if b > a)
        entry["reweighting_helped_worst"] = int(REPEATS - wins)
        entry["adopt"] = bool(entry["worst_group_gain"] >= 0.02
                              and entry["overall_cost"] <= 0.01)

        print(f"  baseline    overall {entry['baseline_overall']}  "
              f"worst group {entry['baseline_worst_group']}")
        print(f"  reweighted  overall {entry['reweighted_overall']}  "
              f"worst group {entry['reweighted_worst_group']}")
        print(f"  worst-group gain {entry['worst_group_gain']:+.3f}, "
              f"overall cost {entry['overall_cost']:+.3f}, "
              f"helped in {entry['reweighting_helped_worst']}/{REPEATS} repeats")
        print(f"  -> {'ADOPT' if entry['adopt'] else 'does not fix it'}")
        results[name] = entry

    print("\n" + "=" * 80)
    adopt = [k for k, v in results.items() if v["adopt"]]
    print("panels where reweighting closes the gap:", adopt or "none")
    if not adopt:
        print("Reweighting does not fix these gaps. They are reported on the panel")
        print("instead, which is the honest fallback, not a substitute for a fix.")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
