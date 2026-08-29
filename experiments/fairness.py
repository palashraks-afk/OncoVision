"""
Does each panel work as well for everyone? Measured, per race and ethnicity.

The claim this replaces
-----------------------
The documentation said, for a long time and in several places, "no race or
ethnicity in any cohort, so accuracy across those groups is unmeasured rather
than acceptable". The second half was the right instinct. The first half was
simply wrong, and it was wrong in the direction that let the project off the
hook.

NHANES records race and ethnicity for every participant, and four of the eight
shipped panels are built on NHANES: general, liver, bowel and lung. That is half
the product, and the fairness question was answerable the whole time.

Race is a STRATIFIER here, never a feature
------------------------------------------
It is deliberately excluded from every model's feature list. Using race as a
predictor bakes population averages in as though they were biology, which is
what medicine has spent recent years removing from estimates like eGFR. But not
recording it means fairness cannot be checked, which is worse. So it rides along
in the data and is used only to ask whether the model that ships treats people
equally.

What counts as a problem
------------------------
A panel is flagged when any group with enough events to estimate at all comes in
materially below the overall figure. Groups with fewer than the minimum events
are reported as unmeasurable rather than given a number that means nothing,
because a subgroup AUC on four cases is noise and printing it would be worse
than admitting the gap.

The four non-NHANES panels are listed at the end as still unmeasurable, because
Wisconsin, Soochow and the biopsy cohort genuinely do not record this.

Run:  python experiments/fairness.py
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

RANDOM_STATE = 42
MIN_EVENTS = 10          # below this a subgroup AUC is noise, so it is not printed
GAP_THAT_MATTERS = 0.05  # how far below overall counts as a real disparity
OUT = "experiments/fairness_result.json"

# panel -> the data file its cohort came from, since bundles do not carry it
SOURCES = {
    "general": "data/nhanes_screening_general.csv",
    "liver": "data/nhanes_liver_multicycle.csv",
    "colorectal": "data/nhanes_colorectal.csv",
    "lung": "data/nhanes_lung_smokers.csv",
}
TARGETS = {
    "general": "recent_cancer", "liver": "liver_disease",
    "colorectal": "colorectal_cancer", "lung": "lung_cancer",
}


def main():
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN or name not in SOURCES:
            continue

        raw = pd.read_csv(SOURCES[name])
        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        # prepare() drops rows, so realign race by the rows that survived.
        race = raw["race_ethnicity"].reset_index(drop=True)
        if len(race) != len(y):
            keep = raw[TARGETS[name]].notna()
            race = raw.loc[keep, "race_ethnicity"].reset_index(drop=True)
        if len(race) != len(y):
            print(f"{name}: could not align race to rows, skipped")
            continue

        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        p = cross_val_predict(
            CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                                   method="isotonic", cv=cv),
            X, y, cv=cv, method="predict_proba")[:, 1]
        overall = roc_auc_score(y, p)

        print(f"\n=== {name} ===  {len(y)} people, {int(y.sum())} cases, "
              f"overall AUC {overall:.3f}")

        entry = {"overall_auc": round(float(overall), 3), "groups": {},
                 "unmeasurable_groups": []}
        for g in sorted(race.dropna().unique()):
            mask = (race == g).values
            yy, pp = y[mask], p[mask]
            events = int(yy.sum())
            if events < MIN_EVENTS or yy.nunique() < 2:
                print(f"    {g:<24} n={int(mask.sum()):<6} events={events:<4} "
                      f"too few to measure")
                entry["unmeasurable_groups"].append(
                    {"group": g, "n": int(mask.sum()), "events": events})
                continue
            a = roc_auc_score(yy, pp)
            ci = bootstrap_ci(np.asarray(yy), pp, roc_auc_score)
            gap = a - overall
            flag = "  <-- BELOW" if gap <= -GAP_THAT_MATTERS else ""
            print(f"    {g:<24} n={int(mask.sum()):<6} events={events:<4} "
                  f"AUC {a:.3f} (CI {ci[0]}-{ci[1]})  {gap:+.3f}{flag}")
            entry["groups"][g] = {"n": int(mask.sum()), "events": events,
                                  "auc": round(float(a), 3), "auc_ci": ci,
                                  "gap_vs_overall": round(float(gap), 3),
                                  "materially_worse": bool(gap <= -GAP_THAT_MATTERS)}

        scored = {g: v["auc"] for g, v in entry["groups"].items()}
        if scored:
            worst = min(scored, key=scored.get)
            entry["worst_group"] = worst
            entry["worst_auc"] = scored[worst]
            entry["spread"] = round(max(scored.values()) - min(scored.values()), 3)
            print(f"    spread across measurable groups: {entry['spread']:.3f}"
                  f"   worst: {worst} at {scored[worst]:.3f}")
        results[name] = entry

    results["_not_measurable"] = {
        "panels": ["breast", "ovarian", "pancreatic", "prostate"],
        "reason": ("The Wisconsin, Soochow, pancreatic tissue-bank and transperineal "
                   "biopsy cohorts do not record race or ethnicity. This is a real gap "
                   "and no analysis of those cohorts can close it."),
    }

    print("\n" + "=" * 78)
    flagged = [(n, g) for n, e in results.items() if isinstance(e, dict)
               for g, v in e.get("groups", {}).items() if v.get("materially_worse")]
    if flagged:
        print("Groups more than 0.05 below their panel's overall AUC:")
        for n, g in flagged:
            print(f"  {n}: {g}")
    else:
        print("No measurable group falls more than 0.05 below its panel's overall AUC.")
    print("Still unmeasurable: breast, ovarian, pancreatic, prostate.")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
