"""
Does the prostate panel's triage pay, on a cohort from another country?

Why this one is different
-------------------------
Every cost model in this project so far has priced a triage decision on the same
data the panel was fitted to, and every lab-report panel lost to a rule built on
age. This one is external: 1,500 men at three Dutch hospitals, scored by a model
fitted on 212 men elsewhere, with the decision a urologist actually faces once
the MRI is back -- who goes to biopsy.

The comparator is the thing that matters, and it is NOT "biopsy everyone". It is
CURRENT PRACTICE: biopsy every man whose MRI reads PI-RADS 3 or above. That rule
is free, it is already standard, and the panel reads PI-RADS itself, so if the
panel cannot beat it the extra machinery buys nothing. This is the same test
that withdrew the bowel panel, where triage on age alone beat the panel.

Priced per 1,000 referred men:
    a biopsy                 its cost, swept
    a missed significant     fifteen life-years at $150,000 per QALY, the
    cancer (ISUP 2+)         convention used everywhere else in this project

Pre-registered bar, written before any figure was produced: a saving is claimed
only if the panel beats BOTH biopsy-everyone AND the PI-RADS 3 rule in every
sweep of both prices. Anything less is reported as no saving.

Carried caveats: 468 of these men were never biopsied, so their ISUP 0 rests on
a negative MRI rather than on tissue, which flatters any rule that leans on
PI-RADS -- including current practice and this panel. The biopsied-only arm is
reported beside it. Complications, anxiety, repeat biopsies and the MRI itself
are not priced; the MRI is common to every arm here because all 1,500 men had
one.

Run:  python experiments/prostate_biopsy_cost.py
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

OUT = "experiments/prostate_biopsy_cost_result.json"
COHORT = "data/prostate_picai.csv"
NAME = "prostate"
PER = 1000

BIOPSY_COST = 1000.0          # a transperineal biopsy, all-in, US facility
MISSED_COST = 2_250_000.0     # fifteen life-years at $150,000, as elsewhere here
SWEEP_BIOPSY = [300.0, 600.0, 1000.0, 2000.0, 3000.0]
SWEEP_MISSED = [500_000.0, 1_000_000.0, 2_250_000.0, 5_000_000.0]


def score_panel(df):
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    med = pd.Series(bundle["feature_medians"])
    X = pd.DataFrame({c: df.get(c, pd.Series(np.nan, index=df.index))
                      for c in bundle["feature_names"]}).astype(float)
    X = X.fillna(med).fillna(0.0)
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    return bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]


def cost_of(sent, y, biopsy_cost, missed_cost):
    """Cost per 1,000 men of a rule that biopsies `sent`."""
    n = len(y)
    biopsies = float(sent.sum()) / n * PER
    missed = float(((~sent) & (y == 1)).sum()) / n * PER
    caught = float((sent & (y == 1)).sum()) / n * PER
    return {"biopsies": round(biopsies, 1), "missed": round(missed, 2),
            "caught": round(caught, 2),
            "cost": round(biopsies * biopsy_cost + missed * missed_cost, 0)}


def best_threshold(p, y, biopsy_cost, missed_cost):
    best, best_cost = None, None
    for t in np.unique(np.round(p, 4)):
        c = cost_of(p >= t, y, biopsy_cost, missed_cost)
        if best_cost is None or c["cost"] < best_cost["cost"]:
            best, best_cost = float(t), c
    return best, best_cost


def arms(df, p, y, biopsy_cost, missed_cost):
    everyone = cost_of(np.ones(len(y), dtype=bool), y, biopsy_cost, missed_cost)
    pirads = df["pi_rads"].fillna(1).to_numpy()
    practice = cost_of(pirads >= 3, y, biopsy_cost, missed_cost)
    thr, panel = best_threshold(p, y, biopsy_cost, missed_cost)
    return {"biopsy_everyone": everyone, "pi_rads_3_or_above": practice,
            "panel": {**panel, "threshold": thr}}


def main():
    df = pd.read_csv(COHORT)
    p = score_panel(df)
    results = {"cohort": "PI-CAI, 3 Dutch centres", "n": int(len(df)),
               "base_inputs": {"biopsy_cost": BIOPSY_COST, "missed_cost": MISSED_COST,
                               "per": PER},
               "populations": {}}

    for pop, mask in (("all_referred", np.ones(len(df), dtype=bool)),
                      ("biopsied_only", df["biopsied"].to_numpy(dtype=bool))):
        sub, ps = df[mask].reset_index(drop=True), p[mask]
        y = sub["significant_cancer"].astype(int).to_numpy()
        base = arms(sub, ps, y, BIOPSY_COST, MISSED_COST)
        best_simple = min(base["biopsy_everyone"]["cost"], base["pi_rads_3_or_above"]["cost"])
        entry = {"n": int(len(y)), "events": int(y.sum()), "base": base,
                 "panel_saves_vs_everyone": round(base["biopsy_everyone"]["cost"]
                                                  - base["panel"]["cost"], 0),
                 "panel_saves_vs_practice": round(base["pi_rads_3_or_above"]["cost"]
                                                  - base["panel"]["cost"], 0),
                 "panel_saves_vs_better_simple": round(best_simple - base["panel"]["cost"], 0),
                 "sweeps": {}}

        wins_practice, wins_everyone = [], []
        for bc in SWEEP_BIOPSY:
            for mc in SWEEP_MISSED:
                a = arms(sub, ps, y, bc, mc)
                key = f"biopsy {bc:.0f} / missed {mc:.0f}"
                entry["sweeps"][key] = {
                    "vs_everyone": round(a["biopsy_everyone"]["cost"] - a["panel"]["cost"], 0),
                    "vs_practice": round(a["pi_rads_3_or_above"]["cost"] - a["panel"]["cost"], 0)}
                wins_everyone.append(a["biopsy_everyone"]["cost"] > a["panel"]["cost"])
                wins_practice.append(a["pi_rads_3_or_above"]["cost"] > a["panel"]["cost"])
        entry["beats_everyone_in_every_sweep"] = bool(all(wins_everyone))
        entry["beats_practice_in_every_sweep"] = bool(all(wins_practice))
        entry["saving_claimed"] = bool(all(wins_everyone) and all(wins_practice))
        results["populations"][pop] = entry

        print(f"\n{pop}: {len(y):,} men, {int(y.sum())} clinically significant cancers")
        for name, a in base.items():
            print(f"  {name:<22} biopsies {a['biopsies']:>6.1f}  missed {a['missed']:>5.2f}  "
                  f"cost ${a['cost']:>12,.0f}" + (f"   (cut {a['threshold']:.3f})"
                                                  if "threshold" in a else ""))
        print(f"  panel vs current practice: ${entry['panel_saves_vs_practice']:,.0f} per "
              f"{PER:,} men; vs biopsy-everyone ${entry['panel_saves_vs_everyone']:,.0f}")
        print(f"  beats practice in every sweep: {entry['beats_practice_in_every_sweep']}; "
              f"beats biopsy-everyone in every sweep: {entry['beats_everyone_in_every_sweep']}")
        print(f"  SAVING CLAIMED: {entry['saving_claimed']}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
