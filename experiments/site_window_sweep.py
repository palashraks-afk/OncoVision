"""
How recent can a site-specific target be and still have enough events?

The tension
-----------
A lifetime target ("ever told you had colon cancer") counts someone cured
thirty years ago as positive. Their bloodwork reflects treatment and elapsed
time, not detection, and a model trained on it mostly learns age. That bug was
found and fixed on the general panel and must not be reintroduced here.

A four-year target fixes that but leaves 60 colorectal and 34 lung events,
below the roughly 96 this project requires.

So sweep the window. For each site and each cut-off, report how many events
survive and whether bloodwork still beats age and sex. The right window is the
tightest one that clears the event floor, and if no window clears it, the panel
does not get built.

Run:  python experiments/site_window_sweep.py
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
from experiments.site_panels import build, SITES, NAMES, MIN_EVENTS

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
WINDOWS = [4, 8, 12, 20, None]      # None means lifetime
OUT = "experiments/site_window_sweep_result.json"


def flags(df, codes, age_cols, window):
    life = pd.Series(False, index=df.index)
    for L in "ABCD":
        life |= df[f"site_{L}"].isin(codes).fillna(False)
    if window is None:
        return life
    recent = pd.Series(False, index=df.index)
    for col in age_cols:
        if col in df.columns:
            recent |= (df["age"] - df[col]).between(0, window).fillna(False)
    return recent & life


def auc_of(X, y, feats):
    Xf = X[feats].apply(pd.to_numeric, errors="coerce")
    Xf = Xf.fillna(Xf.median())
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        Xf, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), bootstrap_ci(np.asarray(y), p, roc_auc_score)


def main():
    df = build()
    blood = list(NAMES.values())
    has_blood = df[blood].notna().any(axis=1)
    results = {}

    for site, (codes, age_cols) in SITES.items():
        print(f"=== {site} ===")
        life_all = flags(df, codes, age_cols, None)
        rows = []
        for w in WINDOWS:
            hit = flags(df, codes, age_cols, w)
            # Controls are people who never had ANY cancer. Cases excluded by
            # the window are dropped entirely rather than moved to the control
            # side, which would label a survivor as healthy.
            keep = (df["never_cancer"] | hit) & has_blood
            sub = df[keep]
            if site == "prostate":
                sub = sub[sub["gender"] == 1]
            y = hit[sub.index].astype(int)
            n = int(y.sum())
            wl = "lifetime" if w is None else f"{w}y"
            if n < MIN_EVENTS:
                print(f"  {wl:>9}: {n:>4} events  -> below the {MIN_EVENTS} floor")
                rows.append({"window": wl, "events": n, "verdict": "too few events"})
                continue

            base = ["age"] if site == "prostate" else ["age", "gender"]
            if site == "lung":
                base = base + ["smoking"]
            a0, c0 = auc_of(sub, y, base)
            a1, c1 = auc_of(sub, y, base + blood)
            gain = a1 - a0
            ok = c1[0] > 0.5 and gain > 0.02
            print(f"  {wl:>9}: {n:>4} events  baseline {a0:.3f}  +blood {a1:.3f} "
                  f"CI {c1}  gain {gain:+.3f}  {'OK' if ok else 'no gain'}")
            rows.append({"window": wl, "events": n,
                         "baseline_auc": round(a0, 3), "with_blood_auc": round(a1, 3),
                         "with_blood_ci": c1, "gain": round(gain, 3),
                         "verdict": "usable" if ok else "bloodwork adds nothing"})
        results[site] = {"lifetime_events": int((life_all & has_blood).sum()), "windows": rows}
        print()

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
