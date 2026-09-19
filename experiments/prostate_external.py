"""
The prostate panel on 1,500 men in another country. The first external test any
case-control panel in this project has had.

Why this matters more than another AUC
--------------------------------------
Section 4.2 of the paper is built on a single lesson: resampling one dataset
cannot detect a dataset-specific effect, and four of this project's panels have
never been measured any other way. The prostate panel ships on 212 men at one
hospital in China. PI-CAI is 1,500 men at three hospitals in the Netherlands,
different scanners, different referral pattern, different radiologists reading
the PI-RADS.

Pre-registered, written before any PI-CAI number was seen
---------------------------------------------------------
PRIMARY: the shipped panel, unchanged, must beat PSA alone on PI-CAI, with the
95% interval on the paired gain excluding zero. PSA is the only baseline that
counts, because every one of these men already had a PSA drawn -- the same bar
the panel had to clear internally, where it gained +0.156 over PSA by adding
PI-RADS.

SECONDARY, and the honest test of the card's wording:
  - the reduced tier (no MRI score) should NOT beat PSA. It ties PSA internally
    (0.668 against 0.670), and the card says so. If it wins here, the card is
    understating; if it loses badly, the card is overstating.
  - both targets: adenocarcinoma (ISUP 1+), which is what the panel was fitted
    on, and clinically significant cancer (ISUP 2+), which is what PI-CAI and
    most urologists care about.
  - per centre, because three hospitals are three chances to fail differently.

Scored exactly as the service scores: values clipped to the range the model has
evidence about, missing values filled with the training median (BMI is not in
this cohort at all, and never carried the panel).

Run:  python experiments/prostate_external.py
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
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/prostate_external_result.json"
COHORT = "data/prostate_picai.csv"
NAME = "prostate"
N_BOOT = 2000
MIN_EVENTS = 25          # below this a centre is reported, not scored


def paired_ci(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    gains = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(gains, 2.5)), 3),
            round(float(np.percentile(gains, 97.5)), 3)]


def main():
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    med = pd.Series(bundle["feature_medians"])
    ranges = bundle.get("feature_ranges") or {}
    df = pd.read_csv(COHORT)

    X = pd.DataFrame({c: df.get(c, pd.Series(np.nan, index=df.index))
                      for c in bundle["feature_names"]}).astype(float)
    missing_entirely = [c for c in X.columns if X[c].isna().all()]
    X = X.fillna(med).fillna(0.0)
    for col, (lo, hi) in ranges.items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    p = bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]

    reduced = bundle.get("reduced")
    pr = (reduced["model"].predict_proba(X[reduced["feature_names"]])[:, 1]
          if reduced else None)

    # PSA alone, ranked as a clinician reads it: higher is worse. No fitting,
    # so nothing about this cohort tunes the comparator.
    psa = df["psa"].fillna(df["psa"].median()).to_numpy()

    results = {"cohort": "PI-CAI, 3 Dutch centres", "n": int(len(df)),
               "features_missing_entirely": missing_entirely,
               "trained_on": "212 men, one centre, China",
               "targets": {}}
    print(f"PI-CAI: {len(df):,} men, panel trained on 212 at one centre elsewhere")
    if missing_entirely:
        print(f"  not in this cohort, filled with the training median: {missing_entirely}")

    for target, label in (("prostate_cancer", "adenocarcinoma, ISUP 1+"),
                          ("significant_cancer", "clinically significant, ISUP 2+")):
        y = df[target].astype(int).to_numpy()
        auc = float(roc_auc_score(y, p))
        psa_auc = float(roc_auc_score(y, psa))
        entry = {
            "label": label, "events": int(y.sum()), "prevalence": round(float(y.mean()), 3),
            "panel_auc": round(auc, 3),
            "panel_auc_ci": bootstrap_ci(y, p, roc_auc_score),
            "psa_auc": round(psa_auc, 3),
            "gain_over_psa": round(auc - psa_auc, 3),
            "gain_ci": paired_ci(y, p, psa),
            "shipped_auc": round(float(bundle["metrics"]["auc"]), 3),
        }
        entry["beats_psa"] = bool(entry["gain_ci"][0] > 0)
        if pr is not None:
            r_auc = float(roc_auc_score(y, pr))
            entry["reduced_auc"] = round(r_auc, 3)
            entry["reduced_gain_over_psa"] = round(r_auc - psa_auc, 3)
            entry["reduced_gain_ci"] = paired_ci(y, pr, psa, seed=1)
            entry["reduced_beats_psa"] = bool(entry["reduced_gain_ci"][0] > 0)

        centres = {}
        for c, part in df.groupby("center"):
            idx = part.index.to_numpy()
            yc = y[idx]
            if yc.sum() < MIN_EVENTS or yc.min() == yc.max():
                centres[c] = {"n": int(len(idx)), "events": int(yc.sum()), "auc": None}
                continue
            centres[c] = {"n": int(len(idx)), "events": int(yc.sum()),
                          "auc": round(float(roc_auc_score(yc, p[idx])), 3),
                          "psa_auc": round(float(roc_auc_score(yc, psa[idx])), 3)}
        entry["centres"] = centres

        # VERIFICATION BIAS, and the reason this experiment cannot stop at the
        # headline. 468 of these men were never biopsied: their MRI was clear,
        # so no tissue was ever taken and they are recorded as ISUP 0. PI-RADS
        # is what decided that, so for those men a high PI-RADS partly PRODUCED
        # the label it is being scored against. Restricting to the men with
        # histopathology breaks that loop -- every one of them has a real
        # answer, whatever their MRI said. If the panel only wins on the full
        # set, the win belongs to the referral pattern, not the model.
        bx = df["biopsied"].to_numpy(dtype=bool)
        ybx = y[bx]
        if ybx.sum() >= MIN_EVENTS and ybx.min() != ybx.max():
            b_auc = float(roc_auc_score(ybx, p[bx]))
            b_psa = float(roc_auc_score(ybx, psa[bx]))
            ci = paired_ci(ybx, p[bx], psa[bx], seed=2)
            entry["biopsied_only"] = {
                "n": int(bx.sum()), "events": int(ybx.sum()),
                "panel_auc": round(b_auc, 3), "psa_auc": round(b_psa, 3),
                "gain_over_psa": round(b_auc - b_psa, 3), "gain_ci": ci,
                "beats_psa": bool(ci[0] > 0)}
        results["targets"][target] = entry

        print(f"\n  {label}: {int(y.sum())} of {len(y):,} ({y.mean():.1%})")
        print(f"    panel        AUC {auc:.3f} {entry['panel_auc_ci']}   "
              f"(shipped, internal: {entry['shipped_auc']:.3f})")
        print(f"    PSA alone    AUC {psa_auc:.3f}")
        print(f"    gain         {entry['gain_over_psa']:+.3f}  95% CI "
              f"{entry['gain_ci'][0]:+.3f} to {entry['gain_ci'][1]:+.3f}  "
              f"-> {'BEATS PSA' if entry['beats_psa'] else 'not shown'}")
        if pr is not None:
            print(f"    no-MRI tier  AUC {entry['reduced_auc']:.3f}   vs PSA "
                  f"{entry['reduced_gain_over_psa']:+.3f} "
                  f"({entry['reduced_gain_ci'][0]:+.3f} to {entry['reduced_gain_ci'][1]:+.3f})"
                  f" -> {'beats PSA' if entry['reduced_beats_psa'] else 'ties or loses, as the card says'}")
        for c, v in centres.items():
            print(f"      {c:<6} n={v['n']:<5} events={v['events']:<4} "
                  + (f"AUC {v['auc']:.3f}  (PSA {v['psa_auc']:.3f})" if v["auc"] else "too few to score"))
        b = entry.get("biopsied_only")
        if b:
            print(f"    biopsied only ({b['n']:,} men with histopathology, {b['events']} cancers): "
                  f"panel {b['panel_auc']:.3f}, PSA {b['psa_auc']:.3f}, gain "
                  f"{b['gain_over_psa']:+.3f} ({b['gain_ci'][0]:+.3f} to {b['gain_ci'][1]:+.3f}) -> "
                  f"{'still beats PSA' if b['beats_psa'] else 'NOT SHOWN once verification bias is removed'}")

    primary = results["targets"]["prostate_cancer"]
    bx_ok = (primary.get("biopsied_only") or {}).get("beats_psa")
    results["primary_confirmed"] = bool(primary["beats_psa"] and bx_ok)
    results["survives_verification_bias"] = bool(bx_ok)
    print(f"\n  PRIMARY (beat PSA alone on adenocarcinoma, lower bound above zero, "
          f"and again among biopsied men only): "
          f"{'CONFIRMED' if results['primary_confirmed'] else 'NOT CONFIRMED'}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
