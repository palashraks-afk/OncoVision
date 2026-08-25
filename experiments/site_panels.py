"""
Can lung, colorectal or prostate panels be built from NHANES?

Why ask NHANES
--------------
These three were requested and none of them has a usable public tabular cohort
pairing the disease with a lab panel. ColonFlag, the best known colorectal
model, is exactly age plus sex plus a complete blood count, and it is
proprietary with no public data behind it. NHANES has the same ingredients: a
complete blood count, a metabolic and liver panel, tobacco history, and a
recorded cancer diagnosis with the site coded.

So instead of guessing whether this works, measure it.

What gets measured, per site
----------------------------
    events        lifetime cases, and cases diagnosed within 4 years, because
                  a lifetime target counts a man cured 30 years ago as positive
                  and mostly predicts age. That is the exact bug this project
                  already fixed once on the general panel.
    usable        how many of those cases actually have bloodwork drawn
    AUC           cross-validated, against the only baseline that matters
    baseline      age and sex alone, plus smoking for lung

The bar for shipping, applied consistently with every other panel here:
roughly 96 events, a confidence interval excluding chance, and a real margin
over age and sex. A site that fails is reported as failing rather than shipped
to make the panel count look better.

Run:  python experiments/site_panels.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci
from fetch_nhanes_screening import grab, CYCLES

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RECENT_YEARS = 4
MIN_EVENTS = 96          # the count this project uses as a floor
OUT = "experiments/site_panels_result.json"

# Verified against the MCQ_D codebook rather than guessed.
#
# Two separate code systems, and confusing them is what made the first run of
# this script report zero recent diagnoses for every site. MCQ230A through D are
# four SLOTS holding the site codes of up to four cancers. MCQ240A through Z are
# per CANCER TYPE, alphabetical by site name, so MCQ240A is age at bladder
# diagnosis regardless of which slot bladder appeared in. They do not line up,
# and pairing MCQ230A with MCQ240A is meaningless.
#
#   site -> (MCQ230 site codes, MCQ240 age-at-diagnosis columns)
SITES = {
    "lung":       ([23],     ["MCQ240N"]),
    "colorectal": ([16, 31], ["MCQ240G", "MCQ240V"]),   # colon and rectum
    "prostate":   ([30],     ["MCQ240U"]),
}

CBC = ["LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXPLTSI"]
BIO = ["LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR", "LBXSTP",
       "LBXSAL", "LBXSASSI", "LBXSATSI", "LBXSTB", "LBXSAPSI"]
NAMES = {
    "LBXWBCSI": "wbc", "LBXRBCSI": "rbc", "LBXHGB": "hemoglobin",
    "LBXPLTSI": "platelets", "LBXSGL": "glucose", "LBXSCA": "calcium",
    "LBXSBU": "bun", "LBXSCR": "creatinine", "LBXSTP": "protein_total",
    "LBXSAL": "albumin", "LBXSASSI": "ast", "LBXSATSI": "alt",
    "LBXSTB": "bilirubin", "LBXSAPSI": "alkaline_phosphatase",
}


def build():
    frames = []
    for year, suf, label in CYCLES:
        demo, mcq = grab(year, suf, "DEMO"), grab(year, suf, "MCQ")
        cbc, bio, smq = grab(year, suf, "CBC"), grab(year, suf, "BIOPRO"), grab(year, suf, "SMQ")
        if any(x is None for x in (demo, mcq, cbc, bio, smq)):
            continue
        d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        for extra in (mcq, cbc, bio, smq):
            cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
            d = d.merge(extra[cols], on="SEQN", how="left", suffixes=("", "_dup"))
        d = d[pd.to_numeric(d["RIDAGEYR"], errors="coerce") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = pd.to_numeric(d["RIDAGEYR"], errors="coerce")
        out["gender"] = (pd.to_numeric(d["RIAGENDR"], errors="coerce") == 1).astype(float)
        for src, key in NAMES.items():
            out[key] = pd.to_numeric(d[src], errors="coerce") if src in d.columns else np.nan

        ever = pd.to_numeric(d.get("SMQ020"), errors="coerce")
        now = pd.to_numeric(d.get("SMQ040"), errors="coerce")
        out["smoking"] = np.where(now.isin([1, 2]), 2.0,
                          np.where(ever == 1, 1.0, np.where(ever == 2, 0.0, np.nan)))

        told = pd.to_numeric(d.get("MCQ220"), errors="coerce")
        out["never_cancer"] = (told == 2)

        for L in "ABCD":
            sc = f"MCQ230{L}"
            out[f"site_{L}"] = pd.to_numeric(d[sc], errors="coerce") if sc in d.columns else np.nan
        # Age at diagnosis, keyed by cancer type rather than by slot.
        for col in sorted({c for _, cols in SITES.values() for c in cols}):
            v = pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)
            out[col] = pd.Series(v, index=d.index).where(pd.Series(v, index=d.index) < 900)
        out["cycle"] = label
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def targets(df, codes, age_cols):
    """Lifetime and recent-diagnosis flags for one site."""
    life = pd.Series(False, index=df.index)
    for L in "ABCD":
        life |= df[f"site_{L}"].isin(codes).fillna(False)

    # Recent means the site's own age-at-diagnosis column puts the diagnosis
    # within RECENT_YEARS of the exam.
    recent = pd.Series(False, index=df.index)
    for col in age_cols:
        if col not in df.columns:
            continue
        since = df["age"] - df[col]
        recent |= since.between(0, RECENT_YEARS).fillna(False)
    return life, recent & life


def score(X, y, feats, label):
    X = X[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    folds = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=folds),
        X, y, cv=folds, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    ci = bootstrap_ci(np.asarray(y), p, roc_auc_score)
    return {"arm": label, "n_features": len(feats), "auc": round(float(auc), 3), "auc_ci": ci}


def main():
    print("building NHANES cohort ...", flush=True)
    df = build()
    print(f"  {len(df)} adults across {df.cycle.nunique()} cycles\n")

    blood = [v for v in NAMES.values()]
    has_blood = df[blood].notna().any(axis=1)

    results = {}
    for site, (codes, age_cols) in SITES.items():
        life, recent = targets(df, codes, age_cols)
        pool = df["never_cancer"] | life
        usable = pool & has_blood

        n_life = int((life & has_blood).sum())
        n_recent = int((recent & has_blood).sum())
        print(f"=== {site} ===")
        print(f"  lifetime cases with bloodwork : {n_life}")
        print(f"  diagnosed within {RECENT_YEARS} years   : {n_recent}")

        entry = {"lifetime_cases_with_bloodwork": n_life,
                 "recent_cases_with_bloodwork": n_recent,
                 "min_events_required": MIN_EVENTS}

        # Prostate is male only; anything else keeps both sexes.
        sub = df[usable].copy()
        if site == "prostate":
            sub = sub[sub["gender"] == 1]
        y = life[sub.index].astype(int)

        if int(y.sum()) < MIN_EVENTS:
            entry["verdict"] = "NOT ENOUGH EVENTS"
            print(f"  VERDICT: below the {MIN_EVENTS} event floor, not built\n")
            results[site] = entry
            continue

        base = ["age"] if site == "prostate" else ["age", "gender"]
        if site == "lung":
            base = base + ["smoking"]
        arms = [
            score(sub, y, base, "baseline: age, sex" + (", smoking" if site == "lung" else "")),
            score(sub, y, base + blood, "baseline + bloodwork"),
            score(sub, y, blood, "bloodwork only"),
        ]
        for a in arms:
            print(f"  {a['arm']:<44} AUC {a['auc']}  CI {a['auc_ci']}")

        gain = arms[1]["auc"] - arms[0]["auc"]
        beats = arms[1]["auc_ci"][0] > 0.5 and gain > 0.02
        entry["arms"] = arms
        entry["gain_over_baseline"] = round(gain, 3)
        entry["verdict"] = "SHIP" if beats else "BLOODWORK ADDS NOTHING"
        print(f"  gain over baseline: {gain:+.3f}")
        print(f"  VERDICT: {entry['verdict']}\n")
        results[site] = entry

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
