"""
Should the lung panel ask a screening question instead of a lifetime one?

The inconsistency
-----------------
The general and bowel panels were both rebuilt around a screening target: a
diagnosis within a fixed window of the blood draw, with long-ago survivors
excluded rather than counted as positive. The reason was that a lifetime target
is mostly a proxy for age. Predicting "were you ever told you had cancer" from a
blood sample largely means predicting how old the person is, and the general
panel's numbers said so plainly: 0.781 against 0.777 for age and sex alone.

The lung panel was never given the same treatment. It still uses MCQ230A-D,
which records the SITE of any cancer ever reported, so a man diagnosed at 45 and
cured is a positive at 70 with completely unremarkable bloodwork.

That was defensible only while it looked unavoidable. It is not: MCQ240N records
the age at lung cancer diagnosis, and it is present in every cycle from 2005 to
2016. All 84 lung cancers in those cycles have one.

The trade
---------
A window costs events. The lifetime target spans ten cycles back to 1999 and
yields 104 cases; a window can only use the six cycles that carry MCQ240N, and
then keeps only those diagnosed recently enough. On a panel this
event-starved that is a real price, and the question is whether the cleaner
question is worth it.

Arms
----
    lifetime            what ships: ever told lung cancer, 10 cycles
    window 4 years      diagnosed within 4 years of the draw
    window 8 years      diagnosed within 8 years, matching the bowel panel

Judged on gain over age and sex rather than on raw AUC, because raw AUC is
exactly the thing a lifetime target inflates. A window that scores lower but
adds more over demographics is the better panel, and saying so requires
measuring both.

Run:  python experiments/lung_window.py
"""

import io
import json
import os
import ssl
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/lung_window_result.json"
REPEATS = 5
LUNG_SITE = 23        # MCQ230A-D site code
AGE_AT_DX = "MCQ240N"  # age when first told lung cancer

CYCLES = [("2005", "D"), ("2007", "E"), ("2009", "F"),
          ("2011", "G"), ("2013", "H"), ("2015", "I")]

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def grab(year, suffix, name):
    url = (f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}"
           f"/DataFiles/{name}_{suffix}.XPT")
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=300, context=_ctx).read()
        return pd.read_sas(io.BytesIO(raw), format="xport")
    except Exception:
        return None


def build_windowed(window_years):
    """The lung cohort, with the target cut to a window instead of a lifetime."""
    frames = []
    for year, suffix in CYCLES:
        demo = grab(year, suffix, "DEMO")
        mcq = grab(year, suffix, "MCQ")
        cbc = grab(year, suffix, "CBC")
        bio = grab(year, suffix, "BIOPRO")
        smq = grab(year, suffix, "SMQ")
        if any(x is None for x in (demo, mcq, cbc, bio)):
            continue
        d = demo[["SEQN", "RIAGENDR", "RIDAGEYR"]]
        for part, cols in ((cbc, ["SEQN", "LBXWBCSI", "LBXRBCSI", "LBXHGB",
                                  "LBXPLTSI", "LBXHCT", "LBXMCVSI", "LBXRDW",
                                  "LBXMPSI"]),
                           (bio, ["SEQN", "LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR",
                                  "LBXSTP", "LBXSAL", "LBXSASSI", "LBXSATSI",
                                  "LBXSTB", "LBXSAPSI"]),
                           (smq, ["SEQN", "SMQ020", "SMQ040"])):
            if part is None:
                continue
            have = [c for c in cols if c in part.columns]
            d = d.merge(part[have], on="SEQN", how="left")

        keep_mcq = ["SEQN", "MCQ220"] + [c for c in mcq.columns
                                         if c.startswith("MCQ230") or c == AGE_AT_DX]
        d = d.merge(mcq[[c for c in keep_mcq if c in mcq.columns]], on="SEQN", how="left")
        d = d[d["RIDAGEYR"] >= 20]

        site_hit = pd.Series(False, index=d.index)
        for L in "ABCD":
            c = f"MCQ230{L}"
            if c in d.columns:
                site_hit |= pd.to_numeric(d[c], errors="coerce").eq(LUNG_SITE).fillna(False)

        age_dx = (pd.to_numeric(d[AGE_AT_DX], errors="coerce").where(lambda s: s < 200)
                  if AGE_AT_DX in d.columns else pd.Series(np.nan, index=d.index))
        years_since = pd.to_numeric(d["RIDAGEYR"], errors="coerce") - age_dx

        told = pd.to_numeric(d["MCQ220"], errors="coerce")
        recent = site_hit & years_since.notna() & (years_since <= window_years)
        never = told == 2
        keep = recent | never

        smoking = pd.Series(np.nan, index=d.index)
        if "SMQ020" in d.columns:
            smoking[d["SMQ020"] == 2] = 0
            if "SMQ040" in d.columns:
                smoking[(d["SMQ020"] == 1) & (d["SMQ040"] == 3)] = 1
                smoking[(d["SMQ020"] == 1) & (d["SMQ040"].isin([1, 2]))] = 2

        def num(c):
            return pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

        out = pd.DataFrame({
            "age": pd.to_numeric(d["RIDAGEYR"], errors="coerce"),
            "gender": (d["RIAGENDR"] == 1).astype(int),
            "smoking": smoking,
            "wbc": num("LBXWBCSI"), "rbc": num("LBXRBCSI"),
            "hemoglobin": num("LBXHGB"), "platelets": num("LBXPLTSI"),
            "hematocrit": num("LBXHCT"), "mcv": num("LBXMCVSI"),
            "rdw": num("LBXRDW"), "mpv": num("LBXMPSI"),
            "glucose": num("LBXSGL"), "calcium": num("LBXSCA"),
            "bun": num("LBXSBU"), "creatinine": num("LBXSCR"),
            "protein_total": num("LBXSTP"), "albumin": num("LBXSAL"),
            "ast": num("LBXSASSI"), "alt": num("LBXSATSI"),
            "bilirubin": num("LBXSTB"), "alkaline_phosphatase": num("LBXSAPSI"),
            "lung_cancer": recent.astype(int),
        })[keep.values]
        frames.append(out.dropna(subset=["age", "gender", "hemoglobin", "albumin"]))
    return pd.concat(frames, ignore_index=True) if frames else None


def cv_auc(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def measure(df, target, label):
    y = df[target].astype(int).reset_index(drop=True)
    feats = [c for c in df.columns if c != target]
    X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X = X.fillna(X.median())
    full = np.array([cv_auc(X, y, s) for s in range(REPEATS)])
    base = np.array([cv_auc(X[["age", "gender"]], y, s) for s in range(REPEATS)])
    gain = full - base
    print(f"  {label:<22} n={len(y):>6}  cases={int(y.sum()):>4}  "
          f"AUC {full.mean():.3f}  age/sex {base.mean():.3f}  "
          f"gain {gain.mean():+.3f}  wins {int((gain > 0).sum())}/{REPEATS}", flush=True)
    return {"n": int(len(y)), "cases": int(y.sum()),
            "auc": round(float(full.mean()), 3),
            "age_sex_auc": round(float(base.mean()), 3),
            "gain": round(float(gain.mean()), 3),
            "wins": int((gain > 0).sum()), "repeats": REPEATS}


def main():
    results = {}
    print("The lung panel's target: lifetime against a screening window\n")

    shipped = pd.read_csv("data/nhanes_lung.csv")
    cols = [c for c in shipped.columns
            if c not in ("race_ethnicity", "cycle", "smoking_packyears",
                         "cotinine", "crp", "neutrophil_pct", "mch", "ggt")]
    full = shipped[cols].dropna(subset=["lung_cancer"])
    results["lifetime (ships today)"] = measure(full, "lung_cancer", "lifetime, 10 cycles")
    # The reference the window has to beat: same lifetime target, same
    # smoker restriction the panel actually ships with.
    results["lifetime, smokers only"] = measure(
        full[full["smoking"].fillna(0) > 0], "lung_cancer", "lifetime, smokers only")

    for w in (4, 8):
        df = build_windowed(w)
        if df is None or df["lung_cancer"].sum() < 10:
            print(f"  window {w}y: too few cases to measure")
            continue
        results[f"window {w} years"] = measure(df, "lung_cancer", f"window {w} years")

    # The arm that would actually ship.
    #
    # The panel does not train on the full cohort. It trains on the
    # smoker-restricted one, because almost every case smoked and most controls
    # did not, so on the full cohort the smoking question separates the groups
    # nearly on its own. Comparing a windowed FULL cohort against the shipped
    # lifetime panel would compare two different populations and conclude
    # nothing, which is what the first version of this file did.
    print()
    for w in (4, 8):
        df = build_windowed(w)
        if df is None:
            continue
        sm = df[df["smoking"].fillna(0) > 0]
        if sm["lung_cancer"].sum() < 10:
            print(f"  window {w}y, smokers only: "
                  f"{int(sm['lung_cancer'].sum())} cases, too few to measure")
            results[f"window {w} years, smokers only"] = {
                "n": int(len(sm)), "cases": int(sm["lung_cancer"].sum()),
                "too_few": True}
            continue
        results[f"window {w} years, smokers only"] = measure(
            sm, "lung_cancer", f"window {w}y, smokers only")

    print("\n" + "=" * 82)
    ship = results["lifetime (ships today)"]
    # Compare like with like. The reference is the SHIPPED configuration —
    # lifetime target, smokers only — not the full-cohort lifetime arm, which
    # is a different population and gains more simply because the smoking
    # question separates the groups there. An earlier version of this file
    # compared against the full-cohort number and reached the right conclusion
    # for the wrong reason.
    ship = results.get("lifetime, smokers only") or results["lifetime (ships today)"]
    shippable = [k for k in results
                 if "window" in k and "smokers only" in k
                 and not results[k].get("too_few")]
    best = max(shippable, key=lambda k: results[k]["gain"], default=None)
    if best:
        d = results[best]["gain"] - ship["gain"]
        print(f"  best window arm: {best}, gain {results[best]['gain']:+.3f} "
              f"against {ship['gain']:+.3f} for the lifetime target ({d:+.3f})")
        adopt = d > 0.01 and results[best]["cases"] >= 40
        print(f"  -> {'ADOPT the window' if adopt else 'keep the lifetime target'}")
        if not adopt and results[best]["cases"] < 40:
            print(f"     ({results[best]['cases']} cases is too few to rebuild a panel on)")
    else:
        adopt = False

    with open(OUT, "w") as f:
        json.dump({"arms": results, "adopt_window": bool(adopt)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
