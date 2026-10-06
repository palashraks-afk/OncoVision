"""
The navigator's evidence cohort: NHANES 1999-2014 with the facts the rules read.

Why this exists
---------------
The navigator applies symptom-and-lab rules. To test them on real people with real
outcomes, each person needs the lab values the rules use (haemoglobin, MCV, platelets,
white count, and FERRITIN where it was measured, so iron deficiency is measured and not
assumed) plus whatever symptom-like facts NHANES asked about, and a death certificate
years later.

Eight continuous cycles, 1999-2000 through 2013-2014, are linked to the National Death
Index through 31 December 2019. 2015-2018 are left out for the reason fetch_nhanes_mortality
gives: their follow-up is too short for a five-year outcome to be fully observable.

Symptom proxies (NHANES asked none of these as cancer symptoms, so each is a proxy and is
named as one everywhere it is used):
    weight_loss   weight a year ago minus weight now is at least 5% of the earlier weight
                  AND the person says they were not trying to lose weight (WHQ070 = 2).
                  Cycles that did not ask WHQ070 count the loss only if it is at least 10%.
    cough         coughing on most days for 3 months in a row (RDQ031 = 1), a chronic cough,
                  asked 1999-2012 only
    shortness_of_breath   short of breath on stairs/hills (CDQ010 = 1), asked to age 40+ only
    ever_smoked   100 or more cigarettes in a lifetime (SMQ020 = 1)

Not available in NHANES at all: bleeding, bowel change, lumps, appetite, fatigue, jaundice,
dysphagia. Most of the guideline's symptoms are therefore invisible here, which means the
alert rates measured on this cohort are LOWER BOUNDS on what a person who reports every
symptom would see.

Run:  python fetch_nhanes_navigator.py     writes data/nhanes_navigator.csv.gz
"""

import os

import numpy as np
import pandas as pd

import fetch_nhanes_mortality as fm

OUT = os.path.join("data", "nhanes_navigator.csv.gz")
CYCLES = fm.CYCLES[:8]      # 1999-2000 ... 2013-2014

FERRITIN_NAMES = ["FERTIN", "L06TFR", "TFR", "L06"]


def pick(df, *names):
    if df is None:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def one_cycle(year, suffix, a, b):
    mort = fm.grab_mortality(a, b)
    demo = fm.take(fm.grab_xpt(year, suffix, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR"])
    if mort is None or demo is None:
        print(f"  {a}-{b}: missing demographics or mortality, skipped")
        return None
    df = demo.merge(mort[["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM"]], on="SEQN")
    df = df[(df["RIDAGEYR"] >= 20) & (df["ELIGSTAT"] == 1)]

    mcq = fm.grab_xpt(year, suffix, "MCQ")
    if mcq is not None and "MCQ220" in mcq.columns:
        df = df.merge(mcq[["SEQN", "MCQ220"]], on="SEQN", how="left")
        df = df[df["MCQ220"] != 1].drop(columns=["MCQ220"])

    cbc = fm.first_of(year, suffix, ["CBC", "L25", "LAB25"])
    if cbc is not None:
        c = pd.DataFrame({"SEQN": cbc["SEQN"], "hemoglobin": pick(cbc, "LBXHGB"),
                          "mcv": pick(cbc, "LBXMCVSI"), "platelets": pick(cbc, "LBXPLTSI"),
                          "wbc": pick(cbc, "LBXWBCSI")})
        df = df.merge(c.drop_duplicates("SEQN"), on="SEQN", how="left")
    else:
        for col in ("hemoglobin", "mcv", "platelets", "wbc"):
            df[col] = np.nan

    fer = None
    for name in FERRITIN_NAMES:
        fer = fm.grab_xpt(year, suffix, name)
        if fer is not None and "LBXFER" in fer.columns:
            break
        fer = None
    if fer is not None:
        f = pd.DataFrame({"SEQN": fer["SEQN"], "ferritin": pick(fer, "LBXFER")})
        df = df.merge(f.drop_duplicates("SEQN"), on="SEQN", how="left")
    else:
        df["ferritin"] = np.nan

    smq = fm.grab_xpt(year, suffix, "SMQ")
    df = df.merge(pd.DataFrame({"SEQN": smq["SEQN"], "SMQ020": pick(smq, "SMQ020")}).drop_duplicates("SEQN"),
                  on="SEQN", how="left") if smq is not None else df.assign(SMQ020=np.nan)

    whq = fm.grab_xpt(year, suffix, "WHQ")
    if whq is not None:
        w = pd.DataFrame({"SEQN": whq["SEQN"], "w_now": pick(whq, "WHD020"), "w_1y": pick(whq, "WHD050"),
                          "trying": pick(whq, "WHQ070")}).drop_duplicates("SEQN")
        df = df.merge(w, on="SEQN", how="left")
    else:
        df["w_now"] = df["w_1y"] = df["trying"] = np.nan

    rdq = fm.grab_xpt(year, suffix, "RDQ")
    df = df.merge(pd.DataFrame({"SEQN": rdq["SEQN"], "RDQ031": pick(rdq, "RDQ031")}).drop_duplicates("SEQN"),
                  on="SEQN", how="left") if rdq is not None else df.assign(RDQ031=np.nan)

    cdq = fm.grab_xpt(year, suffix, "CDQ")
    df = df.merge(pd.DataFrame({"SEQN": cdq["SEQN"], "CDQ010": pick(cdq, "CDQ010")}).drop_duplicates("SEQN"),
                  on="SEQN", how="left") if cdq is not None else df.assign(CDQ010=np.nan)

    loss = (df["w_1y"] - df["w_now"]) / df["w_1y"]
    valid = (df["w_1y"] > 50) & (df["w_now"] > 50) & (df["w_1y"] < 777) & (df["w_now"] < 777)
    pct = np.where(df["trying"].notna(), 0.05, 0.10)
    wl = valid & (loss >= pct) & ~(df["trying"] == 1)
    out = pd.DataFrame({
        "SEQN": df["SEQN"], "cycle": f"{a}-{b}", "age": df["RIDAGEYR"],
        "sex": np.where(df["RIAGENDR"] == 1, "male", "female"),
        "died": df["MORTSTAT"], "ucod": df["UCOD_LEADING"], "months": df["PERMTH_EXM"],
        "hemoglobin": df["hemoglobin"], "mcv": df["mcv"], "platelets": df["platelets"],
        "wbc": df["wbc"], "ferritin": df["ferritin"],
        "ever_smoked": np.where(df["SMQ020"] == 1, 1, np.where(df["SMQ020"] == 2, 0, np.nan)),
        "weight_loss": np.where(valid, wl.astype(float), np.nan),
        "cough": np.where(df["RDQ031"].isin([1, 2]), (df["RDQ031"] == 1).astype(float), np.nan),
        "shortness_of_breath": np.where(df["CDQ010"].isin([1, 2]), (df["CDQ010"] == 1).astype(float), np.nan),
    })
    print(f"  {a}-{b}: {len(out):>6,} adults  ferritin {out['ferritin'].notna().mean():>5.0%}  "
          f"weight-loss measured {out['weight_loss'].notna().mean():>4.0%}  "
          f"cough {out['cough'].notna().mean():>4.0%}  deaths {int((out['died'] == 1).sum())}", flush=True)
    return out


def main():
    print("Navigator cohort: NHANES 1999-2014 linked to the National Death Index\n")
    parts = [p for p in (one_cycle(*c) for c in CYCLES) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    cancer = int(((df["died"] == 1) & (df["ucod"] == fm.CANCER)).sum())
    print(f"\nwrote {OUT}: {len(df):,} adults, {int((df['died'] == 1).sum()):,} deaths, {cancer} from cancer")


if __name__ == "__main__":
    main()
