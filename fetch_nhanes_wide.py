"""
The whole picture: every lab, exam finding, questionnaire answer and habit NHANES
recorded for a person, not just the 22 routine blood values.

Why
---
Every test so far gave a model a few dozen values and asked it to find cancer.
The way a clinician thinks about it is different: a low haemoglobin means
something different in someone who has lost weight, smokes, has a family history
and is short of breath than in someone who is well. The overall picture is the
unit, and a pattern across many kinds of information is the thing to match. This
builds that picture from every component NHANES carries, so the question can be
asked properly: does breadth help, or was breadth never the missing ingredient?

What goes in
------------
Laboratory:    chemistry, blood count, HbA1c, lipids, CRP, cotinine, urine albumin
Examination:   body measures, blood pressure
Questionnaire: smoking, alcohol, diabetes, blood pressure and cholesterol history,
               medical conditions (heart, lung, liver, thyroid, arthritis...),
               physical activity, WEIGHT HISTORY (weight a year ago, so weight
               loss, a classic early warning that no lab value contains), general
               health, healthcare use, physical limitations, chest and breathing
               symptoms, bone, kidney and diet questions.

Columns are taken generically rather than hand-picked, so that nothing is chosen
because it looks like it should work. A column is kept if it was recorded in at
least four of the five cycles and is present for at least 30% of people.

What stays out, on purpose
--------------------------
    race and ethnicity, nativity, income, education: this project treats race as a
        stratifier and never a feature, and the others are close proxies for it.
        Only age and sex are taken from the demographics file.
    anything about a cancer diagnosis (MCQ220, MCQ230, MCQ240): that is the
        outcome's neighbour, and people who reported cancer at baseline are
        removed in any case.
    survey weights, sample design codes and lab comment codes: process, not health.

Special codes (7, 9, 77, 99, 777, 999 for refused and don't know) are set missing
on columns that are clearly categorical. Missingness itself is informative in a
questionnaire with skip patterns (an unanswered "age you started smoking" means
you never did), so a missing-indicator is added for columns that are partly
missing.

Cohort, and the lesson of last time
-----------------------------------
Adults 20+, linkage-eligible, no reported cancer at baseline. Cycles 1999-2008
only: a ten-year horizon needs ten years of follow-up, which follow-up ending in
2019 gives only those cycles. NOBODY is dropped for how they died. The previous
cohort dropped everyone who died of another cause inside the window, which made
every early death a cancer death; this one keeps them, with the cause, and the
analysis decides what to do with them in the open.

Run:  python fetch_nhanes_wide.py
"""

import os

import numpy as np
import pandas as pd

import fetch_nhanes_mortality as fm

# WIDE_CYCLES=8 extends to 2013-2014 (a later era for out-of-era tests) and writes a separate file.
N_CYCLES = int(os.environ.get("WIDE_CYCLES", "5"))
OUT = os.path.join("data", "nhanes_wide.csv.gz" if N_CYCLES == 5 else f"nhanes_wide_{N_CYCLES}c.csv.gz")
CYCLES = fm.CYCLES[:N_CYCLES]     # 5: 1999-2000 through 2007-2008; 8: through 2013-2014

# component -> candidate file names, tried in order. NHANES renamed its
# laboratory files between 1999 and 2005, so the same measurement has up to three
# names. fm.first_of tries each against the cycle's suffix.
COMPONENTS = {
    # laboratory
    "biochem": ["BIOPRO", "L40", "LAB18"],
    "cbc": ["CBC", "L25", "LAB25"],
    "hba1c": ["GHB", "L10", "LAB10"],
    "hdl": ["HDL", "L13", "LAB13"],
    "tchol": ["TCHOL"],
    "trigly": ["TRIGLY", "L13AM", "LAB13AM"],
    "crp": ["CRP", "L11", "LAB11"],
    "cotinine": ["COT", "L06COT", "LAB06", "COTNAL"],
    "urine": ["ALB_CR", "L16", "LAB16"],
    # examination
    "body": ["BMX"],
    "bp": ["BPX"],
    # questionnaire
    "smoking": ["SMQ"], "alcohol": ["ALQ"], "diabetes": ["DIQ"],
    "bp_history": ["BPQ"], "conditions": ["MCQ"], "activity": ["PAQ"],
    "weight_history": ["WHQ"], "general_health": ["HSQ"], "healthcare": ["HUQ"],
    "function": ["PFQ"], "cardio": ["CDQ"], "respiratory": ["RDQ"],
    "bone": ["OSQ"], "kidney": ["KIQ_U", "KIQ"], "diet": ["DBQ"],
    "insurance": ["HIQ"], "depression": ["DPQ"], "sleep": ["SLQ"],
}

# Never features. The cancer fields are the outcome's neighbours.
EXCLUDE_PREFIX = ("SEQN", "WT", "SDMV", "SDDSRVYR", "MCQ220", "MCQ230", "MCQ240", "MCQ250",
                  "MCQ300", "RIDRETH", "DMDBORN", "DMDCITZN", "INDFMPIR", "DMDEDUC",
                  "DMDMARTL", "RIDEXMON", "RIDEXAGM", "RIDSTATR")
EXCLUDE_SUFFIX = ("LC", "COM", "STATS", "STAT")
SPECIAL = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}
MIN_CYCLES, MIN_PRESENT = (4 if N_CYCLES == 5 else 6), 0.30


def usable(name):
    if name.startswith(EXCLUDE_PREFIX):
        return False
    if name.endswith(EXCLUDE_SUFFIX):
        return False
    return True


def numeric_columns(df):
    out = {}
    for c in df.columns:
        if c == "SEQN" or not usable(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        # A column with only a few distinct values is categorical, so the
        # refused and don't-know codes are missing rather than numbers.
        if s.dropna().nunique() <= 12:
            s = s.mask(s.isin(SPECIAL))
        out[c] = s
    return pd.DataFrame(out, index=df.index) if out else pd.DataFrame(index=df.index)


def build_cycle(year, suffix, a, b):
    mort = fm.grab_mortality(a, b)
    demo = fm.take(fm.grab_xpt(year, suffix, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR"])
    mcq = fm.grab_xpt(year, suffix, "MCQ")
    if mort is None or demo is None:
        return None
    df = demo.merge(mort[["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM"]],
                    on="SEQN")
    df = df[(df["RIDAGEYR"] >= 20) & (df["ELIGSTAT"] == 1)]
    if mcq is not None and "MCQ220" in mcq.columns:
        told = mcq[["SEQN", "MCQ220"]]
        df = df.merge(told, on="SEQN", how="left")
        df = df[df["MCQ220"] != 1].drop(columns=["MCQ220"])

    found = []
    for comp, names in COMPONENTS.items():
        frame = fm.first_of(year, suffix, names)
        if frame is None:
            continue
        cols = numeric_columns(frame.drop(columns=["SEQN"], errors="ignore"))
        if cols.shape[1] == 0:
            continue
        cols.insert(0, "SEQN", frame["SEQN"].values)
        # a column already contributed by another component is not repeated
        cols = cols[["SEQN"] + [c for c in cols.columns if c != "SEQN" and c not in df.columns]]
        if cols.shape[1] > 1:
            df = df.merge(cols.drop_duplicates("SEQN"), on="SEQN", how="left")
            found.append(comp)

    out = df.drop(columns=["ELIGSTAT"]).rename(columns={
        "RIDAGEYR": "age", "PERMTH_EXM": "followup_months", "MORTSTAT": "died",
        "UCOD_LEADING": "ucod_leading"})
    out["gender"] = (df["RIAGENDR"] == 1).astype(int).values
    out = out.drop(columns=["RIAGENDR"])
    out["cycle"] = f"{a}-{b}"
    print(f"  {a}-{b}  {len(out):>6,} adults, {out.shape[1]:>4} columns, "
          f"{len(found)}/{len(COMPONENTS)} components found", flush=True)
    return out


def main():
    print("The whole picture, NHANES 1999-2008, linked to the National Death Index\n")
    frames = []
    for year, suffix, a, b in CYCLES:
        part = build_cycle(year, suffix, a, b)
        if part is not None:
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    # Keep a column only if it was recorded in most cycles and for most people.
    meta = {"age", "gender", "cycle", "followup_months", "died", "ucod_leading", "SEQN"}
    keep = []
    for c in df.columns:
        if c in meta:
            keep.append(c)
            continue
        cyc_present = df.groupby("cycle")[c].apply(lambda s: s.notna().mean() > 0.05).sum()
        if cyc_present >= MIN_CYCLES and df[c].notna().mean() >= MIN_PRESENT:
            keep.append(c)
    df = df[keep]

    # Missingness as a signal, for columns that are partly missing.
    feats = [c for c in df.columns if c not in meta]
    ind = {}
    for c in feats:
        frac = df[c].isna().mean()
        if 0.05 <= frac <= 0.95:
            ind[f"{c}__missing"] = df[c].isna().astype(int)
    df = pd.concat([df, pd.DataFrame(ind)], axis=1)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    n_died = int((df["died"] == 1).sum())
    n_cancer = int(((df["died"] == 1) & (df["ucod_leading"] == fm.CANCER)).sum())
    print(f"\nwrote {OUT}")
    print(f"  {len(df):,} adults, {len(feats)} measured columns plus {len(ind)} missing-indicators")
    print(f"  {n_died:,} died by 2019, {n_cancer} of cancer")
    print("  nobody dropped for how they died")


if __name__ == "__main__":
    main()
