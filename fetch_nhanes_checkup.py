"""
The rest of the checkup: what else is on the paperwork from an ordinary visit.

Why
---
Every panel in this project reads a blood count and a metabolic panel, and the
bowel panel was withdrawn because those two added nothing to age and sex. But a
routine checkup produces more paper than that, and none of it was ever pulled:

    HbA1c                  ordered at almost every adult visit
    cholesterol, HDL,      the lipid panel, the other universal test
    triglycerides
    urine albumin and      the dipstick and its ratio, standard with diabetes
    creatinine, ACR        or hypertension
    blood pressure,        taken at every visit before anyone draws blood
    waist, BMI
    ferritin               iron studies, ordered when anaemia is suspected --
                           and iron-deficiency anaemia is the classic way a
                           right-sided bowel cancer announces itself

That last one is the reason this file exists. The withdrawn bowel panel could
see a low haemoglobin and a low MCV; it could not see whether iron stores were
actually empty, which is the finding that sends a patient to colonoscopy in real
practice. Ferritin is only in 2005-2010 and 2015-2018, so it is written out and
handled as its own question rather than silently imputed for half the cohort.

The cohort, the target and the exclusions are exactly those of
fetch_nhanes_colorectal.py -- colon or rectal cancer diagnosed within eight
years of the blood draw, survivors from longer ago dropped rather than called
healthy -- so the two files differ in columns and in nothing else. That is what
makes the comparison in experiments/colorectal_checkup.py a fair one.

Run:  python fetch_nhanes_checkup.py
"""

import os

import numpy as np
import pandas as pd

import fetch_nhanes_colorectal as fc
from fetch_nhanes_screening import CYCLES, grab

OUT = os.path.join("data", "nhanes_colorectal_checkup.csv")
OUT_GENERAL = os.path.join("data", "nhanes_general_checkup.csv")

# The general panel's target: any cancer diagnosed within four years of the
# exam, built the same way fetch_nhanes_screening.py builds it. Survivors from
# longer ago are excluded rather than counted as healthy.
RECENT_YEARS = 4

# file -> {variable: column name}. Every one of these is a line on a report a
# patient is handed, not a research assay.
CHECKUP = {
    "GHB":    {"LBXGH": "hba1c"},
    "TCHOL":  {"LBXTC": "cholesterol_total"},
    "HDL":    {"LBDHDD": "hdl"},
    "TRIGLY": {"LBXTR": "triglycerides"},
    "ALB_CR": {"URXUMA": "urine_albumin", "URXUCR": "urine_creatinine",
               "URDACT": "urine_albumin_creatinine_ratio"},
    "BMX":    {"BMXBMI": "bmi", "BMXWAIST": "waist"},
    "FERTIN": {"LBXFER": "ferritin"},
}


def blood_pressure(d):
    """The average of the readings taken, which is what a report prints."""
    sys_cols = [f"BPXSY{i}" for i in (1, 2, 3, 4)]
    dia_cols = [f"BPXDI{i}" for i in (1, 2, 3, 4)]
    sysv = pd.concat([fc._num(d, c) for c in sys_cols], axis=1)
    diav = pd.concat([fc._num(d, c) for c in dia_cols], axis=1)
    # A recorded diastolic of 0 means the sound continued to the cuff's end,
    # which is not a blood pressure of zero and must not be averaged as one.
    diav = diav.where(diav > 0)
    return sysv.mean(axis=1), diav.mean(axis=1)


def main():
    os.makedirs("data", exist_ok=True)
    frames = []
    for year, suf, label in CYCLES:
        demo, mcq = grab(year, suf, "DEMO"), grab(year, suf, "MCQ")
        cbc, bio = grab(year, suf, "CBC"), grab(year, suf, "BIOPRO")
        if any(x is None for x in (demo, mcq, cbc, bio)):
            print(f"  {label}: a required file is missing, skipped")
            continue

        d = demo[[c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDRETH1"]
                  if c in demo.columns]].copy()
        parts = [mcq, cbc, bio, grab(year, suf, "BPX")]
        for name in CHECKUP:
            parts.append(grab(year, suf, name))
        for extra in parts:
            if extra is None:
                continue
            cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
            d = d.merge(extra[cols], on="SEQN", how="left", suffixes=("", "_dup"))
        d = d[fc._num(d, "RIDAGEYR") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = fc._num(d, "RIDAGEYR")
        out["gender"] = (fc._num(d, "RIAGENDR") == 1).astype(float)
        eth = fc._num(d, "RIDRETH3")
        if eth.isna().all():
            eth = fc._num(d, "RIDRETH1")
        out["race_ethnicity"] = eth.map(fc.RACE)
        for src, key in fc.LABS.items():
            out[key] = fc._num(d, src)
        for _, mapping in CHECKUP.items():
            for src, key in mapping.items():
                out[key] = fc._num(d, src)
        out["systolic_bp"], out["diastolic_bp"] = blood_pressure(d)

        site_hit = pd.Series(False, index=d.index)
        for L in "ABCD":
            site_hit |= fc._num(d, f"MCQ230{L}").isin([fc.COLON, fc.RECTUM]).fillna(False)
        recent = pd.Series(False, index=d.index)
        for col in fc.AGE_AT_DX:
            agedx = fc._num(d, col).where(fc._num(d, col) < 900)
            recent |= (out["age"] - agedx).between(0, fc.WINDOW_YEARS).fillna(False)
        told = fc._num(d, "MCQ220")
        out["colorectal_cancer"] = np.where(site_hit & recent, 1.0,
                                            np.where(told == 2, 0.0, np.nan))

        # Any cancer, any site, within four years: the general panel's target.
        dx_cols = [c for c in d.columns if c.startswith("MCQ240")]
        if dx_cols:
            ages = d[dx_cols].apply(pd.to_numeric, errors="coerce")
            earliest = ages.where(ages < 200).min(axis=1)
            recent_any = (told == 1) & (out["age"] - earliest).between(0, RECENT_YEARS)
            out["recent_cancer"] = np.where(recent_any, 1.0,
                                            np.where(told == 2, 0.0, np.nan))
        else:
            out["recent_cancer"] = np.nan
        out["cycle"] = label

        has_lab = out[list(fc.LABS.values())].notna().any(axis=1)
        out = out[has_lab & (out["colorectal_cancer"].notna() | out["recent_cancer"].notna())]
        frames.append(out)
        extra_present = {k: int(out[k].notna().sum())
                         for k in ("hba1c", "cholesterol_total", "urine_albumin_creatinine_ratio",
                                   "systolic_bp", "ferritin")}
        print(f"  {label}: {len(out):>5} adults, {int(out.colorectal_cancer.sum()):>3} bowel "
              f"cancer   " + "  ".join(f"{k} {v}" for k, v in extra_present.items()))

    pooled = pd.concat(frames, ignore_index=True)
    df = pooled[pooled["colorectal_cancer"].notna()].drop(columns=["recent_cancer"])
    df.to_csv(OUT, index=False)
    gen = pooled[pooled["recent_cancer"].notna()].drop(columns=["colorectal_cancer"])
    gen.to_csv(OUT_GENERAL, index=False)
    print(f"\nwrote {OUT_GENERAL}")
    print(f"  {len(gen):,} adults, {int(gen.recent_cancer.sum())} with any cancer within "
          f"{RECENT_YEARS} years ({gen.recent_cancer.mean():.2%})")
    n, pos = len(df), int(df.colorectal_cancer.sum())
    print(f"\nwrote {OUT}")
    print(f"  {n:,} adults, {pos} diagnosed within {fc.WINDOW_YEARS} years ({pos / n:.2%})")
    for key in ("hba1c", "cholesterol_total", "hdl", "triglycerides",
                "urine_albumin_creatinine_ratio", "systolic_bp", "waist", "ferritin"):
        have = df[key].notna()
        print(f"  {key:<32} on {int(have.sum()):>6,} ({have.mean():.0%}), "
              f"cases with it {int(df.loc[have, 'colorectal_cancer'].sum())}")


if __name__ == "__main__":
    main()
