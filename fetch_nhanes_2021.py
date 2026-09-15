"""
NHANES August 2021 to August 2023: a cohort no model in this project has ever seen.

Why
---
The lung and liver panels train on 1999-2016 and withhold 2017-2018. That
withheld cycle is one draw, and split_stability.py showed both panels' published
held-out AUCs sit at the top of their split distribution. A second, later cycle
collected after the pandemic break, on a new laboratory contract, is the
strongest test public data can give either panel short of another country.

These rows are written to their own files and never to the training files, so
nothing in train_models.py can absorb them by accident.

One field differs. 2021-2023 dropped SMD030 (age started smoking regularly).
SMD630 (age first smoked a whole cigarette) is the nearest item and is used in
its place for pack-years. Pack-years added +0.001 to the lung panel when it was
built, so this cannot move the result much, and it is stated rather than hidden.

Run:  python fetch_nhanes_2021.py
"""

import os

import numpy as np
import pandas as pd

import fetch_nhanes_lung as fl
import fetch_nhanes_multicycle as fm

LABEL = "2021-2023"
LUNG_OUT = os.path.join("data", "nhanes_lung_smokers_2021.csv")
LIVER_OUT = os.path.join("data", "nhanes_liver_2021.csv")


def lung():
    g = lambda f: fl.grab("2021", f)
    demo, mcq, cbc, chem = g("DEMO_L"), g("MCQ_L"), g("CBC_L"), g("BIOPRO_L")
    cot, crp, smq = g("COT_L"), g("HSCRP_L"), g("SMQ_L")
    d = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]].copy()
    for extra in (mcq, cbc, chem, cot, crp, smq):
        d = d.merge(extra[[c for c in extra.columns if c != "WTPH2YR"]], on="SEQN", how="left")
    d = d[fl._num(d, "RIDAGEYR") >= 18]

    out = pd.DataFrame(index=d.index)
    out["age"] = fl._num(d, "RIDAGEYR")
    out["gender"] = (fl._num(d, "RIAGENDR") == 1).astype(float)
    out["race_ethnicity"] = fl._num(d, "RIDRETH3").map(fl.RACE)
    for src, key in fl.LABS.items():
        out[key] = fl._num(d, src)
    out["cotinine"] = fl._num(d, "LBXCOT")
    out["crp"] = fl._num(d, "LBXHSCRP")          # mg/L, same unit the panel trained on
    ever, now = fl._num(d, "SMQ020"), fl._num(d, "SMQ040")
    out["smoking"] = np.where(now.isin([1, 2]), 2.0,
                      np.where(ever == 1, 1.0, np.where(ever == 2, 0.0, np.nan)))
    started = fl._num(d, "SMD630").where(fl._num(d, "SMD630").between(5, 80))
    per_day = fl._num(d, "SMD650").where(fl._num(d, "SMD650").between(0, 100))
    out["smoking_packyears"] = (out["age"] - started).clip(lower=0) * (per_day / 20.0)
    out.loc[ever == 2, "smoking_packyears"] = 0.0

    told = fl._num(d, "MCQ220")
    hit = pd.Series(False, index=d.index)
    for L in "ABCD":
        hit |= fl._num(d, f"MCQ230{L}").eq(fl.LUNG_CODE).fillna(False)
    out["lung_cancer"] = np.where(hit, 1.0, np.where(told == 2, 0.0, np.nan))
    out["cycle"] = LABEL
    has_lab = out[list(fl.LABS.values()) + ["cotinine"]].notna().any(axis=1)
    out = out[out["lung_cancer"].notna() & has_lab]
    smoker = (out["smoking"] > 0) | (out["cotinine"] >= fl.COTININE_ACTIVE)
    sm = out[smoker]
    sm.to_csv(LUNG_OUT, index=False)
    print(f"lung   {len(sm)} adults with tobacco exposure, {int(sm.lung_cancer.sum())} lung cancer "
          f"-> {LUNG_OUT}")


def liver():
    df = fm.liver_cycle("2021", "L", LABEL)
    if df is None or df.empty:
        raise SystemExit("liver: 2021-2023 files did not load")
    df.to_csv(LIVER_OUT, index=False)
    print(f"liver  {len(df)} adults, {int(df.liver_disease.sum())} with a liver condition "
          f"-> {LIVER_OUT}")


if __name__ == "__main__":
    lung()
    liver()
