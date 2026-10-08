"""
Cross-sectional NHANES cancer-status cohort, for the audit's controlled demonstration.

Much of the published machine-learning work on NHANES and cancer predicts a person's
self-reported cancer history ("has a doctor ever told you that you had cancer?", MCQ220)
from the other things measured at the same visit. This builds exactly that task, with the
same laboratory, examination and questionnaire columns as fetch_nhanes_wide.py, so the
benchmark gap (a model against age alone) and the effect of random versus cycle-held-out
validation can be measured on the task the literature actually uses.

The label (MCQ220) and the cancer-type items (MCQ230A, MCQ240) are kept as labels and are
never features. Adults 20+. No mortality linkage is needed.

Run:  WIDE_CYCLES=8 python fetch_nhanes_crosssectional.py     writes data/nhanes_xs_8c.csv.gz
"""

import os

os.environ.setdefault("WIDE_CYCLES", "8")

import numpy as np
import pandas as pd

import fetch_nhanes_mortality as fm
import fetch_nhanes_wide as fw

OUT = os.path.join("data", "nhanes_xs_8c.csv.gz")


def one_cycle(year, suffix, a, b):
    demo = fm.take(fm.grab_xpt(year, suffix, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR"])
    mcq = fm.grab_xpt(year, suffix, "MCQ")
    if demo is None or mcq is None or "MCQ220" not in mcq.columns:
        print(f"  {a}-{b}: missing demographics or MCQ, skipped")
        return None
    lab = mcq[["SEQN", "MCQ220"] + [c for c in ("MCQ230A",) if c in mcq.columns]].copy()
    df = demo.merge(lab, on="SEQN")
    df = df[(df["RIDAGEYR"] >= 20) & df["MCQ220"].isin([1, 2])]
    df["cancer"] = (df["MCQ220"] == 1).astype(int)
    found = []
    for comp, names in fw.COMPONENTS.items():
        frame = fm.first_of(year, suffix, names)
        if frame is None:
            continue
        cols = fw.numeric_columns(frame.drop(columns=["SEQN"], errors="ignore"))
        if cols.shape[1] == 0:
            continue
        cols.insert(0, "SEQN", frame["SEQN"].values)
        cols = cols[["SEQN"] + [c for c in cols.columns if c != "SEQN" and c not in df.columns]]
        if cols.shape[1] > 1:
            df = df.merge(cols.drop_duplicates("SEQN"), on="SEQN", how="left")
            found.append(comp)
    out = df.rename(columns={"RIDAGEYR": "age"})
    out["gender"] = (out["RIAGENDR"] == 1).astype(int)
    out = out.drop(columns=["RIAGENDR", "MCQ220"])
    out["cycle"] = f"{a}-{b}"
    print(f"  {a}-{b}  {len(out):>6,} adults, {int(out['cancer'].sum()):>4} with a cancer history, {out.shape[1]:>4} columns", flush=True)
    return out


def main():
    print("Cross-sectional cancer status, NHANES 1999-2014\n")
    frames = [f for f in (one_cycle(*c) for c in fm.CYCLES[:8]) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    meta = {"SEQN", "age", "gender", "cycle", "cancer", "MCQ230A"}
    keep = []
    for c in df.columns:
        if c in meta:
            keep.append(c)
            continue
        if c.startswith(("MCQ220", "MCQ230", "MCQ240", "MCQ250", "MCQ300")):
            continue
        cyc = df.groupby("cycle")[c].apply(lambda s: s.notna().mean() > 0.05).sum()
        if cyc >= 6 and df[c].notna().mean() >= 0.30:
            keep.append(c)
    df = df[keep]
    feats = [c for c in df.columns if c not in meta]
    ind = {f"{c}__missing": df[c].isna().astype(int) for c in feats if 0.05 <= df[c].isna().mean() <= 0.95}
    df = pd.concat([df, pd.DataFrame(ind)], axis=1)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"\nwrote {OUT}: {len(df):,} adults, {int(df['cancer'].sum()):,} with a cancer history, {len(feats)} columns plus {len(ind)} missing-indicators")


if __name__ == "__main__":
    main()
