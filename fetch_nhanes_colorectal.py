"""
A colorectal panel from a complete blood count, the ColonFlag way.

Why this is the one site of the three that worked
-------------------------------------------------
Lung, colorectal and prostate were all requested. Only this one survived
measurement, and the other two failed for different reasons that are worth
recording rather than quietly dropping. See experiments/site_window_sweep.py.

    lung        57 events at the widest possible window, against a floor of
                roughly 96. No public tabular cohort pairs lung cancer with a
                lab panel either; the public lung datasets are CT imaging.
    prostate    plenty of events, 373, and bloodwork adds +0.002 to age alone.
                At every window tried, 4 through 20 years and lifetime, the
                gain was between 0.000 and 0.002. It is an age model wearing a
                lab coat. Prostate needs PSA, and NHANES excludes men with a
                prostate cancer history from PSA testing by design.
    colorectal  96 events at an 8 year window, and bloodwork adds +0.030 over
                age and sex with an interval that excludes chance.

The premise has published support. ColonFlag, the best known colorectal
early-detection model, is exactly age plus sex plus a complete blood count, and
reports AUCs in the low 0.80s. This lands at 0.804 on the same idea, built from
open data.

Why an 8 year window
--------------------
The window is a real trade-off and it was swept rather than picked.

A lifetime target ("ever told you had colon cancer") counts someone cured
thirty years ago as positive. Their blood count reflects treatment and elapsed
time rather than an undetected tumour, and a model trained on it mostly learns
age. That is the precise bug this project already found and fixed on the
general panel.

Four years is the cleanest target but leaves 60 events. Eight years is the
tightest window that clears the event floor, at exactly 96. So 8 it is, and the
count is the smallest of any shipped panel, which is stated on the panel.

Anyone diagnosed longer ago than the window is EXCLUDED, not moved to the
control side. Labelling a survivor as healthy would be worse than including
them.

Run:  python fetch_nhanes_colorectal.py
"""

import os

import numpy as np
import pandas as pd

from fetch_nhanes_screening import grab, CYCLES

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "nhanes_colorectal.csv")

# Verified against the MCQ_D codebook. MCQ230A-D hold site codes in up to four
# slots; MCQ240A-Z hold age at diagnosis per cancer TYPE. They are different
# code systems and do not line up slot for slot.
COLON, RECTUM = 16, 31
AGE_AT_DX = ["MCQ240G", "MCQ240V"]      # G colon, V rectum

WINDOW_YEARS = 8

LABS = {
    "LBXWBCSI": "wbc", "LBXRBCSI": "rbc", "LBXHGB": "hemoglobin",
    "LBXPLTSI": "platelets", "LBXSGL": "glucose", "LBXSCA": "calcium",
    "LBXSBU": "bun", "LBXSCR": "creatinine", "LBXSTP": "protein_total",
    "LBXSAL": "albumin", "LBXSASSI": "ast", "LBXSATSI": "alt",
    "LBXSTB": "bilirubin", "LBXSAPSI": "alkaline_phosphatase",
}


def _num(frame, name):
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    col = frame[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")


def fetch_nhanes_colorectal():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for year, suf, label in CYCLES:
        demo, mcq = grab(year, suf, "DEMO"), grab(year, suf, "MCQ")
        cbc, bio = grab(year, suf, "CBC"), grab(year, suf, "BIOPRO")
        if any(x is None for x in (demo, mcq, cbc, bio)):
            print(f"  {label}: a required file is missing, skipped")
            continue

        d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        for extra in (mcq, cbc, bio):
            cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
            d = d.merge(extra[cols], on="SEQN", how="left", suffixes=("", "_dup"))
        d = d[_num(d, "RIDAGEYR") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = _num(d, "RIDAGEYR")
        out["gender"] = (_num(d, "RIAGENDR") == 1).astype(float)
        for src, key in LABS.items():
            out[key] = _num(d, src)

        # Site: colon or rectum, in any of the four slots.
        site_hit = pd.Series(False, index=d.index)
        for L in "ABCD":
            site_hit |= _num(d, f"MCQ230{L}").isin([COLON, RECTUM]).fillna(False)

        # Recency: the site's own age-at-diagnosis column.
        recent = pd.Series(False, index=d.index)
        for col in AGE_AT_DX:
            agedx = _num(d, col).where(_num(d, col) < 900)
            recent |= (out["age"] - agedx).between(0, WINDOW_YEARS).fillna(False)

        told = _num(d, "MCQ220")
        never_any_cancer = told == 2

        case = site_hit & recent
        out["colorectal_cancer"] = np.where(case, 1.0,
                                            np.where(never_any_cancer, 0.0, np.nan))
        out["cycle"] = label

        # Keep only decided rows that carry at least one lab value. Everyone
        # else, including survivors outside the window and people with a
        # different cancer, is dropped rather than labelled healthy.
        has_lab = out[list(LABS.values())].notna().any(axis=1)
        out = out[out["colorectal_cancer"].notna() & has_lab]
        frames.append(out)
        print(f"  {label}: {len(out)} adults, {int(out.colorectal_cancer.sum())} colorectal")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)

    n, pos = len(df), int(df["colorectal_cancer"].sum())
    print(f"\nwrote {OUT}")
    print(f"  {n} adults, {pos} diagnosed within {WINDOW_YEARS} years ({pos / n:.2%})")
    print(f"  survivors diagnosed longer ago than {WINDOW_YEARS} years are excluded, "
          f"not counted as controls")
    return df


if __name__ == "__main__":
    fetch_nhanes_colorectal()
