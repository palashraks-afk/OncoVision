"""
Pool every NHANES cycle from 1999 to 2018, so the rarer sites have enough events.

Why this exists
---------------
The colorectal panel was built on five cycles, 2005 to 2014, because those are
the only ones carrying age at diagnosis. That gave 96 events, exactly on the
floor this project uses, and it left lung with 57, which was not enough to build
anything and was reported as such.

Age at diagnosis is not needed to count cases, though. Every cycle back to 1999
records whether a person was told they had cancer and which site, and every
cycle has a complete blood count. Pooling all ten roughly doubles the event
count:

    lung          57  ->  117
    colorectal   173  ->  345
    prostate     373  ->  738

That is the difference between "cannot be built" and "can be tested", which is
why this was worth doing rather than restating the limitation.

The cost, stated plainly
------------------------
Pooling buys events by giving up the recency window. The target here is a
LIFETIME diagnosis, so a person treated twenty years ago counts as positive.
That is the exact weakness that made the old general panel mostly an age model,
and it is worse for lung than for anything else: lung cancer five year survival
is around a quarter, so the lung cases who survive to answer a household survey
are a survivor-biased minority, and their bloodwork reflects treatment.

So this file produces a cohort for TESTING, not an automatic promotion to a
shipped panel. Whether any of these sites beats age, sex and smoking is decided
by measurement in experiments/pooled_sites.py, and a site that fails is reported
as failing.

File naming across cycles
-------------------------
NHANES renamed its files twice, which is why the early cycles silently failed
the first time this was attempted:

    complete blood count   1999 LAB25, 2001 L25_B, 2003 L25_C, 2005+ CBC_x
    chemistry              1999 LAB18, 2001 L40_B, 2003 L40_C, 2005+ BIOPRO_x

Run:  python fetch_nhanes_sites.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "nhanes_sites_pooled.csv")

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}.XPT"

# year, suffix, label, cbc file, chemistry file
CYCLES = [
    ("1999", "",   "1999-2000", "LAB25",  "LAB18"),
    ("2001", "_B", "2001-2002", "L25_B",  "L40_B"),
    ("2003", "_C", "2003-2004", "L25_C",  "L40_C"),
    ("2005", "_D", "2005-2006", "CBC_D",  "BIOPRO_D"),
    ("2007", "_E", "2007-2008", "CBC_E",  "BIOPRO_E"),
    ("2009", "_F", "2009-2010", "CBC_F",  "BIOPRO_F"),
    ("2011", "_G", "2011-2012", "CBC_G",  "BIOPRO_G"),
    ("2013", "_H", "2013-2014", "CBC_H",  "BIOPRO_H"),
    ("2015", "_I", "2015-2016", "CBC_I",  "BIOPRO_I"),
    ("2017", "_J", "2017-2018", "CBC_J",  "BIOPRO_J"),
]

# Verified against the MCQ codebook rather than guessed.
SITES = {"lung": [23], "colorectal": [16, 31], "prostate": [30], "breast": [14]}

LABS = {
    "LBXWBCSI": "wbc", "LBXRBCSI": "rbc", "LBXHGB": "hemoglobin",
    "LBXPLTSI": "platelets", "LBXHCT": "hematocrit", "LBXMCVSI": "mcv",
    "LBXRDW": "rdw", "LBXMPSI": "mpv",
    "LBXSGL": "glucose", "LBXSCA": "calcium", "LBXSBU": "bun", "LBXSCR": "creatinine",
    "LBXSTP": "protein_total", "LBXSAL": "albumin", "LBXSASSI": "ast",
    "LBXSATSI": "alt", "LBXSTB": "bilirubin", "LBXSAPSI": "alkaline_phosphatase",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def grab(year, fname):
    try:
        req = urllib.request.Request(BASE.format(year=year, fname=fname),
                                     headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, context=_ctx, timeout=300).read()
        return pd.read_sas(io.BytesIO(raw), format="xport")
    except Exception:
        return None


def _num(frame, name):
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    col = frame[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")


def fetch_nhanes_sites():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for year, suf, label, cbc_file, chem_file in CYCLES:
        demo = grab(year, "DEMO" + suf)
        mcq = grab(year, "MCQ" + suf)
        cbc = grab(year, cbc_file)
        chem = grab(year, chem_file)
        smq = grab(year, "SMQ" + suf)
        if demo is None or mcq is None or cbc is None:
            print(f"  {label}: required file missing, skipped")
            continue

        d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        for extra in (mcq, cbc, chem, smq):
            if extra is None:
                continue
            cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
            d = d.merge(extra[cols], on="SEQN", how="left", suffixes=("", "_dup"))
        d = d[_num(d, "RIDAGEYR") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = _num(d, "RIDAGEYR")
        out["gender"] = (_num(d, "RIAGENDR") == 1).astype(float)
        for src, key in LABS.items():
            out[key] = _num(d, src)

        ever = _num(d, "SMQ020")
        now = _num(d, "SMQ040")
        out["smoking"] = np.where(now.isin([1, 2]), 2.0,
                          np.where(ever == 1, 1.0, np.where(ever == 2, 0.0, np.nan)))

        told = _num(d, "MCQ220")
        never_any = told == 2

        for site, codes in SITES.items():
            hit = pd.Series(False, index=d.index)
            for L in "ABCD":
                c = f"MCQ230{L}"
                if c in d.columns:
                    hit |= _num(d, c).isin(codes).fillna(False)
            # Positive for this site, negative only if never told any cancer.
            # Someone with a different cancer is neither, and is dropped by the
            # per-site mask rather than labelled healthy.
            out[site] = np.where(hit, 1.0, np.where(never_any, 0.0, np.nan))

        out["cycle"] = label
        has_lab = out[list(LABS.values())].notna().any(axis=1)
        out = out[has_lab]
        frames.append(out)

        counts = " ".join(f"{s} {int((out[s] == 1).sum())}" for s in SITES)
        print(f"  {label}: {len(out):>5} adults with bloodwork   {counts}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(f"  {len(df)} adults pooled across {df.cycle.nunique()} cycles, 1999 to 2018")
    for site in SITES:
        n = int((df[site] == 1).sum())
        print(f"  {site:<12} {n:>4} lifetime cases   "
              f"{'clears' if n >= 96 else 'BELOW'} the 96 event floor")
    return df


if __name__ == "__main__":
    fetch_nhanes_sites()
