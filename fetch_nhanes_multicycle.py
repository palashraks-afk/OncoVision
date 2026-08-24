"""
Pull NHANES across every cycle that carries the measurements this project uses.

Why
---
The single-cycle cohorts were the limiting factor. The general panel trained on
1,500 records from a risk-factor cohort and collapsed to 0.596 on a
representative sample. The liver panel had 4,887 US records. NHANES has run the
same instruments for over a decade, so both can be an order of magnitude larger.

Seven cycles carry liver chemistry, cancer history and demographics:

    2005-06 D, 2007-08 E, 2009-10 F, 2011-12 G, 2013-14 H, 2015-16 I, 2017-18 J

That is roughly 48,000 examined people. It also unlocks a test no single cycle
can give: temporal validation. Train on the older cycles and test on the newest,
which is a different sample, a different decade and different assay lots. A model
that survives that has shown something a random split never can.

Two cohorts are written:

    nhanes_liver_multicycle.csv     liver chemistry + any liver condition
    nhanes_general_multicycle.csv   risk factors + ever told had cancer

Both carry a `cycle` column so validation can hold an era out, and
`race_ethnicity`, so subgroup accuracy stays measurable.

Run:  python fetch_nhanes_multicycle.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.XPT"

CYCLES = [
    ("2005", "D", "2005-2006"),
    ("2007", "E", "2007-2008"),
    ("2009", "F", "2009-2010"),
    ("2011", "G", "2011-2012"),
    ("2013", "H", "2013-2014"),
    ("2015", "I", "2015-2016"),
    ("2017", "J", "2017-2018"),
]

RACE = {
    1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
    4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_cache: dict = {}


def grab(year, suffix, name):
    """Fetch one NHANES file, or None if that cycle does not carry it."""
    key = (year, suffix, name)
    if key in _cache:
        return _cache[key]
    url = BASE.format(year=year, name=name, suffix=suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=300, context=_ctx).read()
        if b"HEADER" not in raw[:200] and b"SAS" not in raw[:200]:
            _cache[key] = None
            return None
        df = pd.read_sas(io.BytesIO(raw), format="xport")
    except Exception:
        df = None
    _cache[key] = df
    return df


def cols(df, wanted):
    """Only the columns that exist, so a cycle missing one variable still loads."""
    if df is None:
        return None
    have = [c for c in wanted if c in df.columns]
    return df[have] if "SEQN" in have else None


def liver_cycle(year, suffix, label):
    demo = cols(grab(year, suffix, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3"])
    bio = cols(grab(year, suffix, "BIOPRO"),
               ["SEQN", "LBXSTB", "LBXSAPSI", "LBXSATSI", "LBXSASSI", "LBXSTP", "LBXSAL"])
    mcq = cols(grab(year, suffix, "MCQ"), ["SEQN", "MCQ160L"])
    if demo is None or bio is None or mcq is None:
        return None

    # Risk history, so the panel reads chemistry and the patient together rather
    # than chemistry alone. Hepatitis comes from serology here rather than
    # self-report, which is a better measurement of the same field the app asks
    # a patient to answer.
    diq = cols(grab(year, suffix, "DIQ"), ["SEQN", "DIQ010"])
    hbv = cols(grab(year, suffix, "HEPBD"), ["SEQN", "LBDHBG"])
    hcv = cols(grab(year, suffix, "HEPC"), ["SEQN", "LBXHCR"])

    df = demo.merge(bio, on="SEQN").merge(mcq, on="SEQN")
    for part in (diq, hbv, hcv):
        if part is not None:
            df = df.merge(part, on="SEQN", how="left")
    df = df[df["RIDAGEYR"] >= 20]

    # RIDRETH3 separates Non-Hispanic Asian and only exists from 2011. Older
    # cycles fall back to RIDRETH1, which does not.
    eth = df["RIDRETH3"] if "RIDRETH3" in df.columns else df.get("RIDRETH1")

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),
        "bilirubin": pd.to_numeric(df["LBXSTB"], errors="coerce"),
        "alkaline_phosphatase": pd.to_numeric(df["LBXSAPSI"], errors="coerce"),
        "alt": pd.to_numeric(df["LBXSATSI"], errors="coerce"),
        "ast": pd.to_numeric(df["LBXSASSI"], errors="coerce"),
        "protein_total": pd.to_numeric(df["LBXSTP"], errors="coerce"),
        "albumin": pd.to_numeric(df["LBXSAL"], errors="coerce"),
        # DIQ010: 1 yes, 2 no, 3 borderline. Borderline counts as not diabetic.
        "diabetes": (df["DIQ010"].map({1: 1, 2: 0, 3: 0})
                     if "DIQ010" in df.columns else np.nan),
        # LBDHBG surface antigen, LBXHCR RNA. 1 is positive in both.
        "hepatitis_b": (df["LBDHBG"].map({1: 1, 2: 0})
                        if "LBDHBG" in df.columns else np.nan),
        "hepatitis_c": (df["LBXHCR"].map({1: 1, 2: 0, 3: 0})
                        if "LBXHCR" in df.columns else np.nan),
        "race_ethnicity": eth.map(RACE) if eth is not None else None,
        "cycle": label,
        "liver_disease": df["MCQ160L"].map({1: 1, 2: 0}),
    })
    need = ["age", "gender", "bilirubin", "alkaline_phosphatase", "alt", "ast",
            "protein_total", "albumin", "liver_disease"]
    return out.dropna(subset=need)


def general_cycle(year, suffix, label):
    demo = cols(grab(year, suffix, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3"])
    bmx = cols(grab(year, suffix, "BMX"), ["SEQN", "BMXBMI"])
    smq = cols(grab(year, suffix, "SMQ"), ["SEQN", "SMQ020", "SMQ040"])
    alq = cols(grab(year, suffix, "ALQ"), ["SEQN", "ALQ101", "ALQ111", "ALQ130"])
    paq = cols(grab(year, suffix, "PAQ"), ["SEQN", "PAQ655", "PAD660", "PAQ670", "PAD675"])
    mcq = cols(grab(year, suffix, "MCQ"), ["SEQN", "MCQ220"])
    if demo is None or bmx is None or mcq is None:
        return None

    df = demo.merge(bmx, on="SEQN")
    for part in (smq, alq, paq):
        if part is not None:
            df = df.merge(part, on="SEQN", how="left")
    df = df.merge(mcq, on="SEQN")
    df = df[df["RIDAGEYR"] >= 20]

    smoking = pd.Series(np.nan, index=df.index)
    if "SMQ020" in df.columns:
        smoking[df["SMQ020"] == 2] = 0
        if "SMQ040" in df.columns:
            smoking[(df["SMQ020"] == 1) & (df["SMQ040"] == 3)] = 1
            smoking[(df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))] = 2

    # ALQ111 replaced ALQ101 as the "ever had a drink" gate from 2017.
    alcohol = (pd.to_numeric(df["ALQ130"], errors="coerce").clip(0, 5)
               if "ALQ130" in df.columns else pd.Series(np.nan, index=df.index))
    for gate in ("ALQ111", "ALQ101"):
        if gate in df.columns:
            alcohol[df[gate] == 2] = 0

    if all(c in df.columns for c in ("PAQ655", "PAD660", "PAQ670", "PAD675")):
        mins = (pd.to_numeric(df["PAQ655"], errors="coerce").fillna(0)
                * pd.to_numeric(df["PAD660"], errors="coerce").fillna(0)
                + pd.to_numeric(df["PAQ670"], errors="coerce").fillna(0)
                * pd.to_numeric(df["PAD675"], errors="coerce").fillna(0))
        activity = (mins / 60.0).clip(0, 10)
    else:
        activity = pd.Series(np.nan, index=df.index)

    eth = df["RIDRETH3"] if "RIDRETH3" in df.columns else df.get("RIDRETH1")

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),
        "bmi": pd.to_numeric(df["BMXBMI"], errors="coerce"),
        "smoking": smoking,
        "alcohol_intake": alcohol,
        "physical_activity": activity,
        "race_ethnicity": eth.map(RACE) if eth is not None else None,
        "cycle": label,
        "any_cancer": df["MCQ220"].map({1: 1, 2: 0}),
    })
    return out.dropna(subset=["age", "gender", "bmi", "smoking", "any_cancer"])


def build(name, fn, target, filename):
    print(f"\n{name}")
    frames = []
    for year, suffix, label in CYCLES:
        part = fn(year, suffix, label)
        if part is None or part.empty:
            print(f"  {label}  unavailable")
            continue
        print(f"  {label}  n={len(part):<6} cases={int(part[target].sum()):<5} "
              f"({part[target].mean():.1%})")
        frames.append(part)

    pooled = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    pooled.to_csv(path, index=False)
    print(f"  POOLED n={len(pooled)}  cases={int(pooled[target].sum())} "
          f"({pooled[target].mean():.1%})  ->  {path}")
    return pooled


def main():
    liver = build("Liver chemistry and liver condition", liver_cycle,
                  "liver_disease", "nhanes_liver_multicycle.csv")
    general = build("Risk factors and any cancer diagnosis", general_cycle,
                    "any_cancer", "nhanes_general_multicycle.csv")

    print("\nTemporal split available for both, train on early cycles and test on late:")
    for nm, df, tgt in [("liver", liver, "liver_disease"), ("general", general, "any_cancer")]:
        late = df["cycle"] >= "2015"
        print(f"  {nm:<8} early n={int((~late).sum()):<6} cases={int(df.loc[~late, tgt].sum()):<5} "
              f"| late n={int(late.sum()):<6} cases={int(df.loc[late, tgt].sum())}")


if __name__ == "__main__":
    main()
