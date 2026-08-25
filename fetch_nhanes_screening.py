"""
Build a general panel that actually screens, from bloodwork.

What was wrong
--------------
The general panel predicted "have you ever been told you had cancer" from age,
sex, BMI, smoking, alcohol and exercise. Two things were broken about that.

First, it never read a single blood value, which is the entire premise of this
application. Second, the target was lifetime prevalence, so a person cured
thirty years ago counted as positive. Predicting that from age is mostly
predicting age, and the numbers said so: the model reached 0.781 while age and
sex alone reached 0.777. It was adding 0.004.

What this builds instead
------------------------
A screening target and a screening feature set.

TARGET. NHANES 2005 to 2014 records age at diagnosis (MCQ240 series) as well as
whether a person has had cancer. Subtracting it from age at exam gives years
since diagnosis, so the cohort can be cut properly:

    positive   diagnosed within RECENT_YEARS of the exam, so the disease was
               present or recent when the blood was actually drawn
    negative   never diagnosed
    EXCLUDED   diagnosed long ago. These are survivors. Their bloodwork
               reflects treatment and time, not detection, and leaving them in
               is what made the old target mostly a proxy for age.

FEATURES. The full routine panel this app already parses out of a PDF: complete
blood count, metabolic panel and liver chemistry, alongside the demographics and
lifestyle. Fourteen blood values instead of none.

That makes the question the app claims to answer the question the model is
actually trained on: does this person's current bloodwork look like someone with
cancer now, rather than does this person look old.

Run:  python fetch_nhanes_screening.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.XPT"

# Age at diagnosis (MCQ240) is only collected in these cycles.
CYCLES = [
    ("2005", "D", "2005-2006"),
    ("2007", "E", "2007-2008"),
    ("2009", "F", "2009-2010"),
    ("2011", "G", "2011-2012"),
    ("2013", "H", "2013-2014"),
]

# How close to the exam a diagnosis has to be to count as detectable then.
RECENT_YEARS = 4

RACE = {
    1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
    4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_cache: dict = {}


def grab(year, suffix, name):
    key = (year, suffix, name)
    if key in _cache:
        return _cache[key]
    url = BASE.format(year=year, name=name, suffix=suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=300, context=_ctx).read()
        df = (pd.read_sas(io.BytesIO(raw), format="xport")
              if (b"HEADER" in raw[:200] or b"SAS" in raw[:200]) else None)
    except Exception:
        df = None
    _cache[key] = df
    return df


def take(df, wanted):
    if df is None:
        return None
    have = [c for c in wanted if c in df.columns]
    return df[have] if "SEQN" in have else None


def build_cycle(year, suffix, label):
    demo = take(grab(year, suffix, "DEMO"),
                ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3"])
    cbc = take(grab(year, suffix, "CBC"),
               ["SEQN", "LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXPLTSI"])
    bio = take(grab(year, suffix, "BIOPRO"),
               ["SEQN", "LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR", "LBXSTP",
                "LBXSAL", "LBXSASSI", "LBXSATSI", "LBXSTB", "LBXSAPSI"])
    bmx = take(grab(year, suffix, "BMX"), ["SEQN", "BMXBMI"])
    smq = take(grab(year, suffix, "SMQ"), ["SEQN", "SMQ020", "SMQ040"])
    alq = take(grab(year, suffix, "ALQ"), ["SEQN", "ALQ101", "ALQ111", "ALQ130"])
    mcq = grab(year, suffix, "MCQ")
    if demo is None or cbc is None or bio is None or mcq is None:
        return None

    dx_cols = [c for c in mcq.columns if c.startswith("MCQ240")]
    if not dx_cols:
        return None

    # Earliest age at any diagnosis. 999 style codes are refusals, dropped.
    ages = mcq[dx_cols].where(mcq[dx_cols] < 200)
    earliest_dx = ages.min(axis=1)
    ever = mcq["MCQ220"].map({1: 1, 2: 0})
    mc = pd.DataFrame({"SEQN": mcq["SEQN"], "ever_cancer": ever, "age_dx": earliest_dx})

    df = demo.merge(cbc, on="SEQN").merge(bio, on="SEQN").merge(mc, on="SEQN")
    for part in (bmx, smq, alq):
        if part is not None:
            df = df.merge(part, on="SEQN", how="left")
    df = df[df["RIDAGEYR"] >= 20]

    years_since = df["RIDAGEYR"] - df["age_dx"]

    # The three-way split that makes this a screening target rather than a
    # lifetime one. Survivors are removed, not relabelled.
    recent = (df["ever_cancer"] == 1) & years_since.notna() & (years_since <= RECENT_YEARS)
    never = df["ever_cancer"] == 0
    keep = recent | never

    smoking = pd.Series(np.nan, index=df.index)
    if "SMQ020" in df.columns:
        smoking[df["SMQ020"] == 2] = 0
        if "SMQ040" in df.columns:
            smoking[(df["SMQ020"] == 1) & (df["SMQ040"] == 3)] = 1
            smoking[(df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))] = 2

    alcohol = (pd.to_numeric(df["ALQ130"], errors="coerce").clip(0, 5)
               if "ALQ130" in df.columns else pd.Series(np.nan, index=df.index))
    for gate in ("ALQ111", "ALQ101"):
        if gate in df.columns:
            alcohol[df[gate] == 2] = 0

    eth = df["RIDRETH3"] if "RIDRETH3" in df.columns else df.get("RIDRETH1")

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),
        "bmi": pd.to_numeric(df.get("BMXBMI"), errors="coerce"),
        # Complete blood count
        "wbc": pd.to_numeric(df["LBXWBCSI"], errors="coerce"),
        "rbc": pd.to_numeric(df["LBXRBCSI"], errors="coerce"),
        "hemoglobin": pd.to_numeric(df["LBXHGB"], errors="coerce"),
        "platelets": pd.to_numeric(df["LBXPLTSI"], errors="coerce"),
        # Metabolic panel
        "glucose": pd.to_numeric(df["LBXSGL"], errors="coerce"),
        "calcium": pd.to_numeric(df["LBXSCA"], errors="coerce"),
        "bun": pd.to_numeric(df["LBXSBU"], errors="coerce"),
        "creatinine": pd.to_numeric(df["LBXSCR"], errors="coerce"),
        "protein_total": pd.to_numeric(df["LBXSTP"], errors="coerce"),
        "albumin": pd.to_numeric(df["LBXSAL"], errors="coerce"),
        # Liver chemistry
        "ast": pd.to_numeric(df["LBXSASSI"], errors="coerce"),
        "alt": pd.to_numeric(df["LBXSATSI"], errors="coerce"),
        "bilirubin": pd.to_numeric(df["LBXSTB"], errors="coerce"),
        "alkaline_phosphatase": pd.to_numeric(df["LBXSAPSI"], errors="coerce"),
        # Lifestyle
        "smoking": smoking,
        "alcohol_intake": alcohol,
        "race_ethnicity": eth.map(RACE) if eth is not None else None,
        "cycle": label,
        "years_since_diagnosis": years_since,
        "recent_cancer": recent.astype(int),
    })
    out = out[keep.values]

    need = ["age", "gender", "wbc", "hemoglobin", "platelets", "glucose",
            "calcium", "albumin", "ast", "alt", "recent_cancer"]
    return out.dropna(subset=need)


def main():
    print(f"Screening cohort, cancer diagnosed within {RECENT_YEARS} years of the exam\n")
    frames = []
    for year, suffix, label in CYCLES:
        part = build_cycle(year, suffix, label)
        if part is None or part.empty:
            print(f"  {label}  unavailable")
            continue
        print(f"  {label}  n={len(part):<6} recent cancers={int(part.recent_cancer.sum()):<4} "
              f"({part.recent_cancer.mean():.2%})")
        frames.append(part)

    pooled = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "nhanes_screening_general.csv")
    pooled.to_csv(path, index=False)

    print(f"\n  POOLED n={len(pooled)}  recent cancers={int(pooled.recent_cancer.sum())} "
          f"({pooled.recent_cancer.mean():.2%})  ->  {path}")
    print(f"  blood values per record: 14")
    print(f"  survivors excluded rather than counted as positive, which is what "
          f"made the old target a proxy for age")


if __name__ == "__main__":
    main()
