"""
A prospective cancer cohort: bloodwork first, outcome years later.

Why this file exists
--------------------
Every other cohort in this project is retrospective. The blood and the answer
were recorded at the same visit, or the cases were assembled after the fact and
matched to controls. That design cannot tell you whether a lab report predicts
anything, only whether it correlates with something already known. It is the
single largest weakness in the project, and it is not fixable by better models.

It is partly fixable by better data.

NCHS links every NHANES participant from 1999 to 2018 to the National Death
Index, and publishes the linkage. Vital status and underlying cause of death are
recorded through 31 December 2019. That gives a design this project has not had:

    time 0     a blood sample is drawn at the mobile examination centre
    ...        years pass, and nobody involved knows the outcome
    later      the death certificate is matched by NCHS, not by us

The exposure is measured before the outcome exists. That is a cohort study, and
it is what "this model predicts risk" is supposed to mean.

What the outcome actually is
----------------------------
Death from cancer, not a diagnosis of cancer. UCOD_LEADING == 2 is the NCHS
recode for malignant neoplasms (ICD-10 C00-C97). This matters and is stated
everywhere it is used:

  - It is a later and harsher endpoint than detection. People who developed
    cancer and survived it are counted as non-cases, because they are: they did
    not die of it. So this panel answers "does this bloodwork precede a death
    from cancer", not "does this person have cancer now".
  - It is confounded by everything that determines whether a cancer is
    survivable: stage at detection, treatment access, insurance, comorbidity.
    A model trained on it partly learns who gets treated.
  - Deaths from other causes inside the window are counted as non-cases. That is
    true as stated, but it is a competing risk and a cause-specific hazard model
    would handle it more carefully than a binary classifier does.

None of that makes it useless. It makes it a different question, honestly
labelled, answered on a design none of the other cohorts can offer.

Cohort construction
-------------------
    include    adults 20+, linkage eligible, with the lab work drawn
    exclude    anyone reporting a prior cancer diagnosis at baseline (MCQ220),
               so this is incident cancer death rather than recurrence
    positive   died with cancer as underlying cause within HORIZON months
    negative   followed at least HORIZON months without dying of cancer
    dropped    still alive but followed for less than HORIZON months, because
               their outcome at the horizon is genuinely unknown

The last rule is administrative censoring and it is why the most recent cycles
contribute mostly cases: someone examined in 2018 has at most two years of
follow-up, so they can enter as a death but not as a survivor.

Data use
--------
NCHS publishes these files for statistical reporting and analysis, which is what
this is. The public-use files are perturbed for disclosure control, and for a
small number of records the cause of death is synthetic. That adds noise to the
label; it does not bias it in a known direction.

Run:  python fetch_nhanes_mortality.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.XPT"
MORT = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
        "linked_mortality/NHANES_{a}_{b}_MORT_2019_PUBLIC.dat")

# Every continuous NHANES cycle that has been linked to the NDI.
CYCLES = [
    ("1999", "", "1999", "2000"),
    ("2001", "B", "2001", "2002"),
    ("2003", "C", "2003", "2004"),
    ("2005", "D", "2005", "2006"),
    ("2007", "E", "2007", "2008"),
    ("2009", "F", "2009", "2010"),
    ("2011", "G", "2011", "2012"),
    ("2013", "H", "2013", "2014"),
    # 2015-2016 and 2017-2018 are deliberately NOT here, even though NCHS links
    # them and they would add 39 more cancer deaths.
    #
    # Follow-up ends 31 December 2019. Someone examined in 2017 has at most two
    # years of it, so they can enter this cohort as a death but never as a
    # survivor: nobody from that cycle can be observed alive at 60 months. The
    # first build included them and the case rate came out at 23.6% for
    # 2015-2016 and 100% for 2017-2018, against roughly 1% everywhere else.
    #
    # Those are real people with real outcomes, so it is not that the records
    # are wrong. It is that pooling them contributes positives with no matched
    # negatives, and any drift in assay methods across cycles then looks like
    # cancer signal. Restricting to cycles that could complete the horizon costs
    # about a tenth of the cases and removes the confound.
]

# How long after the blood draw the outcome is read. Five years is a compromise:
# long enough that the death is not usually an already-symptomatic cancer at the
# time of the draw, short enough that most cycles contribute survivors too.
HORIZON_MONTHS = 60

# NCHS underlying-cause recode. 2 is malignant neoplasms.
CANCER = 2

# Fixed-width layout, from the R read-in program NCHS publishes alongside the
# files. Positions are zero-based half-open for pandas.
COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
COLNAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES",
            "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]

RACE = {
    1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
    4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=600, context=_ctx).read()
    except Exception:
        return None


def grab_xpt(year, suffix, name):
    """One NHANES component. The 1999-2000 cycle has no letter suffix."""
    url = (NHANES.format(year=year, name=name, suffix=suffix) if suffix
           else NHANES.format(year=year, name=name, suffix="").replace("_.XPT", ".XPT"))
    raw = _get(url)
    if raw is None:
        return None
    try:
        return pd.read_sas(io.BytesIO(raw), format="xport")
    except Exception:
        return None


def take(df, wanted):
    if df is None:
        return None
    have = [c for c in wanted if c in df.columns]
    return df[have] if "SEQN" in have else None


def first_of(year, suffix, names):
    """
    The first component that exists under any of these names.

    NHANES renamed its files between 1999 and 2003. The biochemistry profile is
    BIOPRO from 2005 on, LAB18 in 1999-2000 and L40 in 2001-2004; the blood
    count is CBC now and LAB25 then. Same measurements, three names. Written as
    an explicit `is not None` chain because `a or b` on DataFrames raises.
    """
    for name in names:
        df = grab_xpt(year, suffix, name)
        if df is not None:
            return df
    return None


def grab_mortality(a, b):
    raw = _get(MORT.format(a=a, b=b))
    if raw is None:
        return None
    m = pd.read_fwf(io.BytesIO(raw), colspecs=COLSPECS, names=COLNAMES)
    for c in ("ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM", "PERMTH_INT"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


def build_cycle(year, suffix, a, b):
    mort = grab_mortality(a, b)
    demo = take(grab_xpt(year, suffix, "DEMO"),
                ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3"])
    cbc = take(first_of(year, suffix, ["CBC", "LAB25", "L25"]),
               ["SEQN", "LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXPLTSI",
                "LBXHCT", "LBXMCVSI", "LBXMC", "LBXRDW", "LBXMPSI", "LBXNEPCT"])
    bio = take(first_of(year, suffix, ["BIOPRO", "L40", "LAB18"]),
               ["SEQN", "LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR", "LBXSTP",
                "LBXSAL", "LBXSASSI", "LBXSATSI", "LBXSTB", "LBXSAPSI", "LBXSGTSI"])
    bmx = take(grab_xpt(year, suffix, "BMX"), ["SEQN", "BMXBMI"])
    smq = take(grab_xpt(year, suffix, "SMQ"), ["SEQN", "SMQ020", "SMQ040"])
    alq = take(grab_xpt(year, suffix, "ALQ"), ["SEQN", "ALQ101", "ALQ111", "ALQ130"])
    mcq = take(grab_xpt(year, suffix, "MCQ"), ["SEQN", "MCQ220"])

    if mort is None or demo is None or bio is None:
        return None

    df = demo.merge(mort, on="SEQN").merge(bio, on="SEQN")
    for part in (cbc, bmx, smq, alq, mcq):
        if part is not None:
            df = df.merge(part, on="SEQN", how="left")

    df = df[(df["RIDAGEYR"] >= 20) & (df["ELIGSTAT"] == 1)]

    # Incident, not recurrent. Anyone already told they had cancer is out.
    if "MCQ220" in df.columns:
        df = df[df["MCQ220"] != 1]

    months = df["PERMTH_EXM"]
    died = df["MORTSTAT"] == 1
    cancer_death = died & (df["UCOD_LEADING"] == CANCER)

    positive = cancer_death & (months <= HORIZON_MONTHS)
    # Followed long enough for "no cancer death by the horizon" to be a fact.
    followed = months >= HORIZON_MONTHS
    keep = positive | followed

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

    def num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),
        "bmi": num("BMXBMI"),
        "wbc": num("LBXWBCSI"), "rbc": num("LBXRBCSI"),
        "hemoglobin": num("LBXHGB"), "platelets": num("LBXPLTSI"),
        "hematocrit": num("LBXHCT"), "mcv": num("LBXMCVSI"), "mch": num("LBXMC"),
        "rdw": num("LBXRDW"), "mpv": num("LBXMPSI"), "neutrophil_pct": num("LBXNEPCT"),
        "glucose": num("LBXSGL"), "calcium": num("LBXSCA"), "bun": num("LBXSBU"),
        "creatinine": num("LBXSCR"), "protein_total": num("LBXSTP"),
        "albumin": num("LBXSAL"), "ast": num("LBXSASSI"), "alt": num("LBXSATSI"),
        "bilirubin": num("LBXSTB"), "alkaline_phosphatase": num("LBXSAPSI"),
        "ggt": num("LBXSGTSI"),
        "smoking": smoking,
        "alcohol_intake": alcohol,
        "race_ethnicity": eth.map(RACE) if eth is not None else None,
        "cycle": f"{a}-{b}",
        "followup_months": months,
        "cancer_death": positive.astype(int),
    })
    out = out[keep.values]

    need = ["age", "gender", "albumin", "ast", "alt", "cancer_death"]
    return out.dropna(subset=need)


def main():
    print(f"Prospective cancer-mortality cohort, NHANES linked to the National "
          f"Death Index\nOutcome: death from malignant neoplasm within "
          f"{HORIZON_MONTHS} months of the blood draw\n")
    frames = []
    for year, suffix, a, b in CYCLES:
        part = build_cycle(year, suffix, a, b)
        if part is None or part.empty:
            print(f"  {a}-{b}  unavailable")
            continue
        print(f"  {a}-{b}  n={len(part):<6} cancer deaths={int(part.cancer_death.sum()):<4} "
              f"({part.cancer_death.mean():.2%})", flush=True)
        frames.append(part)

    if not frames:
        print("\nNo cycles built.")
        return

    pooled = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "nhanes_cancer_mortality.csv")
    pooled.to_csv(path, index=False)

    print(f"\n  POOLED n={len(pooled):,}  cancer deaths={int(pooled.cancer_death.sum()):,} "
          f"({pooled.cancer_death.mean():.2%})  ->  {path}")
    print(f"  median follow-up {pooled.followup_months.median():.0f} months")
    print(f"  the blood was drawn before the outcome existed, which is the "
          f"point of this cohort")


if __name__ == "__main__":
    main()
