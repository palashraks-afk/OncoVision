"""
A lung panel that reads the lab report rather than the questionnaire.

Why the first attempt was not good enough
-----------------------------------------
Pooling all ten NHANES cycles got lung to 117 events, over the roughly 96 floor,
and a model on age, sex, self-reported smoking and routine bloodwork reached
0.857 against 0.840 for the questionnaire alone. A gain of 0.017 for reading a
complete blood count is not a product. Stripped down, that panel says "you are
old and you smoke", which the user already knew.

What was missing was that NHANES measures tobacco exposure in the blood and the
earlier build never used it.

    SERUM COTININE (LBXCOT) is the metabolite of nicotine and the standard
    objective measure of tobacco exposure. It is in every cycle from 1999 to
    2018. Self-reported smoking is a three-level answer that people under-report;
    cotinine is a number on a lab report, which is precisely the kind of input
    this application exists to read.

    C-REACTIVE PROTEIN captures the chronic inflammation that accompanies both
    COPD and tumour biology.

    SPIROMETRY, FEV1 and FVC, is measured in 2007 to 2012. Airflow obstruction is
    a lung cancer risk factor independent of how much someone smoked, which is
    why COPD patients are screened differently.

That turns the question from "does this person smoke" into "what does this
person's blood actually look like", and it is a fair fight only if the baseline
keeps the questionnaire. So the baseline here is age, sex and self-reported
smoking, and every lab value has to earn its place on top of that.

The honest limitation, unchanged
--------------------------------
Pooling ten cycles means giving up age at diagnosis, so the target is a LIFETIME
lung cancer diagnosis. Lung cancer five year survival is roughly a quarter, so
the cases who live to be interviewed in a household survey are a survivor-biased
minority and their bloodwork carries the aftermath of treatment. Whatever this
scores, that caveat travels with it, and experiments/lung_panel.py decides
whether it is worth shipping at all.

Run:  python fetch_nhanes_lung.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "nhanes_lung.csv")
OUT_SMOKERS = os.path.join(DATA_DIR, "nhanes_lung_smokers.csv")

# Serum cotinine above 3 ng/mL is the standard cut for active tobacco exposure.
COTININE_ACTIVE = 3.0
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{f}.XPT"

# NHANES renamed these files repeatedly. Every name below was probed rather than
# assumed, because guessing is what made the early cycles silently vanish from
# an earlier version of this work.
#   year, suffix, label, cbc, chemistry, cotinine, crp, spirometry
CYCLES = [
    ("1999", "",   "1999-2000", "LAB25",   "LAB18",     "LAB06",     "LAB11",    None),
    ("2001", "_B", "2001-2002", "L25_B",   "L40_B",     "L06_B",     "L11_B",    None),
    ("2003", "_C", "2003-2004", "L25_C",   "L40_C",     "L06COT_C",  "L11_C",    None),
    ("2005", "_D", "2005-2006", "CBC_D",   "BIOPRO_D",  "COT_D",     "CRP_D",    None),
    ("2007", "_E", "2007-2008", "CBC_E",   "BIOPRO_E",  "COTNAL_E",  "CRP_E",    "SPX_E"),
    ("2009", "_F", "2009-2010", "CBC_F",   "BIOPRO_F",  "COTNAL_F",  "CRP_F",    "SPX_F"),
    ("2011", "_G", "2011-2012", "CBC_G",   "BIOPRO_G",  "COTNAL_G",  None,       "SPX_G"),
    ("2013", "_H", "2013-2014", "CBC_H",   "BIOPRO_H",  "COT_H",     None,       None),
    ("2015", "_I", "2015-2016", "CBC_I",   "BIOPRO_I",  "COT_I",     "HSCRP_I",  None),
    ("2017", "_J", "2017-2018", "CBC_J",   "BIOPRO_J",  "COT_J",     "HSCRP_J",  None),
]

LUNG_CODE = 23      # verified against the MCQ codebook

LABS = {
    "LBXWBCSI": "wbc", "LBXRBCSI": "rbc", "LBXHGB": "hemoglobin",
    "LBXPLTSI": "platelets", "LBXHCT": "hematocrit", "LBXMCVSI": "mcv",
    "LBXRDW": "rdw", "LBXMPSI": "mpv",
    "LBXSGL": "glucose", "LBXSCA": "calcium", "LBXSBU": "bun", "LBXSCR": "creatinine",
    "LBXSTP": "protein_total", "LBXSAL": "albumin", "LBXSASSI": "ast",
    "LBXSATSI": "alt", "LBXSTB": "bilirubin", "LBXSAPSI": "alkaline_phosphatase",
}

# Race and ethnicity, carried as a STRATIFIER and never as a model feature.
#
# The distinction is deliberate. Using race as a predictor in a clinical risk
# model encodes population averages as if they were biology, which is the
# practice medicine has spent the last few years removing from things like eGFR.
# But not recording it at all means fairness cannot be measured, which is worse:
# "accuracy across those groups is unmeasured" is not the same as "acceptable".
#
# So it rides along in the data file, is excluded from every feature list, and
# exists so experiments/fairness.py can report AUC per group.
RACE = {
    1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
    4: "Non-Hispanic Black", 5: "Other or multiracial",
    6: "Non-Hispanic Asian", 7: "Other or multiracial",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def grab(year, fname):
    if fname is None:
        return None
    try:
        req = urllib.request.Request(BASE.format(year=year, f=fname),
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


def fetch_nhanes_lung():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for year, suf, label, cbc_f, chem_f, cot_f, crp_f, spx_f in CYCLES:
        demo, mcq = grab(year, "DEMO" + suf), grab(year, "MCQ" + suf)
        cbc, chem = grab(year, cbc_f), grab(year, chem_f)
        cot, crp, spx = grab(year, cot_f), grab(year, crp_f), grab(year, spx_f)
        smq = grab(year, "SMQ" + suf)
        if demo is None or mcq is None or cbc is None:
            print(f"  {label}: required file missing, skipped")
            continue

        d = demo[[c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDRETH1"] if c in demo.columns]].copy()
        for extra in (mcq, cbc, chem, cot, crp, spx, smq):
            if extra is None:
                continue
            cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
            d = d.merge(extra[cols], on="SEQN", how="left", suffixes=("", "_dup"))
        d = d[_num(d, "RIDAGEYR") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = _num(d, "RIDAGEYR")
        out["gender"] = (_num(d, "RIAGENDR") == 1).astype(float)
        eth = _num(d, "RIDRETH3")
        if eth.isna().all():
            eth = _num(d, "RIDRETH1")
        out["race_ethnicity"] = eth.map(RACE)
        for src, key in LABS.items():
            out[key] = _num(d, src)

        # The point of this file.
        out["cotinine"] = _num(d, "LBXCOT")
        crp_col = "LBXHSCRP" if "LBXHSCRP" in d.columns else "LBXCRP"
        out["crp"] = _num(d, crp_col)
        # hs-CRP is reported in mg/L, the older assay in mg/dL. Harmonise to mg/L.
        if crp_col == "LBXCRP":
            out["crp"] = out["crp"] * 10.0

        fev1, fvc = _num(d, "SPXNFEV1"), _num(d, "SPXNFVC")
        out["fev1"] = fev1
        out["fvc"] = fvc
        out["fev1_fvc_ratio"] = (fev1 / fvc).replace([np.inf, -np.inf], np.nan)

        # Self-reported tobacco, kept as the BASELINE the labs must beat.
        ever, now = _num(d, "SMQ020"), _num(d, "SMQ040")
        out["smoking"] = np.where(now.isin([1, 2]), 2.0,
                          np.where(ever == 1, 1.0, np.where(ever == 2, 0.0, np.nan)))
        started = _num(d, "SMD030").where(_num(d, "SMD030").between(5, 80))
        per_day = _num(d, "SMD650").where(_num(d, "SMD650").between(0, 100))
        years = (out["age"] - started).clip(lower=0)
        out["smoking_packyears"] = years * (per_day / 20.0)
        out.loc[ever == 2, "smoking_packyears"] = 0.0

        told = _num(d, "MCQ220")
        hit = pd.Series(False, index=d.index)
        for L in "ABCD":
            c = f"MCQ230{L}"
            if c in d.columns:
                hit |= _num(d, c).eq(LUNG_CODE).fillna(False)
        # Positive = lung cancer. Negative = never told any cancer. A person with
        # a different cancer is neither and is dropped rather than called healthy.
        out["lung_cancer"] = np.where(hit, 1.0, np.where(told == 2, 0.0, np.nan))

        out["cycle"] = label
        has_lab = out[list(LABS.values()) + ["cotinine"]].notna().any(axis=1)
        out = out[out["lung_cancer"].notna() & has_lab]
        frames.append(out)

        print(f"  {label}: {len(out):>5} adults  lung {int(out.lung_cancer.sum()):>3}  "
              f"cotinine {int(out.cotinine.notna().sum()):>5}  "
              f"crp {int(out.crp.notna().sum()):>5}  spiro {int(out.fev1.notna().sum()):>5}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)

    # The shipped cohort is the smoker-restricted one. Almost every lung cancer
    # case smoked and most controls did not, so on the full cohort the smoking
    # question separates the groups nearly on its own and the lab report only
    # looks like it is contributing: the questionnaire alone scores 0.836 there.
    # Restricting controls to people with tobacco exposure removes that free
    # ride, drops the baseline to 0.792, and is also the population that
    # actually gets offered lung screening.
    smoker = (df["smoking"] > 0) | (df["cotinine"] >= COTININE_ACTIVE)
    sm = df[smoker]
    sm.to_csv(OUT_SMOKERS, index=False)
    print(f"wrote {OUT_SMOKERS}")
    print(f"  {len(sm)} with tobacco exposure, {int(sm.lung_cancer.sum())} lung cancer "
          f"({sm.lung_cancer.mean():.3%})")

    n, pos = len(df), int(df["lung_cancer"].sum())
    print(f"\nwrote {OUT}")
    print(f"  {n} adults, {pos} lung cancer ({pos / n:.3%})")
    print(f"  cotinine on {df.cotinine.notna().sum()} ({df.cotinine.notna().mean():.0%}), "
          f"CRP on {df.crp.notna().sum()} ({df.crp.notna().mean():.0%}), "
          f"spirometry on {df.fev1.notna().sum()} ({df.fev1.notna().mean():.0%})")
    return df


if __name__ == "__main__":
    fetch_nhanes_lung()
