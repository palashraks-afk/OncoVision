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
    # 2015-2016 still carries the MCQ240 age-at-diagnosis series, so it can be
    # cut into a screening window the same way the earlier cycles are. It was
    # simply never added. 2017-2018 is NOT here and cannot be: that cycle
    # dropped MCQ240 entirely, so there is no way to tell a recent diagnosis
    # from a thirty-year-old one, which is the distinction this cohort is built
    # on. Verified by fetching both files rather than assumed.
    ("2015", "I", "2015-2016"),
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
    # Waist circumference alongside BMI. BMI cannot tell a heavy-set person from
    # a centrally obese one, and it is central adiposity that the obesity-cancer
    # literature ties to risk. It costs nothing: NHANES measures it on the same
    # visit, and a clinic can measure it with a tape.
    #
    # Measured, it is worth +0.003 to the general panel's gain over age and sex,
    # consistently and in the right direction, but under the 0.005 bar set before
    # the experiment ran. So it is NOT in the panel. It is still collected here,
    # because the column is free and a later cohort or a different target may
    # make better use of it than this one does.
    # See experiments/general_body_activity.py.
    bmx = take(grab(year, suffix, "BMX"), ["SEQN", "BMXBMI", "BMXWAIST"])
    smq = take(grab(year, suffix, "SMQ"), ["SEQN", "SMQ020", "SMQ040"])
    alq = take(grab(year, suffix, "ALQ"), ["SEQN", "ALQ101", "ALQ111", "ALQ130"])
    # Serum cotinine and CRP. Neither was ever offered to this panel, which is
    # why "bloodwork does not help the general panel" was only ever tested
    # against a blood count and a metabolic panel. File names move per cycle.
    # grab() appends the cycle suffix itself, so these are base names.
    # 2015-2016 renamed both again: cotinine is COT_I and CRP became the high
    # sensitivity assay, HSCRP_I. Without these two entries the new cycle would
    # have joined with both columns silently empty.
    cot_base = {"D": "COT", "E": "COTNAL", "F": "COTNAL",
                "G": "COTNAL", "H": "COT", "I": "COT"}.get(suffix)
    crp_base = {"D": "CRP", "E": "CRP", "F": "CRP", "I": "HSCRP"}.get(suffix)
    cot = take(grab(year, suffix, cot_base) if cot_base else None, ["SEQN", "LBXCOT"])
    crp = take(grab(year, suffix, crp_base) if crp_base else None,
               ["SEQN", "LBXCRP", "LBXHSCRP"])
    # Physical activity. The application asks "hours of exercise per week" and
    # then no panel read the answer, which is a question asked for nothing.
    # From 2007 NHANES uses the Global Physical Activity Questionnaire, where
    # recreational activity is days per week times minutes per day, asked
    # separately for vigorous and moderate. That maps onto the app's question
    # directly. The 2005-2006 cycle used a different instrument that does not
    # convert cleanly, so it is left missing rather than forced.
    paq = take(grab(year, suffix, "PAQ"),
               ["SEQN", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675"])
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
    for part in (bmx, smq, alq, cot, crp, paq):
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

    # Recreational exercise, hours per week, on the same 0 to 10 scale the app
    # asks for. Work activity is deliberately excluded: a warehouse shift and a
    # run are not the same exposure, and the app's question says "exercise".
    # A "no" to the gate question means zero, not missing.
    def _bout(gate, days, minutes):
        if not all(c in df.columns for c in (gate, days, minutes)):
            return pd.Series(np.nan, index=df.index)
        d = pd.to_numeric(df[days], errors="coerce").where(lambda s: s <= 7)
        m = pd.to_numeric(df[minutes], errors="coerce").where(lambda s: s <= 720)
        hours = d * m / 60.0
        hours[df[gate] == 2] = 0.0
        return hours

    vigorous = _bout("PAQ650", "PAQ655", "PAD660")
    moderate = _bout("PAQ665", "PAQ670", "PAD675")
    if vigorous.notna().any() or moderate.notna().any():
        exercise = vigorous.fillna(0) + moderate.fillna(0)
        exercise[vigorous.isna() & moderate.isna()] = np.nan
        exercise = exercise.clip(0, 10)
    else:
        exercise = pd.Series(np.nan, index=df.index)

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
        # Tobacco exposure measured in blood, and inflammation. Added because
        # the earlier finding that "bloodwork does not help the general panel"
        # was only ever tested against a blood count and a metabolic panel, and
        # never against these two.
        "cotinine": (pd.to_numeric(df["LBXCOT"], errors="coerce")
                     if "LBXCOT" in df.columns else np.nan),
        # Two different CRP assays with two different units, and mixing them up
        # would be a tenfold error in a feature.
        #
        #   LBXCRP    the older assay, mg/dL, median about 0.12, so x10 to mg/L
        #   LBXHSCRP  the high sensitivity assay used from 2015, ALREADY mg/L,
        #             median about 1.2, so no conversion
        #
        # Checked by pulling HSCRP_I and reading the distribution rather than by
        # assuming the newer file was a drop-in rename.
        "crp": (pd.to_numeric(df["LBXCRP"], errors="coerce") * 10.0
                if "LBXCRP" in df.columns
                else (pd.to_numeric(df["LBXHSCRP"], errors="coerce")
                      if "LBXHSCRP" in df.columns else np.nan)),
        "waist": pd.to_numeric(df.get("BMXWAIST"), errors="coerce"),
        # Lifestyle
        "smoking": smoking,
        "alcohol_intake": alcohol,
        "physical_activity": exercise,
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
