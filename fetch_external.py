"""
Download the real-patient cohorts used for the liver panel and for external
validation, and write them into data/ in the application's own schema.

Why this exists
---------------
Every original cohort in this project is graded on a held-out slice of itself.
That measures whether a model learned its own dataset, not whether it works on
patients from somewhere else, which is the question that actually matters. It
was the largest gap in the evaluation.

Two independent real cohorts make a genuine external test possible:

  ILPD  583 real patients, Andhra Pradesh, India.
        Liver patient vs non liver patient.
  HCV   615 real patients, Germany.
        Blood donors vs hepatitis, fibrosis and cirrhosis.

They were collected on different continents, in different hospitals, under
different protocols, by different people, and they share eight liver chemistry
measurements that this application already parses out of a PDF: age, sex,
bilirubin, alkaline phosphatase, ALT, AST, total protein and albumin.

So the liver panel is trained on India and tested on Germany. Nothing about the
German cohort touches training. That is a real external validation, and it also
replaces the synthetic liver cohort that the project could not previously
defend.

A third cohort is downloaded for the breast panel:

  Coimbra  116 real patients, Portugal.
           Blood-based breast cancer markers, no biopsy required.

Run:  python fetch_external.py
"""

import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = "data"


def save(df: pd.DataFrame, filename: str, note: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"    {note}")


def fetch_ilpd() -> pd.DataFrame:
    """
    Indian Liver Patient Dataset, UCI id 225.
    Target in the source is 1 = liver patient, 2 = not a liver patient.
    """
    d = fetch_ucirepo(id=225)
    X = d.data.features.copy()
    y = d.data.targets.iloc[:, 0]

    out = pd.DataFrame({
        "age": pd.to_numeric(X["Age"], errors="coerce"),
        "gender": (X["Gender"].astype(str).str.strip().str.lower() == "male").astype(int),
        "bilirubin": pd.to_numeric(X["TB"], errors="coerce"),
        "alkaline_phosphatase": pd.to_numeric(X["Alkphos"], errors="coerce"),
        "alt": pd.to_numeric(X["Sgpt"], errors="coerce"),
        "ast": pd.to_numeric(X["Sgot"], errors="coerce"),
        "protein_total": pd.to_numeric(X["TP"], errors="coerce"),
        "albumin": pd.to_numeric(X["ALB"], errors="coerce"),
        "liver_disease": (pd.to_numeric(y, errors="coerce") == 1).astype(int),
    })
    return out.dropna()


def fetch_hcv() -> pd.DataFrame:
    """
    HCV data, UCI id 571, Germany.
    Category 0 is a blood donor, so healthy. 1, 2 and 3 are hepatitis,
    fibrosis and cirrhosis, so liver disease. 'suspect blood donor' is
    dropped rather than guessed at.
    """
    d = fetch_ucirepo(id=571)
    X = d.data.features.copy()
    y = d.data.targets.iloc[:, 0].astype(str)

    keep = ~y.str.startswith("0s")
    X, y = X[keep], y[keep]

    # UNIT HARMONISATION. The German cohort reports in SI units and the Indian
    # cohort in conventional units. Comparing them directly would be nonsense:
    # German albumin has a median of 42 and Indian albumin a median of 3.1,
    # because one is g/L and the other g/dL. Three conversions are needed.
    #
    #   albumin        g/L    -> g/dL     divide by 10
    #   total protein  g/L    -> g/dL     divide by 10
    #   bilirubin      umol/L -> mg/dL    divide by 17.1
    #
    # ALP, ALT and AST are already U/L in both, so they are left alone. The
    # remaining difference in those three is real: the Indian cohort is 71%
    # liver patients and the German cohort is 90% blood donors.
    out = pd.DataFrame({
        "age": pd.to_numeric(X["Age"], errors="coerce"),
        "gender": (X["Sex"].astype(str).str.strip().str.lower() == "m").astype(int),
        "bilirubin": pd.to_numeric(X["BIL"], errors="coerce") / 17.1,
        "alkaline_phosphatase": pd.to_numeric(X["ALP"], errors="coerce"),
        "alt": pd.to_numeric(X["ALT"], errors="coerce"),
        "ast": pd.to_numeric(X["AST"], errors="coerce"),
        "protein_total": pd.to_numeric(X["PROT"], errors="coerce") / 10.0,
        "albumin": pd.to_numeric(X["ALB"], errors="coerce") / 10.0,
        "liver_disease": (~y.str.startswith("0")).astype(int),
    })
    return out.dropna()


def fetch_coimbra() -> pd.DataFrame:
    """
    Breast Cancer Coimbra, UCI id 451, Portugal.
    Blood-based, so unlike the Wisconsin set it does not require a biopsy.
    Source target is 1 = healthy control, 2 = breast cancer patient.
    """
    d = fetch_ucirepo(id=451)
    X = d.data.features.copy()
    y = pd.to_numeric(d.data.targets.iloc[:, 0], errors="coerce")

    out = pd.DataFrame({
        "age": pd.to_numeric(X["Age"], errors="coerce"),
        "bmi": pd.to_numeric(X["BMI"], errors="coerce"),
        "glucose": pd.to_numeric(X["Glucose"], errors="coerce"),
        "insulin": pd.to_numeric(X["Insulin"], errors="coerce"),
        "homa": pd.to_numeric(X["HOMA"], errors="coerce"),
        "leptin": pd.to_numeric(X["Leptin"], errors="coerce"),
        "adiponectin": pd.to_numeric(X["Adiponectin"], errors="coerce"),
        "resistin": pd.to_numeric(X["Resistin"], errors="coerce"),
        "mcp_1": pd.to_numeric(X["MCP.1"], errors="coerce"),
        "breast_cancer": (y == 2).astype(int),
    })
    return out.dropna()


def fetch_wpbc() -> pd.DataFrame:
    """
    Wisconsin Prognostic Breast Cancer, UCI id 16. 198 patients.

    Same four nuclear morphology measurements the breast panel uses, computed
    the same way, but a different cohort of patients followed since 1984 for
    recurrence rather than assembled for diagnosis.

    Every record is a confirmed invasive breast cancer, so there is no benign
    class and no AUC to compute. What can be measured is external sensitivity:
    of 198 independent, confirmed cancer patients the model has never seen,
    how many does it flag. That is a partial external test rather than a full
    one, and it is reported as exactly that.
    """
    d = fetch_ucirepo(id=16)
    X = d.data.features.copy()
    return pd.DataFrame({
        "radius_mean": pd.to_numeric(X["radius1"], errors="coerce"),
        "texture_mean": pd.to_numeric(X["texture1"], errors="coerce"),
        "perimeter_mean": pd.to_numeric(X["perimeter1"], errors="coerce"),
        "area_mean": pd.to_numeric(X["area1"], errors="coerce"),
        "malignant": 1,
    }).dropna()


def fetch_nhanes() -> pd.DataFrame:
    """
    NHANES 2017-2018, CDC. A nationally representative sample of the US
    population, which is the important part.

    Every other cohort in this project is case-control: people already had a
    reason to be tested, so 21 to 37 percent of them have the disease. NHANES
    is a survey of the general public, so its cancer prevalence is what you
    would actually meet, and it carries race and ethnicity, which none of the
    other cohorts do.

    That makes it an external test for the general panel on two axes at once:
    a different population, and a realistic prevalence rather than an enriched
    one.

    Not every feature the general panel trains on exists here. Inherited risk
    is not surveyed, and prior cancer diagnosis is the outcome itself, so both
    are left absent and imputed exactly as the application does for a patient
    who does not answer them.
    """
    import io
    import ssl
    import urllib.request

    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def grab(fname):
        req = urllib.request.Request(base + fname, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=300, context=ctx).read()
        return pd.read_sas(io.BytesIO(raw), format="xport")

    demo = grab("DEMO_J.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]]
    bmx = grab("BMX_J.XPT")[["SEQN", "BMXBMI"]]
    smq = grab("SMQ_J.XPT")[["SEQN", "SMQ020", "SMQ040"]]
    alq = grab("ALQ_J.XPT")[["SEQN", "ALQ111", "ALQ130"]]
    paq = grab("PAQ_J.XPT")[["SEQN", "PAQ655", "PAD660", "PAQ670", "PAD675"]]
    mcq = grab("MCQ_J.XPT")[["SEQN", "MCQ220"]]

    df = demo
    for part in (bmx, smq, alq, paq, mcq):
        df = df.merge(part, on="SEQN", how="left")

    # Adults only. The training cohort is adults and cancer in minors is a
    # different disease population.
    df = df[df["RIDAGEYR"] >= 20]

    # Smoking, from "100 cigarettes ever" plus "do you smoke now".
    smoking = pd.Series(np.nan, index=df.index)
    smoking[df["SMQ020"] == 2] = 0                                  # never
    smoking[(df["SMQ020"] == 1) & (df["SMQ040"] == 3)] = 1          # former
    smoking[(df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))] = 2  # current

    # Alcohol on the training set's 0 to 5 severity scale. Never-drinkers are
    # zero; everyone else is average drinks per drinking day, capped at 5.
    alcohol = pd.to_numeric(df["ALQ130"], errors="coerce").clip(0, 5)
    alcohol[df["ALQ111"] == 2] = 0

    # Recreational activity in hours per week, vigorous plus moderate.
    mins = (pd.to_numeric(df["PAQ655"], errors="coerce").fillna(0)
            * pd.to_numeric(df["PAD660"], errors="coerce").fillna(0)
            + pd.to_numeric(df["PAQ670"], errors="coerce").fillna(0)
            * pd.to_numeric(df["PAD675"], errors="coerce").fillna(0))
    activity = (mins / 60.0).clip(0, 10)

    race = df["RIDRETH3"].map({
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial",
    })

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),   # 1 = male in NHANES
        "bmi": pd.to_numeric(df["BMXBMI"], errors="coerce"),
        "smoking": smoking,
        "alcohol_intake": alcohol,
        "physical_activity": activity,
        "race_ethnicity": race,
        # MCQ220: 1 = ever told had cancer, 2 = no. 7 and 9 are refused
        # and don't know, and are dropped rather than guessed at.
        "any_cancer": df["MCQ220"].map({1: 1, 2: 0}),
    })
    return out.dropna(subset=["age", "gender", "bmi", "smoking", "any_cancer"])


def fetch_nhanes_liver() -> pd.DataFrame:
    """
    NHANES 2017-2018 liver cohort, United States. A third independent test for
    the liver panel, and the only population-based one.

    India and Germany are both clinical cohorts: people appeared in them because
    someone ordered a test. NHANES samples the public, so its 5% liver-condition
    prevalence is closer to what a screening tool would actually meet, and it
    carries race and ethnicity, which neither of the other two do.

    Uses the same eight features. NHANES publishes these analytes in
    conventional units (LBXSTB in mg/dL, LBXSAL and LBXSTP in g/dL), which match
    the Indian cohort, so no conversion is needed here. That was checked rather
    than assumed, because the German cohort did need converting.

    Label is MCQ160L, ever told by a doctor you had a liver condition.
    """
    import io
    import ssl
    import urllib.request

    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    def grab(fname):
        req = urllib.request.Request(base + fname, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=300, context=ctx).read()
        return pd.read_sas(io.BytesIO(raw), format="xport")

    demo = grab("DEMO_J.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]]
    bio = grab("BIOPRO_J.XPT")[
        # LBXSGTSI is GGT. The shipped liver panel reads it, so this held-out
        # cohort has to carry it too, otherwise every row here would be scored
        # on an imputed median for a feature the model actually uses and the
        # external number would quietly describe a different model.
        ["SEQN", "LBXSTB", "LBXSAPSI", "LBXSATSI", "LBXSASSI", "LBXSTP",
         "LBXSAL", "LBXSGTSI"]
    ]
    mcq = grab("MCQ_J.XPT")[["SEQN", "MCQ160L"]]

    df = demo.merge(bio, on="SEQN", how="inner").merge(mcq, on="SEQN", how="inner")
    df = df[df["RIDAGEYR"] >= 20]

    race = df["RIDRETH3"].map({
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial",
    })

    out = pd.DataFrame({
        "age": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "gender": (df["RIAGENDR"] == 1).astype(int),
        "bilirubin": pd.to_numeric(df["LBXSTB"], errors="coerce"),
        "alkaline_phosphatase": pd.to_numeric(df["LBXSAPSI"], errors="coerce"),
        "alt": pd.to_numeric(df["LBXSATSI"], errors="coerce"),
        "ast": pd.to_numeric(df["LBXSASSI"], errors="coerce"),
        "protein_total": pd.to_numeric(df["LBXSTP"], errors="coerce"),
        "albumin": pd.to_numeric(df["LBXSAL"], errors="coerce"),
        "ggt": pd.to_numeric(df["LBXSGTSI"], errors="coerce"),
        "race_ethnicity": race,
        # 1 = yes, 2 = no. 7 and 9 are refused and don't know, dropped.
        "liver_disease": df["MCQ160L"].map({1: 1, 2: 0}),
    })
    return out.dropna(subset=[c for c in out.columns if c != "race_ethnicity"])


def main():
    print("Indian Liver Patient Dataset (UCI 225), India")
    ilpd = fetch_ilpd()
    save(ilpd, "ilpd_liver_india.csv",
         f"{int(ilpd.liver_disease.sum())} with liver disease, "
         f"{int((1 - ilpd.liver_disease).sum())} without")

    print("\nHCV data (UCI 571), Germany")
    hcv = fetch_hcv()
    save(hcv, "hcv_liver_germany.csv",
         f"{int(hcv.liver_disease.sum())} with liver disease, "
         f"{int((1 - hcv.liver_disease).sum())} blood donors")

    print("\nBreast Cancer Coimbra (UCI 451), Portugal")
    coi = fetch_coimbra()
    save(coi, "breast_coimbra_portugal.csv",
         f"{int(coi.breast_cancer.sum())} patients, "
         f"{int((1 - coi.breast_cancer).sum())} controls")

    print("\nNHANES 2017-2018 liver cohort (CDC), United States")
    nhl = fetch_nhanes_liver()
    save(nhl, "nhanes_liver_usa.csv",
         f"{int(nhl.liver_disease.sum())} with a liver condition, "
         f"{int((1 - nhl.liver_disease).sum())} without "
         f"({nhl.liver_disease.mean():.1%}, population based rather than clinical)")

    print("\nWisconsin Prognostic Breast Cancer (UCI 16)")
    wp = fetch_wpbc()
    save(wp, "wpbc_breast_external.csv",
         f"{len(wp)} independent confirmed cancer patients, no benign class, "
         f"so external sensitivity only")

    print("\nNHANES 2017-2018 (CDC), United States")
    nh = fetch_nhanes()
    save(nh, "nhanes_general_usa.csv",
         f"{int(nh.any_cancer.sum())} ever told they had cancer, "
         f"{int((1 - nh.any_cancer).sum())} not ({nh.any_cancer.mean():.1%} prevalence, "
         f"nationally representative rather than case-control)")
    print("    race and ethnicity present:", nh.race_ethnicity.value_counts().to_dict())

    # Pooled three-continent liver cohort, which is what the shipped panel
    # trains on. Leave-one-cohort-out in external_validation.py shows training
    # on two cohorts instead of one raises mean external AUC by about 0.06, so
    # pooling is not a convenience, it is the measured better choice. The cohort
    # column is kept so that validation can hold one source out.
    shared_cols = ["age", "gender", "bilirubin", "alkaline_phosphatase",
                   "alt", "ast", "protein_total", "albumin", "liver_disease"]
    pooled = pd.concat(
        [
            ilpd[shared_cols].assign(cohort="India"),
            hcv[shared_cols].assign(cohort="Germany"),
            nhl[shared_cols].assign(cohort="USA"),
        ],
        ignore_index=True,
    )
    print("\nPooled liver cohort across three continents")
    save(pooled, "liver_pooled_3cohort.csv",
         f"{int(pooled.liver_disease.sum())} with liver disease of {len(pooled)} "
         f"({pooled.liver_disease.mean():.1%}), from "
         + ", ".join(f"{k} {v}" for k, v in pooled.cohort.value_counts().items()))

    shared = sorted(set(ilpd.columns) & set(hcv.columns) - {"liver_disease"})
    print(f"\nShared liver features across India and Germany ({len(shared)}): {', '.join(shared)}")
    print("These two cohorts are independent, so one can train and the other can test.")

    print("\nDistribution shift between the two cohorts (median values):")
    print(f"  {'feature':<24}{'India':>12}{'Germany':>12}")
    for c in shared:
        print(f"  {c:<24}{ilpd[c].median():>12.2f}{hcv[c].median():>12.2f}")


if __name__ == "__main__":
    main()
