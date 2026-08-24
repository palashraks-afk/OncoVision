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

    shared = sorted(set(ilpd.columns) & set(hcv.columns) - {"liver_disease"})
    print(f"\nShared liver features across India and Germany ({len(shared)}): {', '.join(shared)}")
    print("These two cohorts are independent, so one can train and the other can test.")

    print("\nDistribution shift between the two cohorts (median values):")
    print(f"  {'feature':<24}{'India':>12}{'Germany':>12}")
    for c in shared:
        print(f"  {c:<24}{ilpd[c].median():>12.2f}{hcv[c].median():>12.2f}")


if __name__ == "__main__":
    main()
