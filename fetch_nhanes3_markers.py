"""
NHANES III (1988-1994) cohort for replicating the cheap-marker tests in a separate survey.

Takes the blood count with its three-part differential, albumin, RDW, BMI, smoking and
earlier self-reported non-skin cancer, links deaths through 2019, and writes one file.

Differential: NHANES III reports lymphocyte, mononuclear (monocyte) and granulocyte numbers
(Coulter), not the five-part differential of the continuous survey. Granulocytes include
eosinophils and basophils, so neutrophil-based markers become granulocyte analogues.

Downloads about 260 MB (lab 40 MB, adult 65 MB, exam 190 MB) from CDC.

Run:  python fetch_nhanes3_markers.py     writes data/nhanes3_markers.csv.gz
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat"
ADULT = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"
EXAM = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat"
MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_III_MORT_2019_PUBLIC.dat"
OUT = os.path.join("data", "nhanes3_markers.csv.gz")

LAB_SPEC = {
    "SEQN": (1, 5), "HSSEX": (15, 15), "HSAGEIR": (16, 17),
    "wbc": (1273, 1277), "lym_pct": (1283, 1287), "mono_pct": (1288, 1292), "gran_pct": (1293, 1297),
    "lym_n": (1298, 1302), "mono_n": (1303, 1306), "gran_n": (1307, 1311),
    "hemoglobin": (1320, 1324), "rdw": (1360, 1364), "platelets": (1371, 1375), "albumin": (1846, 1848),
}
EXAM_SPEC = {"SEQN": (1, 5), "bmi_raw": (1524, 1527)}
ADULT_SPEC = {"SEQN": (1, 5), "other_cancer": (1479, 1479), "smoked100": (2281, 2281)}
MORT_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)]
MORT_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_INT", "PERMTH_EXM"]

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=1800, context=_ctx).read()


def fixed(raw, spec):
    return pd.read_fwf(io.BytesIO(raw), colspecs=[(a - 1, b) for a, b in spec.values()], names=list(spec))


def clean(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s >= 88888).mask(s.isin([8888, 9999, 888, 999, 88, 99]))


def main():
    print("Downloading NHANES III lab file ...", flush=True)
    lab = fixed(get(LAB), LAB_SPEC)
    print(f"  {len(lab):,} rows", flush=True)
    print("Downloading adult file ...", flush=True)
    adult = fixed(get(ADULT), ADULT_SPEC)
    print("Downloading exam file ...", flush=True)
    exam = fixed(get(EXAM), EXAM_SPEC)
    mort = fixed(get(MORT), dict(zip(MORT_NAMES, MORT_COLSPECS))) if False else pd.read_fwf(io.BytesIO(get(MORT)), colspecs=MORT_COLSPECS, names=MORT_NAMES)
    df = lab.merge(adult, on="SEQN", how="left").merge(exam, on="SEQN", how="left").merge(mort, on="SEQN", how="inner")
    df["age"] = clean(df["HSAGEIR"])
    df["male"] = (pd.to_numeric(df["HSSEX"], errors="coerce") == 1).astype(float)
    for c in ("wbc", "lym_pct", "mono_pct", "gran_pct", "lym_n", "mono_n", "gran_n", "hemoglobin", "rdw", "platelets", "albumin"):
        df[c] = clean(df[c])
    # the counts are thousands per microlitre; the differential numbers are written with two implied decimals
    for c, scale in (("wbc", 10.0), ("lym_n", 100.0), ("mono_n", 100.0), ("gran_n", 100.0), ("hemoglobin", 10.0), ("rdw", 10.0), ("albumin", 10.0)):
        med = float(df[c].median())
        print(f"    {c:<12} raw median {med:>9.2f}")
    bmi = clean(df["bmi_raw"])
    df["bmi"] = bmi / 10.0 if 10 < float(bmi.median()) / 10.0 < 60 else bmi
    df["smoked"] = np.where(pd.to_numeric(df["smoked100"], errors="coerce") == 1, 1.0, np.where(pd.to_numeric(df["smoked100"], errors="coerce") == 2, 0.0, np.nan))
    df["prior_cancer"] = (pd.to_numeric(df["other_cancer"], errors="coerce") == 1).astype(int)
    df = df[(df["ELIGSTAT"] == 1) & (df["age"] >= 20)].rename(columns={"MORTSTAT": "died", "UCOD_LEADING": "ucod", "PERMTH_EXM": "months"})
    keep = ["SEQN", "age", "male", "smoked", "bmi", "prior_cancer", "wbc", "lym_pct", "mono_pct", "gran_pct", "lym_n", "mono_n", "gran_n",
            "hemoglobin", "rdw", "platelets", "albumin", "died", "ucod", "months"]
    df = df[keep]
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"\nwrote {OUT}: {len(df):,} adults, {int((df['died'] == 1).sum()):,} deaths, {int(((df['died'] == 1) & (df['ucod'] == 2)).sum())} from cancer")
    print(df[["age", "bmi", "wbc", "lym_n", "mono_n", "gran_n", "hemoglobin", "rdw", "platelets", "albumin"]].describe().loc[["count", "mean", "50%"]].round(2).to_string())


if __name__ == "__main__":
    main()
