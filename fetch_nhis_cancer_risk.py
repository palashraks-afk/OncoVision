"""
National Health Interview Survey (NHIS) 1997-2009, linked to the National Death Index through 2019.

Why NHIS
--------
Cancer death is rare in NHANES (about 1,300 deaths in 40,000 adults), which is too few to ask
how cancer risk differs by smoking, education or race, or what share of adults in their forties
already carry the risk of a fifty-year-old. NHIS is a much larger national interview survey with
public-use mortality linkage: about 180,000 adults aged 35 to 84 without earlier cancer in the analysis, with ten or more years of follow-up
and several thousand cancer deaths. It carries only FREE information, the kind a person can answer
in a minute with no test: age, sex, height and weight, smoking, schooling, race and ethnicity,
marital status, self-rated health.

What is built
-------------
Sample Adult and Person files for 1997-2002 and 2005-2009 (2003 has the adult file only as a loose
.DAT and no person file merge key problems were found, 2004 has no layout file in the standard
place, so 2004 is left out and said so). The layouts are read from NCHS's own SAS programs, so no
column position is typed by hand. Deaths come from the public-use Linked Mortality Files (follow-up
through 31 December 2019); the key is SRVY_YR + HHX + FMX + FPX (PX before 2000), 14 characters.

Outcome: death from cancer (underlying cause 2, ICD-10 C00-C97) within ten years of interview.
Everyone alive at ten years, or dead of another cause, is a non-case; nobody is dropped for how they died.
People who reported a cancer diagnosis at interview are removed (cancer history is not a prediction target here).

Run:  python fetch_nhis_cancer_risk.py      writes data/nhis_cancer_risk.csv.gz
"""

import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

requests.packages.urllib3.disable_warnings()
H = {"User-Agent": "Mozilla/5.0"}
DATA = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/{y}/{f}"
CODE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Program_Code/NHIS/{y}/{f}.sas"
LMF = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHIS_{y}_MORT_2019_PUBLIC.dat"
YEARS = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009]
OUT = os.path.join("data", "nhis_cancer_risk.csv.gz")

ADULT_VARS = ["SRVY_YR", "HHX", "FMX", "FPX", "PX", "INTV_QRT", "WTFA_SA", "AGE_P", "SEX", "BMI", "SMKEV", "SMKNOW", "SMKSTAT2", "CIGSDAY", "SMKQTNO", "SMKQTTP",
              "CANEV", "VIGFREQW", "REGION"]
PERSON_VARS = ["SRVY_YR", "HHX", "FMX", "FPX", "PX", "EDUC", "EDUC1", "PHSTAT", "R_MARITL", "HISPAN_I", "HISPAN_P", "RACERPI2", "RACEREC", "RACE", "ORIGIN_I", "ORIGIN"]


def get(url):
    r = requests.get(url, verify=False, timeout=240, headers=H)
    r.raise_for_status()
    return r.content


def parse_layout(txt):
    best = {}
    for m0 in re.finditer(r"\bINPUT\b", txt):
        j = txt.find(";", m0.end())
        out = {}
        for m in re.finditer(r"([A-Z][A-Z0-9_]*)\s+(\$\s*)?(\d+)\s*-\s*(\d+)(?:\s+\.(\d))?", txt[m0.end():j]):
            out[m.group(1)] = (int(m.group(3)), int(m.group(4)), int(m.group(5) or 0))
        if len(out) > len(best):
            best = out
    return best


def read_file(raw, layout, wanted):
    use = [v for v in wanted if v in layout]
    cols = [(layout[v][0] - 1, layout[v][1]) for v in use]
    df = pd.read_fwf(io.BytesIO(raw), colspecs=cols, names=use, dtype=str)
    for v in use:
        if layout[v][2]:
            df[v] = pd.to_numeric(df[v], errors="coerce") / (10 ** layout[v][2])
    return df


def key(df):
    fp = df["FPX"] if "FPX" in df else df["PX"]
    return df["HHX"].str.strip().str.zfill(6) + df["FMX"].str.strip().str.zfill(2) + fp.str.strip().str.zfill(2)


def one_year(y):
    lay_a = parse_layout(get(CODE.format(y=y, f="samadult")).decode("latin-1"))
    lay_p = parse_layout(get(CODE.format(y=y, f="personsx")).decode("latin-1"))
    if y == 2003:
        raw_a = get(f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2003/samadult/SAMADULT.DAT")
    else:
        z = zipfile.ZipFile(io.BytesIO(get(DATA.format(y=y, f="samadult.zip"))))
        raw_a = z.read(z.namelist()[0])
    zp = zipfile.ZipFile(io.BytesIO(get(DATA.format(y=y, f="personsx.zip"))))
    raw_p = zp.read(zp.namelist()[0])
    a = read_file(raw_a, lay_a, ADULT_VARS)
    p = read_file(raw_p, lay_p, PERSON_VARS)
    a["k"], p["k"] = key(a), key(p)
    p = p.drop(columns=[c for c in ("SRVY_YR", "HHX", "FMX", "FPX", "PX") if c in p.columns]).drop_duplicates("k")
    df = a.merge(p, on="k", how="left")
    df["pid"] = df["SRVY_YR"].astype(str).str.strip() + df["k"]
    mort = pd.read_fwf(io.BytesIO(get(LMF.format(y=y))), colspecs=[(0, 14), (14, 15), (15, 16), (16, 19), (22, 23), (22, 26)], names=["pid", "elig", "died", "ucod", "dodq", "dody"], dtype=str)
    mort = pd.read_fwf(io.BytesIO(get(LMF.format(y=y))), colspecs=[(0, 14), (14, 15), (15, 16), (16, 19), (21, 22), (22, 26)], names=["pid", "elig", "died", "ucod", "dodq", "dody"], dtype=str)
    df = df.merge(mort, on="pid", how="left")
    df["year"] = y
    print(f"  {y}: {len(df):,} adults, {int(df['died'].notna().sum()):,} linked, ids matched {df['elig'].notna().mean():.1%}", flush=True)
    return df


def clean(df):
    n = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)  # noqa: E731
    o = pd.DataFrame({"year": df["year"], "age": n("AGE_P"), "male": (n("SEX") == 1).astype(float), "bmi": n("BMI"), "weight": n("WTFA_SA")})
    o.loc[o["bmi"] > 90, "bmi"] = np.nan
    ever, now = n("SMKEV"), n("SMKNOW")
    status = np.where(ever == 2, 0, np.where((ever == 1) & now.isin([1, 2]), 2, np.where((ever == 1) & (now == 3), 1, np.nan)))
    o["smoke_status"] = status                                   # 0 never, 1 former, 2 current
    cig = n("CIGSDAY").where(n("CIGSDAY").between(1, 96))
    o["cigs_per_day"] = np.where(status == 2, cig, 0.0)
    qn, qt = n("SMKQTNO"), n("SMKQTTP")
    unit = {1: 1 / 365.25, 2: 1 / 52.18, 3: 1 / 12.0, 4: 1.0}
    yrs = qn.where(qn.between(0, 80)) * qt.map(unit)
    o["quit_years"] = np.where(status == 1, yrs, np.nan)
    edu = n("EDUC1").where(n("EDUC1").notna(), n("EDUC")) if "EDUC1" in df else n("EDUC")
    o["educ_raw"] = edu
    o["self_health"] = n("PHSTAT").where(n("PHSTAT").between(1, 5))
    o["married"] = n("R_MARITL").isin([1, 2, 3]).astype(float).where(n("R_MARITL").between(1, 9))
    o["vigorous_per_week"] = n("VIGFREQW").where(n("VIGFREQW").between(0, 95))
    hisp = n("HISPAN_I").where(n("HISPAN_I").notna(), n("HISPAN_P")) if "HISPAN_I" in df else n("HISPAN_P")
    race = n("RACERPI2") if "RACERPI2" in df else n("RACEREC")
    o["hisp_raw"], o["race_raw"] = hisp, race
    o["prior_cancer"] = (n("CANEV") == 1).astype(int)
    o["elig"] = pd.to_numeric(df["elig"], errors="coerce")
    o["mortstat"] = pd.to_numeric(df["died"], errors="coerce")
    o["ucod"] = pd.to_numeric(df["ucod"], errors="coerce")
    q = n("INTV_QRT").where(n("INTV_QRT").between(1, 4), 2.0)
    o["t0"] = o["year"] + (q - 0.5) / 4.0
    dq = pd.to_numeric(df["dodq"], errors="coerce")
    dy = pd.to_numeric(df["dody"], errors="coerce")
    o["tdeath"] = dy + (dq - 0.5) / 4.0
    return o


def main():
    frames = []
    print("NHIS 1997-2009 linked to the National Death Index through 2019\n", flush=True)
    for y in YEARS:
        try:
            frames.append(clean(one_year(y)))
        except Exception as e:  # report and carry on, the year is left out and said so
            print(f"  {y}: SKIPPED ({type(e).__name__}: {e})", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["age"] >= 18) & (d["elig"] == 1)]
    d["months"] = np.where(d["mortstat"] == 1, (d["tdeath"] - d["t0"]) * 12, (2020.0 - d["t0"]) * 12)
    os.makedirs("data", exist_ok=True)
    d.to_csv(OUT, index=False, compression="gzip")
    died = d["mortstat"] == 1
    print(f"\nwrote {OUT}: {len(d):,} adults, {int(died.sum()):,} deaths, {int((died & (d['ucod'] == 2)).sum()):,} from cancer")
    print(d[["age", "male", "bmi", "smoke_status", "cigs_per_day", "quit_years", "educ_raw", "self_health", "months"]].describe().loc[["count", "mean", "50%"]].round(2).to_string())
    print("\neducation raw codes:", d["educ_raw"].value_counts().sort_index().to_dict() if d["educ_raw"].nunique() < 40 else "many")
    print("race raw:", d["race_raw"].value_counts().sort_index().to_dict(), " hispanic raw:", d["hisp_raw"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        YEARS = [int(a) for a in sys.argv[1:]]
        OUT = os.path.join("data", "nhis_test.csv.gz")
    main()
