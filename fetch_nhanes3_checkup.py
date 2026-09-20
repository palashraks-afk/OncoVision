"""
NHANES III with the rest of the checkup, as the external test of a revival.

Why
---
On the continuous survey, adding HbA1c, cholesterol, HDL, urine albumin, blood
pressure and waist to the general panel cleared both bars this project sets for
a withdrawn panel: it beat the stronger age-and-sex model by +0.016 and its
rule-out cut excluded 4.9% more adults at the same share of cancers caught.

That is an internal result, and the standing rule here is that an internal
result cannot revive anything. Section 4.2 of the paper exists because the
prospective panel gained inside NHANES, held on leave-one-cycle-out, and then
reversed on NHANES III. Cycles of one survey share a protocol, a laboratory
contract and a pipeline; a cohort from 1988-1994 shares none of them.

So the same panel is rebuilt on NHANES III and the revival stands or falls
there.

What harmonises and what does not
---------------------------------
    age, gender, blood count,    direct, same positions the bowel external
    metabolic panel              cohort already uses
    HbA1c        GHP             glycated haemoglobin, percent
    cholesterol  TCP             total cholesterol, mg/dL
    HDL          HDP             mg/dL
    urine albumin UAP            mg/L
    systolic, diastolic          PEPMNK1R and PEPMNK5R, the averaged readings
    BMI, waist   BMPBMI, BMPWAIST

NOT harmonised: triglycerides and the urine albumin-to-creatinine ratio.
NHANES III measured triglycerides only in the fasting subsample and did not
publish a urinary creatinine in this file, so the ratio cannot be built. Both
are filled with the training median, which is what the service does for a value
a patient does not have. That makes this a test of the panel as used by someone
missing two lines of their report, and it is written up that way.

Every parsed column is checked against the range a population of adults
actually occupies. These are fixed-width fields with implied decimals, and a
misread moves a whole column by a factor of ten while leaving every downstream
number looking plausible -- which already happened once in this project, to BMI.

Run:  python fetch_nhanes3_checkup.py
"""

import io
import ssl
import urllib.request

import pandas as pd

ADULT = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"
LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat"
EXAM = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat"
OUT = "data/nhanes3_general_checkup.csv"

RECENT_YEARS = 4

LAB_SPEC = {
    "SEQN": (1, 5), "HSSEX": (15, 15), "HSAGEIR": (16, 17),
    "wbc": (1273, 1277), "rbc": (1312, 1315), "hemoglobin": (1320, 1324),
    "hematocrit": (1330, 1334), "mcv": (1340, 1344), "mch": (1345, 1349),
    "rdw": (1360, 1364), "platelets": (1371, 1375), "mpv": (1386, 1390),
    "calcium": (1731, 1734), "glucose": (1758, 1760), "bun": (1766, 1768),
    "bilirubin": (1774, 1777), "creatinine": (1784, 1787),
    "ast": (1821, 1823), "alt": (1824, 1826),
    "alkaline_phosphatase": (1835, 1838), "protein_total": (1839, 1842),
    "albumin": (1846, 1848),
    # The rest of the checkup, from lab.sas
    "cholesterol_total": (1598, 1600),     # TCP, mg/dL
    "hdl": (1622, 1624),                   # HDP, mg/dL
    "urine_albumin": (1749, 1752),         # UAP, mg/L
    "hba1c": (1861, 1864),                 # GHP, percent
}

EXAM_SPEC = {
    "SEQN": (1, 5),
    "systolic_bp": (1423, 1425),           # PEPMNK1R
    "diastolic_bp": (1428, 1430),          # PEPMNK5R
    "bmi_raw": (1524, 1527),               # BMPBMI, Z6.1, decimal written
    "waist_raw": (1590, 1594),             # BMPWAIST, Z7.1, decimal written
}

ADULT_SPEC = {
    "SEQN": (1, 5),
    "age_exam": (18, 19),
    "skin_cancer": (1478, 1478),
    "other_cancer": (1479, 1479),
    "age_at_dx": (1524, 1526),
}

# What a population of adults actually looks like. A column landing outside its
# range means the field was parsed wrong, and the run stops rather than writing
# a cohort that looks fine and is not.
PLAUSIBLE = {
    "hba1c": (4.0, 7.5), "cholesterol_total": (150.0, 240.0), "hdl": (30.0, 70.0),
    "systolic_bp": (100.0, 145.0), "diastolic_bp": (55.0, 95.0),
    "bmi": (20.0, 35.0), "waist": (75.0, 115.0), "urine_albumin": (0.1, 60.0),
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=1800, context=_ctx).read()


def read_fixed(raw, spec):
    return pd.read_fwf(io.BytesIO(raw), colspecs=[(a - 1, b) for a, b in spec.values()],
                       names=list(spec))


def clean(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s >= 88888).mask(s.isin([8888, 9999, 888, 999, 88, 99]))


def scaled(series, name):
    """The column, with its implied decimal resolved by measurement not guesswork."""
    s = clean(series)
    lo, hi = PLAUSIBLE[name]
    med = float(s.median())
    if lo <= med <= hi:
        print(f"    {name:<20} median {med:>8.1f}  as written")
        return s
    if lo <= med / 10 <= hi:
        print(f"    {name:<20} median {med / 10:>8.1f}  one implied decimal applied")
        return s / 10.0
    if lo <= med * 10 <= hi:
        print(f"    {name:<20} median {med * 10:>8.1f}  scaled up by ten")
        return s * 10.0
    raise ValueError(f"{name}: median parses to {med}, which is not a plausible adult "
                     f"value at any scale between {lo} and {hi}. The byte positions are wrong.")


def main():
    print("NHANES III (1988-1994) with the rest of the checkup\n")
    print("  downloading the household adult file ...", flush=True)
    adult = read_fixed(get(ADULT), ADULT_SPEC)
    print("  downloading the laboratory file ...", flush=True)
    lab = read_fixed(get(LAB), LAB_SPEC)
    print("  downloading the examination file ...", flush=True)
    exam = read_fixed(get(EXAM), EXAM_SPEC)

    df = lab.merge(exam, on="SEQN", how="left").merge(adult, on="SEQN", how="left")
    age = pd.to_numeric(df["HSAGEIR"], errors="coerce")
    df = df[age >= 20].copy()

    out = pd.DataFrame({"age": pd.to_numeric(df["HSAGEIR"], errors="coerce"),
                        "gender": (pd.to_numeric(df["HSSEX"], errors="coerce") == 1).astype(float)})
    for key in ("wbc", "rbc", "hemoglobin", "hematocrit", "mcv", "mch", "rdw", "platelets",
                "mpv", "calcium", "glucose", "bun", "bilirubin", "creatinine", "ast", "alt",
                "alkaline_phosphatase", "protein_total", "albumin"):
        out[key] = clean(df[key])
    # These carry the implied decimals, so each one is checked.
    print("  checking every parsed column against a plausible adult range:")
    out["bmi"] = scaled(df["bmi_raw"], "bmi")
    out["waist"] = scaled(df["waist_raw"], "waist")
    for key in ("hba1c", "cholesterol_total", "hdl", "systolic_bp", "diastolic_bp",
                "urine_albumin"):
        out[key] = scaled(df[key], key)

    ever = ((pd.to_numeric(df["other_cancer"], errors="coerce") == 1)
            | (pd.to_numeric(df["skin_cancer"], errors="coerce") == 1))
    age_dx = clean(df["age_at_dx"])
    years_since = out["age"] - age_dx
    recent = ever & years_since.notna() & years_since.between(0, RECENT_YEARS)
    never = ~ever
    out["recent_cancer"] = pd.Series(float("nan"), index=out.index)
    out.loc[recent, "recent_cancer"] = 1.0
    out.loc[never & ~recent, "recent_cancer"] = 0.0

    need = ["age", "gender", "wbc", "hemoglobin", "platelets", "glucose", "calcium",
            "albumin", "ast", "alt", "recent_cancer"]
    out = out.dropna(subset=need)
    out.to_csv(OUT, index=False)
    n, pos = len(out), int(out.recent_cancer.sum())
    print(f"\nwrote {OUT}")
    print(f"  {n:,} adults, {pos} diagnosed within {RECENT_YEARS} years ({pos / n:.2%})")
    print("  survivors from longer ago are excluded, not counted as healthy")
    for key in ("hba1c", "cholesterol_total", "hdl", "urine_albumin", "systolic_bp", "waist"):
        print(f"  {key:<20} on {int(out[key].notna().sum()):>6,} ({out[key].notna().mean():.0%})")


if __name__ == "__main__":
    main()
