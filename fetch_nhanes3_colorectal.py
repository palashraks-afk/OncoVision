"""
An external test cohort for the bowel panel, from NHANES III.

Why this one matters most
-------------------------
The bowel panel is one of only two panels in this project that screen for a
named cancer from a routine lab report alone. It has never been tested outside
the survey it was fitted on, and the prospective mortality analysis has just
shown what that omission can hide: a gain of +0.013 inside NHANES 1999-2014
became -0.013 on a cohort from a different decade. Every internal validation
strategy used here passed that panel.

So the bowel panel gets the same treatment.

NHANES III asked, of anyone reporting a cancer, both where it was located
(HAC3OS) and how old they were when first told (HAC3OR). With age at exam that
gives years since diagnosis, which is exactly what the training cohort's
eight-year window is built from. Same question, same window, different decade.

    train    NHANES 2005-2014, 23,794 adults, 96 colorectal cancers
    test     NHANES III 1988-1994, colorectal coded as site 04

Cohort construction, matched to the training cohort deliberately
----------------------------------------------------------------
    positive   colorectal cancer first told within WINDOW_YEARS of the exam
    negative   never told they had any cancer, skin included
    EXCLUDED   told longer ago than the window, and anyone with a cancer at
               another site

That last exclusion is the same rule the training cohort uses. A survivor
diagnosed twenty years ago has bloodwork reflecting treatment and time rather
than detection, and counting them as a control is what made an earlier version
of the general panel mostly a proxy for age.

Site 15 is lung, with 28 cases. That is too few to validate the lung panel
against on its own, and it is emitted anyway so the number is on record rather
than quietly dropped.

Run:  python fetch_nhanes3_colorectal.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat"
ADULT = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"

WINDOW_YEARS = 8          # matches fetch_nhanes_colorectal.py
COLORECTAL, LUNG = 4, 15  # HAC3OS site codes, from the NHANES III codebook

# Positions are one-based inclusive as written in the SAS input statements.
LAB_SPEC = {
    "SEQN": (1, 5),
    "HSSEX": (15, 15),
    "HSAGEIR": (16, 17),
    "wbc": (1273, 1277), "rbc": (1312, 1315), "hemoglobin": (1320, 1324),
    "hematocrit": (1330, 1334), "mcv": (1340, 1344), "mch": (1345, 1349),
    "rdw": (1360, 1364), "platelets": (1371, 1375), "mpv": (1386, 1390),
    "calcium": (1731, 1734), "glucose": (1758, 1760), "bun": (1766, 1768),
    "bilirubin": (1774, 1777), "creatinine": (1784, 1787),
    "ast": (1821, 1823), "alt": (1824, 1826), "ggt": (1827, 1830),
    "alkaline_phosphatase": (1835, 1838), "protein_total": (1839, 1842),
    "albumin": (1846, 1848),
}

ADULT_SPEC = {
    "SEQN": (1, 5),
    "age_exam": (18, 19),
    "skin_cancer": (1478, 1478),   # HAC1N, 1 yes 2 no
    "other_cancer": (1479, 1479),  # HAC1O, 1 yes 2 no
    "age_at_dx": (1524, 1526),     # HAC3OR, 004-089 valid, 090 is 90+
    "site": (1527, 1528),          # HAC3OS
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=1200, context=_ctx).read()


def read_fixed(raw: bytes, spec: dict) -> pd.DataFrame:
    return pd.read_fwf(io.BytesIO(raw),
                       colspecs=[(a - 1, b) for a, b in spec.values()],
                       names=list(spec))


def clean(series: pd.Series) -> pd.Series:
    """Drop the 8s-and-9s sentinels NHANES III uses for refused and unknown."""
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s >= 88888).mask(s.isin([8888, 9999, 888, 999, 88, 99]))


def main():
    print("NHANES III (1988-1994), external test cohort for the bowel panel\n")

    print("  downloading the laboratory file (56 MB) ...", flush=True)
    lab = read_fixed(get(LAB), LAB_SPEC)
    print(f"    {len(lab):,} records", flush=True)

    print("  downloading the household adult file (65 MB) ...", flush=True)
    adult = read_fixed(get(ADULT), ADULT_SPEC)
    print(f"    {len(adult):,} records", flush=True)

    df = lab.merge(adult, on="SEQN")
    age = pd.to_numeric(df["HSAGEIR"], errors="coerce")
    df = df[age >= 20]

    site = clean(df["site"])
    age_dx = clean(df["age_at_dx"]).mask(lambda s: s > 90)
    years_since = pd.to_numeric(df["HSAGEIR"], errors="coerce") - age_dx

    ever_any = (pd.to_numeric(df["other_cancer"], errors="coerce") == 1) | \
               (pd.to_numeric(df["skin_cancer"], errors="coerce") == 1)

    for name, code, out_name in (("colorectal", COLORECTAL, "nhanes3_colorectal.csv"),
                                 ("lung", LUNG, "nhanes3_lung.csv")):
        recent = (site == code) & years_since.notna() & (years_since <= WINDOW_YEARS)
        never = ~ever_any
        keep = recent | never

        out = pd.DataFrame({
            "age": pd.to_numeric(df["HSAGEIR"], errors="coerce"),
            # NHANES III codes 1 male, 2 female; this application uses 0 female.
            "gender": (pd.to_numeric(df["HSSEX"], errors="coerce") == 1).astype(int),
            **{k: clean(df[k]) for k in LAB_SPEC
               if k not in ("SEQN", "HSSEX", "HSAGEIR")},
            "years_since_diagnosis": years_since,
            f"{name}_cancer": recent.astype(int),
        })
        out = out[keep.values].dropna(
            subset=["age", "gender", "hemoglobin", "albumin", "alt", f"{name}_cancer"])

        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, out_name)
        out.to_csv(path, index=False)
        n_pos = int(out[f"{name}_cancer"].sum())
        print(f"\n  {name}: n={len(out):,}  cases within {WINDOW_YEARS} years="
              f"{n_pos}  ({out[f'{name}_cancer'].mean():.2%})  ->  {path}")
        if n_pos < 30:
            print(f"    only {n_pos} cases, which is too few to validate a panel "
                  f"against on its own. Emitted so the number is on record.")

    print("\n  survivors diagnosed longer ago than the window are excluded rather "
          "than counted as controls, matching the training cohort's rule")


if __name__ == "__main__":
    main()
