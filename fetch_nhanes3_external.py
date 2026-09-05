"""
An external test cohort for the prospective panel: NHANES III, 1988-1994.

Why this is worth the trouble
----------------------------
The prospective cancer-mortality panel is trained on NHANES 1999-2014. Repeated
cross-validation says whether it is stable inside that survey. It cannot say
whether it would work on people measured somewhere else, by different staff, on
different analysers, in a different decade. That question is what decides whether
a model is a finding or an artefact of one dataset, and for most of the panels in
this project it has never been asked.

NHANES III is the same survey design run five to twenty years earlier, and NCHS
links it to the National Death Index too. So it gives a genuine external test:

    train    NHANES 1999-2014, 33,834 adults
    test     NHANES III 1988-1994, never seen during fitting

Different decade, different laboratory methods, different field staff, and a
population with a different smoking prevalence and a different age structure.
If the panel transfers to that, it is reading physiology. If it collapses, it was
reading the 1999-2014 survey.

Why the feature set is smaller here
-----------------------------------
Deliberately. NHANES III keeps BMI in a 190 MB examination file and smoking and
alcohol in a 65 MB household file, while everything this test needs — age, sex,
the blood count and the chemistry panel — is in the laboratory file alone.

Rather than download 250 MB to add three lifestyle variables, the comparison is
run on demographics plus bloodwork on BOTH sides. That is the honest way to do
it anyway: an external test has to use identical features, and restricting the
training side to match is better than imputing a median for every test subject.

It also isolates the question that matters. The claim being tested is that
routine blood work carries signal, not that asking about smoking does.

Layout
------
NHANES III ships as fixed-width ASCII with a SAS input statement rather than as
XPT. Column positions below are read from that statement, converted from SAS
one-based inclusive to Python zero-based half-open.

Run:  python fetch_nhanes3_external.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = "data"
LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat"
MORT = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
        "linked_mortality/NHANES_III_MORT_2019_PUBLIC.dat")

HORIZON_MONTHS = 60
CANCER = 2

# name -> (SAS start, SAS end), one-based inclusive, from lab.sas
LAB_SPEC = {
    "SEQN": (1, 5),
    "HSSEX": (15, 15),
    "HSAGEIR": (16, 17),
    # Complete blood count
    "wbc": (1273, 1277),          # WCP, thousands per uL
    "rbc": (1312, 1315),          # RCP, millions per uL
    "hemoglobin": (1320, 1324),   # HGP, g/dL
    "hematocrit": (1330, 1334),   # HTP, %
    "mcv": (1340, 1344),          # MVPSI, fL
    "mch": (1345, 1349),          # MCPSI, pg
    "rdw": (1360, 1364),          # RWP, %
    "platelets": (1371, 1375),    # PLP
    "mpv": (1386, 1390),          # PVPSI, fL
    # Chemistry, all in the conventional units this application uses
    "calcium": (1731, 1734),              # SCP, mg/dL
    "glucose": (1758, 1760),              # SGP, mg/dL
    "bun": (1766, 1768),                  # BUP, mg/dL
    "bilirubin": (1774, 1777),            # TBP, mg/dL
    "creatinine": (1784, 1787),           # CEP, mg/dL
    "ast": (1821, 1823),                  # ASPSI, U/L
    "alt": (1824, 1826),                  # ATPSI, U/L
    "ggt": (1827, 1830),                  # GGPSI, U/L
    "alkaline_phosphatase": (1835, 1838),  # APPSI, U/L
    "protein_total": (1839, 1842),        # TPP, g/dL
    "albumin": (1846, 1848),              # AMP, g/dL
}

MORT_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
MORT_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES",
              "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]

# NHANES III codes refusals and unknowns as runs of 8s and 9s in the field's own
# width, so a blank is not the only kind of missing. Anything at or above these
# is dropped rather than treated as a value.
def _blank_high(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    # 88888/99999 style sentinels sit far outside any plausible analyte range.
    return s.mask(s >= 88888).mask(s.isin([8888, 9999, 888, 999, 88, 99]))


_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=900, context=_ctx).read()


def main():
    print("NHANES III (1988-1994), external test cohort for the prospective panel\n")

    print("  downloading the laboratory file (56 MB) ...", flush=True)
    raw = get(LAB)
    names = list(LAB_SPEC)
    colspecs = [(a - 1, b) for a, b in LAB_SPEC.values()]
    lab = pd.read_fwf(io.BytesIO(raw), colspecs=colspecs, names=names)
    print(f"    {len(lab):,} records", flush=True)

    print("  downloading the linked mortality file ...", flush=True)
    mort = pd.read_fwf(io.BytesIO(get(MORT)), colspecs=MORT_COLSPECS, names=MORT_NAMES)
    for c in ("ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM"):
        mort[c] = pd.to_numeric(mort[c], errors="coerce")
    print(f"    {len(mort):,} records, {int((mort.UCOD_LEADING == CANCER).sum())} "
          f"cancer deaths", flush=True)

    df = lab.merge(mort, on="SEQN")
    df = df[(pd.to_numeric(df["HSAGEIR"], errors="coerce") >= 20) & (df["ELIGSTAT"] == 1)]

    months = df["PERMTH_EXM"]
    cancer_death = (df["MORTSTAT"] == 1) & (df["UCOD_LEADING"] == CANCER)
    positive = cancer_death & (months <= HORIZON_MONTHS)
    keep = positive | (months >= HORIZON_MONTHS)

    out = pd.DataFrame({
        "age": pd.to_numeric(df["HSAGEIR"], errors="coerce"),
        # NHANES III codes 1 male, 2 female. This application uses 0 female,
        # 1 male, and getting it backwards would invert every sex effect.
        "gender": (pd.to_numeric(df["HSSEX"], errors="coerce") == 1).astype(int),
        **{k: _blank_high(df[k]) for k in LAB_SPEC
           if k not in ("SEQN", "HSSEX", "HSAGEIR")},
        "followup_months": months,
        "cancer_death": positive.astype(int),
    })
    out = out[keep.values]
    out = out.dropna(subset=["age", "gender", "albumin", "ast", "alt", "cancer_death"])

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "nhanes3_cancer_mortality.csv")
    out.to_csv(path, index=False)

    print(f"\n  n={len(out):,}  cancer deaths within {HORIZON_MONTHS} months="
          f"{int(out.cancer_death.sum()):,} ({out.cancer_death.mean():.2%})")
    print(f"  median follow-up {out.followup_months.median():.0f} months")
    print(f"  ->  {path}")
    print("\n  measured five to twenty years before the training cohort, on "
          "different analysers")

    # A units check, because a silent unit mismatch is the failure mode that
    # makes an external cohort look like a transfer failure when it is really a
    # conversion bug. This project has been caught by that once already.
    ref = {"hemoglobin": (10, 18), "albumin": (3, 5.5), "alt": (5, 60),
           "calcium": (8, 11), "creatinine": (0.4, 1.6), "glucose": (60, 140)}
    print("\n  units sanity check, medians against expected adult ranges:")
    for k, (lo, hi) in ref.items():
        med = out[k].median()
        ok = "ok" if lo <= med <= hi else "SUSPECT"
        print(f"    {k:<22} median {med:>7.2f}   expected {lo} to {hi}   {ok}")


if __name__ == "__main__":
    main()
