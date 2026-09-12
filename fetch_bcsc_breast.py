"""
The Breast Cancer Surveillance Consortium risk estimation cohort.

Why this cohort exists here
---------------------------
The breast panel that ships reads thirty nuclear morphology measurements off a
fine needle aspirate and scores 0.997. That number is real and it answers a
question almost nobody asks: the patient already has a biopsy, and somebody has
already decided it was worth taking. It is an interpretation panel and it is
labelled as one.

What the project did not have was a breast panel for the screening question --
a woman who has had a mammogram and wants to know what the report plus what she
knows about herself actually implies. The Wisconsin cohort cannot answer that
at any sample size: 569 patients, 37% of them malignant, cases and matched
controls. A screening population is the thing it lacks by construction.

BCSC is that population.

    2,392,998 screening mammograms
    11,638 breast cancers within one year of the mammogram  (0.49%)
    a prevalence that is the real one, not an enrichment
    a training/validation split specified by the people who built it

The outcome is prospective in the sense that matters: the risk factors are
recorded at the index mammogram and the cancer is diagnosed in the year after
it. That is the direction a screening tool has to work in, and it is the
direction the Wisconsin cohort cannot supply.

What this panel will and will not do
------------------------------------
It will score around 0.63, against 0.60 for age alone. That is a modest gain
and it is the honest one: published BCSC and Gail-type models land between 0.58
and 0.66, and anything here that scored 0.9 would be measuring something other
than screening. A panel that adds +0.03 over age on two million mammograms is
worth more to this project than one that adds 0.4 over nothing on 569.

Race is read and never used as a feature
----------------------------------------
The cohort records race and Hispanic ethnicity. They are kept in the file so
that subgroup accuracy can be measured, and they are excluded from the feature
list, which is the same rule the eGFR precedent set for every other panel here.

The unknown code is 9, and it is not a level
--------------------------------------------
Every categorical in this file uses 9 for unknown, and 27% of mammograms have
unknown density. Read as a number, 9 sorts above 4 and the model would learn
that the least known women have the densest breasts. Every 9 is mapped to
missing here, once, at the source.

Citation required by the data provider
--------------------------------------
"Data collection and sharing was supported by the National Cancer
Institute-funded Breast Cancer Surveillance Consortium (HHSN261201100031C).
You can learn more about the BCSC at: http://www.bcsc-research.org/."

Run:  python fetch_bcsc_breast.py
"""

import io
import ssl
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = "https://www.bcsc-research.org/index.php/download_file/view/34/345"
MEMBER = "risk.txt"

TRAIN_OUT = "data/bcsc_breast_train.csv"
VALID_OUT = "data/bcsc_breast_validation.csv"

# Column order is fixed by the BCSC documentation for this file; it carries no
# header of its own.
COLUMNS = ["menopaus", "agegrp", "density", "race", "hispanic", "bmi",
           "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt",
           "invasive", "cancer", "training", "count"]

# Everything the application could reasonably ask a woman who is holding a
# mammogram report. Race and Hispanic ethnicity are deliberately absent.
FEATURES = ["agegrp", "density", "bmi", "agefirst", "nrelbc", "brstproc",
            "lastmamm", "menopaus", "surgmeno", "hrt"]

# The training rows are a frequency table and the pipeline downstream expects
# individual patients, so they are expanded by sampling. The whole table is
# 1.79M mammograms, which is more rows than ten categorical features can carry
# information for -- there are only a few hundred thousand distinct
# combinations, and the table already holds every one of them.
#
# The validation rows are NOT sampled. They stay aggregated and are scored with
# their counts as weights, so the number that judges this panel uses all
# 597,859 mammograms and loses no precision.
TRAIN_SAMPLE = 400_000
SEED = 0

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def main():
    print("BCSC risk estimation cohort\n")
    print("  downloading ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=600, context=_ctx).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(io.BytesIO(z.read(MEMBER)), sep=r"\s+",
                     header=None, names=COLUMNS)
    print(f"    {len(df):,} covariate combinations, "
          f"{int(df['count'].sum()):,} mammograms")

    # Checked against the figures the BCSC publishes for this file. A silent
    # change of column order would otherwise produce a cohort that parses
    # cleanly and means something else entirely.
    if int(df["count"].sum()) != 2_392_998:
        raise ValueError(
            f"expected 2,392,998 mammograms, parsed {int(df['count'].sum()):,}; "
            f"the file layout has changed and the column order must be rechecked")

    # 9 is unknown in every categorical here, including density, where it
    # covers 27% of mammograms. As a number it outranks 4.
    for col in FEATURES:
        df[col] = df[col].replace(9, np.nan)

    cancers = int(df.loc[df["cancer"] == 1, "count"].sum())
    print(f"    {cancers:,} cancers within one year "
          f"({cancers / df['count'].sum():.2%})")

    keep = FEATURES + ["race", "hispanic", "cancer", "invasive", "count"]
    train_tbl = df[df["training"] == 1][keep]
    valid_tbl = df[df["training"] == 0][keep]

    # A multinomial draw from the training population, which is a simple random
    # sample of real mammograms rather than a reweighting of the table.
    rng = np.random.default_rng(SEED)
    p = train_tbl["count"].to_numpy(dtype=float)
    idx = rng.choice(len(train_tbl), size=TRAIN_SAMPLE, p=p / p.sum())
    train = train_tbl.iloc[idx].drop(columns=["count"]).reset_index(drop=True)

    train.to_csv(TRAIN_OUT, index=False)
    valid_tbl.to_csv(VALID_OUT, index=False)

    print(f"\n  train       {len(train):,} mammograms sampled from "
          f"{int(train_tbl['count'].sum()):,}, "
          f"{int(train['cancer'].sum()):,} cancers "
          f"({train['cancer'].mean():.2%})  ->  {TRAIN_OUT}")
    print(f"  validation  {int(valid_tbl['count'].sum()):,} mammograms in "
          f"{len(valid_tbl):,} rows, "
          f"{int(valid_tbl.loc[valid_tbl['cancer'] == 1, 'count'].sum()):,} "
          f"cancers  ->  {VALID_OUT}")
    print("\n  the validation split is the one BCSC specifies, kept aggregated "
          "and scored with counts as weights")
    print("  race and Hispanic ethnicity are carried for subgroup measurement "
          "and are not features")


if __name__ == "__main__":
    main()
