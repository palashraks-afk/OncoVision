"""
PI-CAI: 1,500 men worked up for prostate cancer at three Dutch hospitals.

Why this cohort
---------------
Every case-control panel in this project has the same hole: no external cohort,
so its AUC has only ever been measured on the institution that produced it. This
is the first one that closes for prostate. PI-CAI's public clinical marksheet
carries exactly the inputs the prostate panel reads -- age, PSA, prostate volume,
PSA density and a radiologist's PI-RADS score -- against histopathology, from
Radboud UMC, Ziekenhuis Groep Twente and the Prostate Cancer Nijmegen network.
The shipped panel was fitted on 212 men at one centre in China; these are 1,500
men in the Netherlands, scanned between 2012 and 2021 on Siemens and Philips
machines.

Source: github.com/DIAGNijmegen/picai_labels, clinical_information/marksheet.csv
(PI-CAI Challenge, CC-BY-NC-4.0). Nothing is downloaded but that one file.

Two differences from the training cohort, both carried into the write-up rather
than smoothed over:

  THE CONTROLS ARE NOT ALL BIOPSIED. The training cohort's negatives are men
  whose biopsy came back benign. Here, 468 men have no histopathology at all:
  their MRI was negative, so no biopsy followed and the case is recorded as
  ISUP 0. That makes this population slightly easier -- some negatives were
  never put to the test -- and it is the population a urologist actually sees.

  TWO TARGETS, BOTH REPORTED. The panel's target is adenocarcinoma on biopsy,
  which is ISUP 1 or above. PI-CAI's own target is clinically significant
  cancer, ISUP 2 or above, the thing anyone would actually treat. Both are
  written out and the experiment scores both.

BMI is not in this file. The app's own behaviour when a value is missing is to
fill the training median, so that is what the experiment does, and BMI carried
almost none of this panel anyway.

Run:  python fetch_picai.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

URL = ("https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main/"
       "clinical_information/marksheet.csv")
OUT = os.path.join("data", "prostate_picai.csv")

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def case_pirads(value):
    """The worst lesion's PI-RADS, which is what a report leads with.

    The column holds one score per lesion, comma separated, and the entries are
    strings: "4", "2,3", or "N/A" where no lesion was marked. A man with no
    suspicious lesion is PI-RADS 1, not missing, because the radiologist did
    look and found nothing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    scores = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            scores.append(int(part))
    if not scores:
        return np.nan
    return float(max(scores))


def main():
    os.makedirs("data", exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=_ctx, timeout=120).read()
    d = pd.read_csv(io.BytesIO(raw))

    out = pd.DataFrame({
        "age": pd.to_numeric(d["patient_age"], errors="coerce"),
        "psa": pd.to_numeric(d["psa"], errors="coerce"),
        "prostate_volume": pd.to_numeric(d["prostate_volume"], errors="coerce"),
        "psa_density": pd.to_numeric(d["psad"], errors="coerce"),
        "pi_rads": d["lesion_PIRADS"].map(case_pirads),
        "case_isup": pd.to_numeric(d["case_ISUP"], errors="coerce"),
        "biopsied": d["histopath_type"].notna(),
        "center": d["center"],
    })
    # PSA density is PSA divided by volume. Recorded for two thirds of these men
    # and computable for most of the rest, which is what a clinician would do
    # rather than leave the field empty.
    computed = out["psa"] / out["prostate_volume"]
    out["psa_density"] = out["psa_density"].fillna(computed.replace([np.inf, -np.inf], np.nan))

    out["prostate_cancer"] = (out["case_isup"] >= 1).astype(float)
    out["significant_cancer"] = (out["case_isup"] >= 2).astype(float)
    out = out[out["case_isup"].notna() & out["age"].notna()]
    out.to_csv(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"  {len(out):,} men, {int(out.prostate_cancer.sum())} with cancer "
          f"({out.prostate_cancer.mean():.1%}), {int(out.significant_cancer.sum())} "
          f"clinically significant ({out.significant_cancer.mean():.1%})")
    print(f"  biopsied {int(out.biopsied.sum())}, MRI-negative and not biopsied "
          f"{int((~out.biopsied).sum())}")
    for col in ("psa", "prostate_volume", "psa_density", "pi_rads"):
        print(f"  {col:<16} present on {out[col].notna().sum():>5} "
              f"({out[col].notna().mean():.0%})")
    print("  by centre: " + ", ".join(f"{k} {v}" for k, v in out.center.value_counts().items()))


if __name__ == "__main__":
    main()
