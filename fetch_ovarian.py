"""
Ovarian cancer against benign ovarian tumour, from a routine lab panel.

Why this cohort
---------------
Every panel added to this project has to answer a question the application can
actually be handed. This one does, and it is a harder question than most public
cancer datasets pose.

The controls are not healthy volunteers. They are 178 women who presented with
an ovarian mass that turned out to be benign, against 171 whose mass turned out
to be cancer. All 349 went to surgery at the Third Affiliated Hospital of
Soochow University between July 2011 and July 2018, so every label is
histology from a resected specimen rather than a questionnaire answer.

That matters because healthy-versus-cancer is an easy problem and a clinically
useless one. Nobody needs a model to separate a woman with a large ovarian
mass from a woman without one. The real decision is whether a mass that has
already been found is malignant, which determines whether she is referred to a
gynaecologic oncologist. That is the decision this cohort encodes, and it is
the same decision the ROMA and RMI indices are used for in practice.

Source
------
Mendeley Data, doi 10.17632/th7fztbrv9, "Supplementary data 1.xlsx", from
Lu et al., "Using machine learning to predict ovarian cancer",
International Journal of Medical Informatics, 2020.

Two things have to be cleaned
-----------------------------
CENSORED ASSAY VALUES. CA125 and HE4 saturate, so the sheet contains strings
like ">5000.00" rather than numbers. Those are pushed to the assay ceiling
rather than dropped, because a saturated CA125 is the single most informative
value in the row and discarding it would throw away the strongest cases.

UNITS. This is a Chinese laboratory reporting in SI, and the rest of this
project is in conventional US units because that is what the PDF parser reads
off American lab reports. Albumin at 42 is 4.2 g/dL, not a critical value.
Mixing the two would poison the model and every conversion below is applied for
that reason. This is the same failure that made the German liver transfer score
below chance, so it is done explicitly and stated here.

Run:  python fetch_ovarian.py
"""

import io
import json
import os
import ssl
import urllib.request

import pandas as pd

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "ovarian_soochow.csv")

DOI = "th7fztbrv9"
VERSION = 11
LISTING = ("https://data.mendeley.com/public-api/datasets/"
           f"{DOI}/files?folder_id=root&version={VERSION}")
SHEET = "All Raw Data"
WANTED = "Supplementary data 1.xlsx"

# SI to conventional. A value of None means the source unit already matches.
#   source column -> (schema key, multiplier)
CONVERT = {
    "Age":       ("age", 1.0),
    "Menopause": ("menopause", 1.0),

    # Chemistry. g/L to g/dL, mmol/L to mg/dL, umol/L to mg/dL.
    "ALB":   ("albumin", 1 / 10),
    "TP":    ("protein_total", 1 / 10),
    "GLU.":  ("glucose", 18.0),
    "Ca":    ("calcium", 4.008),
    "CREA":  ("creatinine", 1 / 88.4),
    "BUN":   ("bun", 2.8),          # mmol/L urea to mg/dL urea nitrogen
    "TBIL":  ("bilirubin", 1 / 17.1),

    # Enzymes are U/L in both systems.
    "ALT": ("alt", 1.0),
    "AST": ("ast", 1.0),
    "ALP": ("alkaline_phosphatase", 1.0),
    "GGT": ("ggt", 1.0),

    # Haematology. g/L to g/dL; fraction to percent. RBC and PLT already match.
    "HGB": ("hemoglobin", 1 / 10),
    "RBC": ("rbc", 1.0),
    "PLT": ("platelets", 1.0),
    "HCT": ("hematocrit", 100.0),
    "MCV": ("mcv", 1.0),
    "MCH": ("mch", 1.0),
    "RDW": ("rdw", 1.0),
    "MPV": ("mpv", 1.0),
    "NEU": ("neutrophil_pct", 1.0),

    # Tumour markers are reported in the same units worldwide.
    "CA125":  ("ca125", 1.0),
    "HE4":    ("he4", 1.0),
    "CEA":    ("cea", 1.0),
    "AFP":    ("alpha_fetoprotein_level", 1.0),
    "CA19-9": ("plasma_ca19_9", 1.0),
}

# CA72-4 is missing for 68.8 percent of the cohort and is left out rather than
# imputed into two thirds of the rows.
DROPPED = {"CA72-4": "missing for 68.8 percent of patients"}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, context=_ctx, timeout=300).read()


def _numeric(col):
    """Strip censoring marks and thousands separators, then coerce.

    ">5000.00" becomes 5000.0. The ceiling is the informative part.
    """
    if col.dtype != object:
        return pd.to_numeric(col, errors="coerce")
    cleaned = col.astype(str).str.replace(r"[<>\t\s,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def fetch_ovarian():
    os.makedirs(DATA_DIR, exist_ok=True)

    listing = json.loads(_get(LISTING))
    entry = next(f for f in listing if f["filename"] == WANTED)
    raw = pd.ExcelFile(io.BytesIO(_get(entry["content_details"]["download_url"])))
    src = raw.parse(SHEET)

    out = pd.DataFrame(index=src.index)
    for col, (key, mult) in CONVERT.items():
        if col not in src.columns:
            raise KeyError(f"{col} missing from {WANTED}; the deposit changed")
        out[key] = _numeric(src[col]) * mult

    # Everyone in this cohort is a woman with an ovarian mass.
    out["gender"] = 0

    # TYPE is 0 for ovarian cancer and 1 for benign ovarian tumour. Verified by
    # direction rather than trusted: median CA125 is 241.5 in TYPE 0 against
    # 22.7 in TYPE 1, and median HE4 is 140.9 against 43.8.
    out["ovarian_cancer"] = (_numeric(src["TYPE"]) == 0).astype(int)

    out = out[out["ovarian_cancer"].notna()]
    out.to_csv(OUT, index=False)

    n, pos = len(out), int(out["ovarian_cancer"].sum())
    print(f"wrote {OUT}")
    print(f"  {n} women, {pos} ovarian cancer, {n - pos} benign ovarian tumour "
          f"({pos / n:.1%} malignant)")
    print(f"  {out.shape[1] - 1} features, converted from SI to conventional units")
    for col, why in DROPPED.items():
        print(f"  dropped {col}: {why}")
    return out


if __name__ == "__main__":
    fetch_ovarian()
