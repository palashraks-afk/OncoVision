"""
Cervical pre-cancer, from risk history, in a colposcopy referral clinic.

Why this cohort
---------------
858 women assessed at Hospital Universitario de Caracas, 55 of them with a
positive cervical biopsy. The outcome is histology, not a questionnaire answer
and not a screening test result, which makes it a harder and more useful label
than most public cohorts carry.

What this panel is and is not
-----------------------------
It is not a population screen and the numbers say why. Projected onto SEER
population incidence it would flag roughly 7,383 women per true case, which
would be indefensible to ship. Projected onto the 6.4 percent prevalence in the
referred group it flags 9.4 per case at 96.2 percent NPV, which is the kind of
number a triage step is actually judged on.

So this panel answers "among women already being assessed, who should be
prioritised for colposcopy", in the same way the ovarian panel answers "among
women already found to have a mass, whose is malignant".

Leakage
-------
The source carries Dx, Dx:Cancer, Dx:CIN and Dx:HPV, which record a diagnosis
the patient already has. Training on those would be predicting a biopsy result
from the fact that the disease is already known, and they are dropped. This is
the whole reason the panel scores 0.725 rather than the near-perfect numbers
that get reported on this dataset elsewhere.

Honest limits, carried onto the panel itself
--------------------------------------------
55 positives is below the roughly 96 events needed to pin an estimate down, so
the interval is wide: 0.539 to 0.888. It excludes chance and it beats age alone
by a wide margin, 0.725 against 0.458, which is why it ships. The width is
stated on the panel rather than hidden.

Cutting the input list was tried and failed. Eight fields scored 0.665 with an
interval of 0.458 to 0.849 that contains chance, and six scored 0.552. All
fifteen are needed, so all fifteen are asked for.

Source: UCI Machine Learning Repository id 383, Fernandes, Cardoso and Fernandes.

Run:  python fetch_cervical.py
"""

import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "cervical_caracas.csv")

# source column -> schema key
COLUMNS = {
    "Age": "age",
    "Number of sexual partners": "sexual_partners",
    "First sexual intercourse": "first_intercourse_age",
    "Num of pregnancies": "pregnancies",
    "Smokes": "smokes",
    "Smokes (years)": "smoking_years",
    "Smokes (packs/year)": "smoking_packyears",
    "Hormonal Contraceptives": "hormonal_contraceptives",
    "Hormonal Contraceptives (years)": "hormonal_contraceptives_years",
    "IUD": "iud",
    "IUD (years)": "iud_years",
    "STDs": "stds",
    "STDs (number)": "stds_number",
    "STDs:HPV": "stds_hpv",
    "STDs: Number of diagnosis": "stds_diagnoses",
}

# Prior diagnosis. Excluded so the panel predicts a biopsy rather than recalls
# a chart.
LEAKAGE_PREFIX = "Dx"


def fetch_cervical():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = fetch_ucirepo(id=383).data.original

    leaked = [c for c in raw.columns if c.startswith(LEAKAGE_PREFIX)]

    out = pd.DataFrame(index=raw.index)
    for src, key in COLUMNS.items():
        if src not in raw.columns:
            raise KeyError(f"{src} missing from UCI 383; the deposit changed")
        out[key] = pd.to_numeric(raw[src], errors="coerce")

    # Everyone in this cohort is a woman.
    out["gender"] = 0

    out["cervical_biopsy_positive"] = pd.to_numeric(raw["Biopsy"], errors="coerce")
    out = out[out["cervical_biopsy_positive"].notna()]
    out["cervical_biopsy_positive"] = out["cervical_biopsy_positive"].astype(int)

    out.to_csv(OUT, index=False)

    n = len(out)
    pos = int(out["cervical_biopsy_positive"].sum())
    print(f"wrote {OUT}")
    print(f"  {n} women, {pos} biopsy-positive ({pos / n:.1%})")
    print(f"  {len(COLUMNS)} risk-history features")
    print(f"  dropped as leakage (prior diagnosis): {leaked}")
    return out


if __name__ == "__main__":
    fetch_cervical()
