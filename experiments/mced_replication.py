"""
Replication of the Medicare multi-cancer test law analysis in newer data: NHIS 2010-2014, then 2005-2014 pooled.
Rules and bars: docs/MCED_LAW_PREREG.md, "Replication in newer data" (committed before the data were downloaded).
Reuses experiments/mced_law.py unchanged; only the test years differ. The ranking model stays fitted on 1997-2002.

Run:  python experiments/mced_replication.py     needs data/nhis_cancer_risk.csv.gz and data/nhis_2010_2014.csv.gz
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import mced_law
from cancer_age import CANCER

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_years(test_years):
    """Same filters as mced_law.load(), over the fitting years (1997-2002) plus the chosen test years."""
    d = pd.concat([pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz")), pd.read_csv(os.path.join(ROOT, "data", "nhis_2010_2014.csv.gz"))], ignore_index=True)
    d = d[d["year"].isin(list(range(1997, 2003)) + list(test_years))]
    d = d[(d["age"] >= 35) & (d["age"] <= 84) & (d["prior_cancer"] == 0)].copy()
    d = d[(d["months"] >= 60) | (d["mortstat"] == 1)]
    d["y"] = ((d["mortstat"] == 1) & (d["ucod"] == CANCER) & (d["months"] <= 60)).astype(int)
    d = d.dropna(subset=["smoke_status", "bmi"]).reset_index(drop=True)
    d["bmi"] = d["bmi"].clip(15, 55)
    d["quit_years"] = d["quit_years"].fillna(0).clip(0, 50)
    d["cigs_per_day"] = d["cigs_per_day"].fillna(0).clip(0, 60)
    return d


def main():
    for tag, years in (("2010_2014", range(2010, 2015)), ("2005_2014", range(2005, 2015))):
        print(f"\n=================== test years {tag.replace('_', '-')} ===================")
        mced_law.load = lambda years=years: load_years(years)
        mced_law.OUT = os.path.join(ROOT, "experiments", f"mced_law_result_{tag}.json")
        mced_law.RNG = np.random.default_rng(20261127)
        mced_law.main()


if __name__ == "__main__":
    main()
