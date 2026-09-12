"""
The two ways the BCSC breast panel can be silently wrong.

Both produce a model that trains, validates and looks reasonable while being
fed nonsense, which is why they are asserted rather than trusted.

1. The unknown code. Every categorical in the BCSC file uses 9 for unknown, and
   27% of mammograms have unknown density. Read as a number, 9 outranks 4 and
   the least-known women become the densest.

2. The bands. BCSC records age and BMI as band indices (1-10, 1-4). The service
   sends real ages and real BMIs. A model trained on the index and handed a BMI
   of 22 reads it as a band five places past the top of the scale -- the same
   class of error as the CRP field once read in the wrong units.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_models as tm  # noqa: E402

TRAIN = os.path.join(tm.DATA_DIR, "bcsc_breast_train.csv")
pytestmark = pytest.mark.skipif(not os.path.isfile(TRAIN),
                                reason="run fetch_bcsc_breast.py first")


@pytest.fixture(scope="module")
def features():
    cfg = next(c for c in tm.DATASETS if c["name"] == "breast_screening")
    return tm.build_features(cfg, pd.read_csv(TRAIN))


def test_unknown_code_never_reaches_the_model(features):
    for col in ("breast_density", "family_history_breast", "prior_breast_biopsy",
                "age_at_first_birth", "last_mammogram_result", "menopause",
                "surgical_menopause", "hormone_therapy"):
        assert not (features[col] == 9).any(), (
            f"{col} still carries the BCSC unknown code 9 as a value")


def test_density_is_on_the_bi_rads_scale(features):
    assert set(features["breast_density"].dropna().unique()) <= {1, 2, 3, 4}


def test_age_is_in_years_not_band_index(features):
    age = features["age"].dropna()
    assert age.min() >= 35 and age.max() <= 85, (
        f"age runs {age.min()} to {age.max()}; band indices run 1 to 10")


def test_bmi_is_in_kg_per_m2_not_band_index(features):
    bmi = features["bmi"].dropna()
    assert bmi.min() >= 18 and bmi.max() <= 40, (
        f"BMI runs {bmi.min()} to {bmi.max()}; band indices run 1 to 4")


def test_race_is_carried_but_not_a_feature(features):
    assert "race" not in features.columns
    assert "hispanic" not in features.columns
    assert "race" in pd.read_csv(TRAIN, nrows=5).columns
