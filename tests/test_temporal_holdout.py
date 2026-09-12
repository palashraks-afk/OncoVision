"""
The temporal holdout has to actually hold.

A withheld cycle is only evidence while it is genuinely withheld, and the way
that breaks is silent: someone rebuilds a cohort file without the `cycle`
column, or renames a cycle, and prepare() quietly trains on everything. Every
downstream number still looks fine. The external validation becomes a
restatement of the training set and nobody finds out.

So the holdout is asserted rather than trusted, from both ends: the training
matrix must not contain the withheld rows, and the withheld rows must still
exist to be tested on.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_models as tm  # noqa: E402

PANELS = sorted(tm.TEMPORAL_HOLDOUT)


@pytest.mark.parametrize("name", PANELS)
def test_withheld_cycle_is_absent_from_training(name):
    cfg = next(c for c in tm.DATASETS if c["name"] == name)
    df = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    train_df, held_df = tm.split_temporal(df, name)

    held = tm.TEMPORAL_HOLDOUT[name]
    assert held not in set(train_df["cycle"].astype(str)), (
        f"{name}: the {held} cycle is still in the training rows")
    assert set(held_df["cycle"].astype(str)) == {held}
    assert len(train_df) + len(held_df) == len(df), "rows went missing in the split"
    assert len(held_df) > 0, f"{name}: nothing was withheld"


@pytest.mark.parametrize("name", PANELS)
def test_shipped_model_was_trained_without_the_withheld_rows(name):
    """The bundle on disk, not just the splitting function.

    prepare() could be correct while the shipped model predates it, which is
    exactly the state this repository was in before the holdout existed.
    """
    joblib = pytest.importorskip("joblib")
    path = f"backend/models/model_{name}.joblib"
    if not os.path.isfile(path):
        pytest.skip("models not built")
    bundle = joblib.load(path)

    cfg = next(c for c in tm.DATASETS if c["name"] == name)
    df = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    _, held_df = tm.split_temporal(df, name)

    n_trained = bundle["metrics"]["n_samples"]
    assert n_trained <= len(df) - len(held_df), (
        f"{name}: the shipped model was fitted on {n_trained} rows, more than "
        f"the {len(df) - len(held_df)} that remain once {tm.TEMPORAL_HOLDOUT[name]} "
        f"is withheld, so it saw the test set")


def test_a_missing_cycle_column_raises_instead_of_passing_everything_through():
    """The failure this is all guarding against."""
    name = PANELS[0]
    df = pd.DataFrame({"age": [50, 60], "gender": [1, 0]})
    with pytest.raises(ValueError, match="no 'cycle' column"):
        tm.split_temporal(df, name)


def test_an_unknown_cycle_name_raises_instead_of_withholding_nothing():
    name = PANELS[0]
    df = pd.DataFrame({"cycle": ["1066-1067"] * 3, "age": [50, 60, 70]})
    with pytest.raises(ValueError, match="not present in the cohort"):
        tm.split_temporal(df, name)


def test_panels_without_a_holdout_are_passed_through_untouched():
    df = pd.DataFrame({"cycle": ["2005-2006"] * 3, "age": [50, 60, 70]})
    train_df, held_df = tm.split_temporal(df, "a-panel-with-no-holdout")
    assert held_df is None
    assert len(train_df) == 3
