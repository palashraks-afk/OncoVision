"""Checks for experiments/red_flag_labs.py: the design-based estimator and the guideline definitions. No downloads."""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mod():
    spec = importlib.util.spec_from_file_location("red_flag_labs", os.path.join(ROOT, "experiments", "red_flag_labs.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ratio_estimate_matches_a_hand_calculation():
    rf = mod()
    # two strata, two PSUs each, one person per PSU
    y = np.array([1.0, 0.0, 1.0, 1.0])
    x = np.ones(4)
    w = np.array([1.0, 1.0, 2.0, 2.0])
    h = np.array([1, 1, 2, 2])
    c = np.array([1, 2, 1, 2])
    r, ci = rf.ratio_est(y, x, w, h, c)
    assert r == pytest.approx(5 / 6)
    # linearised z = w (y - r x) / sum(w x): stratum 1 has z = 1/36 and -5/36, stratum 2 has 2/36 and 2/36
    z1 = np.array([1 / 36, -5 / 36])
    var = 2 / 1 * ((z1 - z1.mean()) ** 2).sum()
    se = np.sqrt(var)
    assert ci[0] == pytest.approx(max(r - 1.96 * se, 0))
    assert ci[1] == pytest.approx(min(r + 1.96 * se, 1))


def test_guideline_definitions():
    rf = mod()
    hgb = np.array([12.9, 12.9, 11.9, 11.9, 13.0, 12.0])
    fer = np.array([44.9, 45.0, 10.0, 80.0, 10.0, 10.0])
    male = np.array([1, 1, 0, 0, 1, 0])
    anaemia, ida, nia = rf.flags(hgb, fer, male)
    assert list(anaemia) == [True, True, True, True, False, False]   # men < 13, women < 12; boundaries not anaemic
    assert list(ida) == [True, False, True, False, False, False]      # ferritin strictly below 45
    assert list(nia) == [False, True, False, True, False, False]


def test_ida_and_non_iron_anaemia_never_overlap():
    rf = mod()
    rng = np.random.default_rng(0)
    hgb, fer, male = rng.uniform(8, 16, 500), rng.uniform(1, 200, 500), rng.integers(0, 2, 500)
    anaemia, ida, nia = rf.flags(hgb, fer, male)
    assert not (ida & nia).any()
    assert ((ida | nia) == anaemia).all()
