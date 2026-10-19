"""
End-to-end tests of experiments/discovery_screen.py on synthetic data with planted effects.

The screen had never completed a run when it was written, so these exist to prove the pipeline behaves before
real data is touched, and to hold each fix made against docs/DISCOVERY_SCREEN_PREREG.md in place:

  * a hit is never dropped without a recorded reason when the replication years lack a category,
  * the equity screen uses the registered bar (0.01 AUC), the all-adults screen uses 0.005,
  * a group with too few events is reported, not crashed on,
  * replication refuses to run if the frozen hit list changed.

No downloads, no network. The planted item must be found and replicated; pure noise must not be.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    spec = importlib.util.spec_from_file_location("ds_under_test", os.path.join(ROOT, "experiments", "discovery_screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(n_per_year=5500, seed=7):
    """A cohort shaped like load()'s output, with one real cancer-specific item and some noise."""
    rng = np.random.default_rng(seed)
    years = [1997, 1998, 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2008, 2009]
    year = np.repeat(years, n_per_year)
    n = len(year)
    age = rng.uniform(35, 84, n)
    male = rng.integers(0, 2, n).astype(float)
    smoke = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]).astype(float)
    bmi = rng.normal(27, 5, n).clip(15, 55)
    plant = rng.choice([0, 1, 2, 3], n, p=[0.5, 0.25, 0.15, 0.10]).astype(float)
    rare = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]).astype(float)
    rare[:20] = 9                                   # 20 people, in 1997 only: pooled in discovery, absent in replication
    noise = {f"NOISE{i}": rng.choice([0, 1, 2, 3], n).astype(float) for i in range(3)}
    logit = -7.2 + 0.07 * age + 0.3 * male + 0.5 * (smoke == 2) + 1.0 * (plant == 3) + 0.6 * (rare == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    y_other = ((rng.random(n) < 1 / (1 + np.exp(-(-5.5 + 0.06 * age)))) & (y == 0)).astype(int)
    o = pd.DataFrame({"year": year, "age": age, "male": male, "bmi": bmi, "smoke_status": smoke,
                      "cigs_per_day": np.where(smoke == 2, 15.0, 0.0), "quit_years": np.where(smoke == 1, 8.0, 0.0),
                      "y": y, "y_other": y_other, "race4": rng.choice([0, 1, 2, 3], n, p=[0.55, 0.30, 0.10, 0.05])})
    raw = pd.DataFrame({"year": year, "ITEMA": plant, "ITEMR": rare, **noise})
    labels = {yr: {"ITEMA": "synthetic item a", "ITEMR": "synthetic item r",
                   **{f"NOISE{i}": f"synthetic noise {i}" for i in range(3)}} for yr in years}
    return raw, o, labels


@pytest.fixture()
def ds(tmp_path, monkeypatch):
    mod = load_module()
    raw, o, labels = synthetic()
    monkeypatch.setattr(mod, "load", lambda: (raw.copy(), o.copy()))
    monkeypatch.setattr(mod, "load_labels", lambda: labels)
    monkeypatch.setattr(mod, "RES", str(tmp_path))
    monkeypatch.setattr(mod, "N_JOBS", 1)
    monkeypatch.setattr(mod, "MIN_PEOPLE", 500)
    return mod


def test_planted_item_is_found_replicates_and_noise_is_not(ds, tmp_path):
    ds.discover()
    stage = json.load(open(tmp_path / "discovery_screen_stage1.json"))
    assert stage["candidates"] == 5 and stage["screened"] == 5
    hits = pd.read_csv(tmp_path / "discovery_screen_hits.csv")
    assert "ITEMA" in set(hits["item"])
    assert not any(i.startswith("NOISE") for i in hits["item"]), "pure noise passed the discovery screen"
    ds.replicate()
    res = json.load(open(tmp_path / "discovery_screen_result.json"))
    assert res["items"]["ITEMA"]["replicated"] is True
    assert res["items"]["ITEMA"]["specific"] is True
    assert res["stages"]["screened"] == 5 and res["stages"]["discovery_hits"] == len(hits)


def test_a_hit_is_never_dropped_when_replication_lacks_a_category(ds, tmp_path):
    """ITEMR has a code (9) pooled in discovery and absent in replication. It used to vanish with no record."""
    ds.discover()
    ds.replicate()
    res = json.load(open(tmp_path / "discovery_screen_result.json"))
    assert "ITEMR" in res["items"], "a hit disappeared from the report"
    info = res["items"]["ITEMR"]
    assert info["replicated"] is True
    assert info["discovery_df"] == info["replication_df"] + 1, "the empty replication column should be left out of the degrees of freedom"


def test_blocks_layout_is_the_discovery_layout():
    ds = load_module()
    rng = np.random.default_rng(1)
    disc = pd.Series(rng.choice([1.0, 2.0, 3.0], 4000))
    disc.iloc[:20] = 9.0
    repl = pd.Series(rng.choice([1.0, 2.0, 3.0], 4000))
    M0, lv0, ref = ds.blocks(disc)
    Mt, lv1, _ = ds.blocks(repl, ref=ref)
    assert Mt.shape[1] == M0.shape[1] and lv0 == lv1


def test_the_equity_screen_uses_the_registered_bar(ds, tmp_path):
    ds.discover(1)
    ds.replicate(1)
    res = json.load(open(tmp_path / "discovery_screen_result_race1.json"))
    assert res["equity_screen"] is True and res["auc_bar"] == 0.01


def test_the_all_adults_screen_uses_its_own_bar(ds, tmp_path):
    ds.discover()
    ds.replicate()
    res = json.load(open(tmp_path / "discovery_screen_result.json"))
    assert res["equity_screen"] is False and res["auc_bar"] == 0.005


def test_a_group_with_too_few_events_is_reported_not_crashed_on(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MIN_EVENTS", 10 ** 6)
    ds.discover(3)
    stage = json.load(open(tmp_path / "discovery_screen_stage1_race3.json"))
    assert stage["screened"] == 0 and stage["discovery_hits"] == 0
    assert "cancer deaths" in json.dumps(stage["skipped_reasons"])
    skipped = pd.read_csv(tmp_path / "discovery_screen_skipped_race3.csv")
    assert len(skipped) == stage["candidates"]
    ds.replicate(3)                                   # must also be a clean no-op
    assert json.load(open(tmp_path / "discovery_screen_result_race3.json"))["discovery_hits"] == 0


def test_replication_refuses_if_the_frozen_hit_list_changed(ds, tmp_path):
    ds.discover()
    hits = tmp_path / "discovery_screen_hits.csv"
    hits.write_text(hits.read_text() + "\n")
    with pytest.raises(SystemExit):
        ds.replicate()


def test_every_excluded_column_is_listed_with_a_reason(ds, tmp_path):
    ds.discover()
    ex = pd.read_csv(tmp_path / "discovery_screen_excluded.csv")
    assert "year" in set(ex["item"]) and set(ex["reason"]) <= {
        "name", "label", "no label in the SAS programs",
        "label wording differs between years (asked in enough years by name)",
        "asked in fewer than 2 discovery or 2 replication years"}


# ---- checks on the real data; skipped where it has not been downloaded (CI) -------------------------------------------
REAL = os.path.join(ROOT, "data", "audit", "nhis_full", "2009.csv.gz")
needs_data = pytest.mark.skipif(not os.path.exists(REAL), reason="NHIS full questionnaire not downloaded (python fetch_nhis_full.py)")


@pytest.fixture(scope="module")
def real():
    mod = load_module()
    raw, o, tag, cand, counts, excluded = mod.setup(None)
    return mod, raw, o, cand


@needs_data
def test_replication_cohort_matches_the_age_paper(real):
    """2005-2009 race groups must equal docs/cancer_age_paper exactly: adults and cancer deaths in each group."""
    mod, raw, o, cand = real
    t = o[o["year"].isin(mod.REPL)]
    expected = {0: (45815, 1384), 1: (11431, 397), 2: (11379, 194), 3: (4543, 97)}
    for g, (n, c) in expected.items():
        s = t[t["race4"] == g]
        assert (len(s), int(s["y"].sum())) == (n, c), mod.GROUPS[g]


@needs_data
def test_every_race_group_exists_in_every_year(real):
    """Race sits under a different variable name in different years. 1997-2002 once came out with no White or Black adults."""
    mod, raw, o, cand = real
    for y, g in o.groupby("year"):
        for grp in (0, 1, 2, 3):
            assert (g["race4"] == grp).sum() > 100, (y, mod.GROUPS[grp])
        assert (g["race4"] == -1).mean() < 0.01, y


@needs_data
def test_race_and_ethnicity_are_never_screened_as_items(real):
    mod, raw, o, cand = real
    bad = [c for c, (lab, yrs) in cand.items() if any(w in lab for w in ("race", "hispanic", "latino", "ethnic"))]
    assert not bad and not any(c.startswith(("RACE", "HISPAN", "ORIGIN", "MRAC")) for c in cand)


@needs_data
def test_the_test_is_calibrated_on_shuffled_items(real):
    """Shuffled items have no link to cancer. The old solver gave null p-values averaging 0.61 and 14% impossible zeros."""
    mod, raw, o, cand = real
    rng = np.random.default_rng(5)
    ps, zeros = [], 0
    for _ in range(2):
        for c in sorted(cand):
            r = mod.screen_one(c, cand[c][0], cand[c][1], pd.Series(rng.permutation(raw[c].to_numpy())), o, mod.DISC)
            if "skipped" not in r:
                ps.append(r["p"])
                zeros += r["lr"] == 0
    assert 0.40 < np.mean(ps) < 0.60, np.mean(ps)
    assert zeros / len(ps) < 0.03, zeros / len(ps)
