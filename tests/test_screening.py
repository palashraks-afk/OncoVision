"""
Screening lookup tests. Every expected answer is taken from the USPSTF recommendation text
(read 2026-10-05), including the boundary ages, so a transcription slip fails a test.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
import screening as sc  # noqa: E402


def st(req, item):
    return {i["id"]: i["status"] for i in sc.evaluate(req)["items"]}.get(item)


@pytest.mark.parametrize("age,expected", [(39, "not_yet"), (40, "due"), (74, "due"), (75, "no_recommendation"), (90, "no_recommendation")])
def test_breast_boundaries(age, expected):
    assert st({"age": age, "sex_at_birth": "female"}, "breast") == expected


def test_breast_not_shown_to_men():
    assert st({"age": 55, "sex_at_birth": "male"}, "breast") is None


@pytest.mark.parametrize("age,expected", [(20, "not_yet"), (21, "due"), (29, "due"), (30, "due"), (65, "due"), (66, "not_needed")])
def test_cervical_boundaries(age, expected):
    assert st({"age": age, "sex_at_birth": "female", "has_cervix": True}, "cervical") == expected


def test_cervical_not_needed_after_hysterectomy_and_asks_when_unknown():
    assert st({"age": 40, "sex_at_birth": "female", "has_cervix": False}, "cervical") == "not_needed"
    assert st({"age": 40, "sex_at_birth": "female"}, "cervical") == "need_info"


def test_cervical_age_bands_give_different_advice():
    a = [i for i in sc.evaluate({"age": 25, "sex_at_birth": "female", "has_cervix": True})["items"] if i["id"] == "cervical"][0]["headline"]
    b = [i for i in sc.evaluate({"age": 40, "sex_at_birth": "female", "has_cervix": True})["items"] if i["id"] == "cervical"][0]["headline"]
    assert "every 3 years" in a and a != b


@pytest.mark.parametrize("age,expected", [(44, "not_yet"), (45, "due"), (49, "due"), (50, "due"), (75, "due"), (76, "individual"), (85, "individual"), (86, "not_needed")])
def test_colorectal_boundaries(age, expected):
    assert st({"age": age}, "colorectal") == expected


def test_colorectal_lists_the_tests_when_due_only():
    due = [i for i in sc.evaluate({"age": 55})["items"] if i["id"] == "colorectal"][0]
    assert len(due["options"]) >= 5
    old = [i for i in sc.evaluate({"age": 80})["items"] if i["id"] == "colorectal"][0]
    assert old["options"] == []


def test_lung_eligible_current_smoker():
    assert st({"age": 55, "ever_smoked": True, "smokes_now": True, "pack_years": 20}, "lung") == "due"


def test_lung_pack_year_boundary():
    assert st({"age": 55, "ever_smoked": True, "smokes_now": True, "pack_years": 19.9}, "lung") == "not_needed"


def test_lung_quit_window_boundary():
    base = {"age": 60, "ever_smoked": True, "smokes_now": False, "pack_years": 30}
    assert st({**base, "years_since_quit": 15}, "lung") == "due"
    assert st({**base, "years_since_quit": 15.5}, "lung") == "not_needed"


@pytest.mark.parametrize("age,expected", [(49, "not_yet"), (50, "due"), (80, "due"), (81, "not_needed")])
def test_lung_age_boundaries(age, expected):
    assert st({"age": age, "ever_smoked": True, "smokes_now": True, "pack_years": 40}, "lung") == expected


def test_lung_never_smoker_not_eligible_and_unknown_is_asked():
    assert st({"age": 60, "ever_smoked": False}, "lung") == "not_needed"
    assert st({"age": 60}, "lung") == "need_info"
    out = sc.evaluate({"age": 60, "ever_smoked": True})
    lung = [i for i in out["items"] if i["id"] == "lung"][0]
    assert lung["status"] == "need_info" and lung["needs"]


@pytest.mark.parametrize("age,expected", [(54, "not_yet"), (55, "individual"), (69, "individual"), (70, "not_recommended")])
def test_prostate_boundaries(age, expected):
    assert st({"age": age, "sex_at_birth": "male"}, "prostate") == expected


def test_prostate_is_never_called_due():
    for age in range(40, 100):
        assert st({"age": age, "sex_at_birth": "male"}, "prostate") != "due"


def test_missing_sex_is_asked_not_assumed():
    out = sc.evaluate({"age": 50})
    assert "sex at birth" in out["needs"]


def test_impossible_input_refused():
    for bad in ({"age": -1}, {"age": 200}, {"age": 50, "pack_years": 9999}, {"age": 50, "years_since_quit": -2}, {}):
        with pytest.raises(ValueError):
            sc.evaluate(bad)


def test_every_item_cites_its_source_and_the_scope_is_stated():
    out = sc.evaluate({"age": 55, "sex_at_birth": "female", "has_cervix": True, "ever_smoked": True, "smokes_now": True, "pack_years": 30})
    assert all(i["source"] and i["url"].startswith("https://www.uspreventiveservicestaskforce.org") for i in out["items"])
    assert "relative" in out["family_note"].lower() and "research prototype" in out["disclaimer"].lower()
    assert out["meta"]["review_status"].startswith("NOT YET REVIEWED")


def test_api_endpoint():
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    r = c.post("/screening", json={"age": 52, "sex_at_birth": "female", "has_cervix": True})
    assert r.status_code == 200 and r.json()["status"] == "success"
    assert c.post("/screening", json={"age": 500}).status_code == 422
    assert c.post("/screening", json={}).status_code == 422
