"""
The navigator through the HTTP interface, which is what the interface actually calls.

The engine is tested in test_navigator.py. This checks the wiring: that the vocabulary
the form is built from matches the rules, that a request reaches the engine, that
impossible input is refused with a readable message and a 422 rather than a 500, and
that the endpoint never leaks anything a person did not send.

Run:  python -m pytest tests/test_navigator_api.py -q
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import api  # noqa: E402
import navigator  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_vocabulary_lists_every_symptom_a_rule_can_use(client):
    body = client.get("/navigator/vocabulary").json()
    assert body["status"] == "success"
    offered = {s["key"] for g in body["symptom_groups"] for s in g["symptoms"]}
    used = set()
    for r in navigator.load()["rules"]:
        for path in r["paths"]:
            for c in path:
                for k in ("sym", "not_sym"):
                    if k in c:
                        used.add(c[k])
                for k in ("sym_min", "freq_min"):
                    if k in c:
                        used.update(c[k]["of"])
    assert used <= offered, f"rules use symptoms the form cannot ask about: {sorted(used - offered)}"
    assert body["meta"]["review_status"].startswith("NOT YET REVIEWED")


def test_a_prompt_check_pattern_comes_back_as_talk_soon(client):
    r = client.post("/navigator", json={"age": 58, "sex": "male",
                                         "symptoms": [{"key": "haemoptysis"}]}).json()
    assert r["status"] == "success"
    assert r["state"] == "talk_soon"
    assert r["matches"][0]["id"] == "lung_haemoptysis"
    assert "not medical advice" in r["disclaimer"].lower()


def test_labs_and_symptoms_are_read_together(client):
    r = client.post("/navigator", json={"age": 66, "sex": "female",
                                         "labs": {"hemoglobin": 10.5, "ferritin": 6}}).json()
    assert r["state"] == "talk_soon"
    assert any(m["id"] == "colorectal_urgent" for m in r["matches"])
    assert r["lab_findings"]["iron_deficiency_anaemia"] is True


def test_an_empty_request_is_answered_not_crashed(client):
    r = client.post("/navigator", json={})
    assert r.status_code == 200
    assert r.json()["state"] in ("nothing_meets", "need_info")


def test_impossible_values_are_a_422_with_a_message(client):
    for bad in ({"age": 500}, {"labs": {"hemoglobin": 99}},
                {"symptoms": [{"key": "bloating", "times_per_month": 99}]}):
        r = client.post("/navigator", json=bad)
        assert r.status_code == 422, bad
        assert r.json()["message"]


def test_unknown_fields_are_ignored_and_unknown_symptoms_reported(client):
    r = client.post("/navigator", json={"age": 50, "name": "A. Person", "email": "a@b.c",
                                         "symptoms": [{"key": "made_up"}]}).json()
    assert r["ignored_symptoms"] == ["made_up"]
    assert "A. Person" not in str(r) and "a@b.c" not in str(r)
