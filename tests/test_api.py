"""
Tests for the behaviour that has actually broken.

Every test here corresponds to a defect that shipped and was found by hand. That
is the selection rule: this file is not trying to cover the API, it is trying to
stop the same things happening twice.

  gating            a male lab report came back carrying an ovarian and a
                    cervical risk score
  silent drops      a panel whose features were all blank vanished from the
                    response entirely, neither scored nor skipped nor mentioned
  empty input       an empty request produced a score built from training
                    medians instead of saying there was no data
  coverage          the cervical panel reported "Raised" off one input in
                    fifteen, looking exactly as confident as a full panel
  dead fields       the service accepted a sexual history that no model read
  bounds            a typo of 5000 for a haemoglobin was scored as a real value
  registry shape    the page renders held-out fields, so the registry has to
                    serve them

Run:  python -m pytest tests/ -q
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import api  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # The context manager matters: without it FastAPI's startup event never
    # fires, the model registry stays empty, and every test passes vacuously
    # against a service that has loaded nothing.
    with TestClient(api.app) as c:
        yield c


FULL_BLOODS = {
    "wbc": 6.5, "rbc": 4.8, "hemoglobin": 14.2, "platelets": 240,
    "glucose": 95, "calcium": 9.5, "bun": 15, "creatinine": 1.0,
    "protein_total": 7.1, "albumin": 4.3, "ast": 24, "alt": 22,
    "bilirubin": 0.7, "alkaline_phosphatase": 82, "ggt": 30,
}


# --------------------------------------------------------------- registry
def test_models_load(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert len(body["models"]) >= 8, "panels went missing from the registry"


def test_registry_serves_held_out_fields(client):
    """
    The methodology tables render auc_ci, n_test, brier and PPV.

    Those live in held_out, not in metrics. When the page spread only metrics,
    four columns rendered "n/a", the PPV column rendered NaN, and the headline
    AUC silently became a cross-validated training number.
    """
    body = client.get("/models").json()
    for name, m in body["models"].items():
        held = m.get("held_out") or {}
        for field in ("auc", "auc_ci", "n_test", "brier",
                      "ppv_at_population_prevalence"):
            assert field in held, f"{name} is missing held_out.{field}"


def test_every_panel_declares_its_kind(client):
    """A panel with no declared type used to default to claiming it screens."""
    body = client.get("/models").json()
    for name, m in body["models"].items():
        kind = (m.get("held_out") or {}).get("panel_kind") or m.get("panel_kind")
        bundle = api.models.get(name, {})
        kind = kind or bundle.get("panel_kind")
        assert kind in ("screening", "triage", "interpretation"), \
            f"{name} has no panel kind, so the card cannot say what it is"


# --------------------------------------------------------------- input handling
def test_empty_request_says_so(client):
    r = client.post("/predict", json={})
    body = r.json()
    assert body["status"] == "error"
    assert "no data" in body["message"].lower()


def test_no_panel_disappears_silently(client):
    """
    Every live panel must appear somewhere in the response.

    A woman entering routine bloodwork used to get seven results and no mention
    of a breast panel at all, because breast reads only biopsy morphology and
    `if not supplied: continue` dropped it without a word.
    """
    payload = {"age": 55, "gender": 0, **FULL_BLOODS}
    body = client.post("/predict", json=payload).json()
    assert body["status"] == "success"
    accounted = set(body["predictions"]) | set(body["skipped"]) | set(body["ignored"])
    labels = {b.get("label", n) for n, b in api.models.items()}
    missing = labels - accounted
    assert not missing, f"panels vanished from the response: {sorted(missing)}"


def test_sex_gating_both_directions(client):
    male = {"age": 62, "gender": 1, "psa": 6.2, **FULL_BLOODS}
    body = client.post("/predict", json=male).json()
    skipped = " ".join(body["skipped"]).lower()
    assert "ovarian" in skipped, "an ovarian score was offered to a man"

    female = {"age": 55, "gender": 0, **FULL_BLOODS}
    body = client.post("/predict", json=female).json()
    skipped = " ".join(body["skipped"]).lower()
    assert "prostate" in skipped, "a prostate score was offered to a woman"


def test_thin_input_is_refused_not_guessed(client):
    """
    One value out of many is not enough, and the panel has to say so rather
    than print a number built from training medians.
    """
    body = client.post("/predict", json={"age": 60, "gender": 1, "alt": 30}).json()
    text = " ".join(body.get("skipped", {}).values()).lower()
    assert body["status"] == "success"
    assert "not enough data" in text or "no data for this panel" in text


def test_out_of_range_values_are_rejected(client):
    """A haemoglobin of 5000 is a typo, and scoring it is worse than ignoring it."""
    body = client.post("/predict",
                       json={"age": 55, "gender": 0, "hemoglobin": 5000,
                             **{k: v for k, v in FULL_BLOODS.items()
                                if k != "hemoglobin"}}).json()
    assert body["status"] == "success"
    assert any("hemoglobin" in k for k in body["ignored"]), \
        "an impossible value was accepted as real"


def test_probabilities_are_bounded(client):
    body = client.post("/predict",
                       json={"age": 55, "gender": 0, **FULL_BLOODS}).json()
    for name, p in body["predictions"].items():
        risk = p.get("risk")
        if risk is None:
            continue
        assert 0.0 <= risk <= 100.0, f"{name} returned a risk of {risk}"


def test_triage_panels_refuse_without_their_defining_test(client):
    """
    Coverage counts every feature equally, and these panels are not built that
    way. The ovarian panel reads 27 values, 22 of them a routine blood count and
    metabolic panel and 5 of them the tumour markers a pelvic mass work-up
    actually turns on.

    A woman entering ordinary bloodwork and nothing else reached 81% coverage,
    was labelled HIGH confidence with no caveat, and was told her ovarian
    malignancy risk was 94.9% — with not one tumour marker supplied. The score
    came from training medians standing in for the only features that
    discriminate, on a case-control cohort with a 49% base rate.
    """
    payload = {"age": 58, "gender": 0, "menopause": 1, **FULL_BLOODS,
               "hematocrit": 40, "mcv": 88, "mch": 29, "rdw": 13.4,
               "mpv": 9.8, "neutrophil_pct": 60}
    body = client.post("/predict", json=payload).json()
    scored = " ".join(body["predictions"]).lower()
    skipped = " ".join(body["skipped"]).lower()

    assert "ovarian" not in scored, \
        "an ovarian malignancy score was returned with no tumour marker at all"
    assert "ovarian" in skipped
    assert "pancreatic" not in scored, \
        "a pancreatic score was returned with no CA 19-9"

    # And it must still score once the defining test is actually present,
    # otherwise the gate has simply broken the panel.
    body = client.post("/predict", json={**payload, "ca125": 420, "he4": 310}).json()
    assert any("ovarian" in k.lower() for k in body["predictions"]), \
        "the ovarian panel refused even with CA 125 and HE4 supplied"


def test_screening_panels_still_work_from_routine_bloodwork(client):
    """The gate must not catch the panels whose whole point is a lab report."""
    payload = {"age": 58, "gender": 0, **FULL_BLOODS}
    body = client.post("/predict", json=payload).json()
    scored = " ".join(body["predictions"]).lower()
    for panel in ("liver", "bowel", "lung"):
        assert panel in scored, f"the {panel} panel stopped scoring from routine bloodwork"


# --------------------------------------------------------------- exposure
def test_cors_is_not_open_to_everyone(client):
    """
    It was allow_origins=["*"] with allow_credentials=True, which is both
    permissive and self-contradictory: browsers refuse to send credentials to a
    wildcard origin, so the combination never did what it looked like it did.
    """
    assert "*" not in api.ALLOWED_ORIGINS
    r = client.get("/models", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}, \
        "a disallowed origin was granted CORS access"


def test_rate_limit_returns_429(client):
    """
    /parse-pdf hands whatever it is given to pdfplumber, so an unlimited request
    rate on a 512 MB instance is a denial-of-service surface.

    The counter is reset around this test rather than fired 60 times, so it
    cannot exhaust the budget for the tests that run after it.
    """
    saved = dict(api._hits)
    api._hits.clear()
    try:
        codes = [client.get("/models").status_code
                 for _ in range(api.RATE_LIMIT + 5)]
        assert 429 in codes, "the rate limit never triggered"
        assert codes[0] == 200, "the very first request was rejected"
    finally:
        api._hits.clear()
        api._hits.update(saved)


def test_upload_is_size_capped():
    """files[:5] bounded the number of documents and never their size."""
    assert api.MAX_UPLOAD_BYTES > 0
    assert api.MAX_UPLOAD_BYTES <= 32 * 1024 * 1024, \
        "the upload cap is large enough not to be a cap"
    assert api.MAX_PDF_PAGES > 0


# --------------------------------------------------------------- schema hygiene
def test_service_accepts_nothing_it_cannot_read():
    """
    The API went on accepting the withdrawn cervical panel's sexual history long
    after nothing scored it. An endpoint that accepts a sexual history is one
    that can receive and log one.
    """
    import re

    import train_models as tm

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fields_ts = open(os.path.join(root, "frontend/src/app/fields.ts"),
                     encoding="utf-8").read()
    form_keys = set(re.findall(r'\bkey:\s*"([A-Za-z0-9_]+)"', fields_ts))
    assert form_keys, "could not read the form schema"

    accepted = set(api.PatientData.model_fields)
    used = {f for cfg in tm.DATASETS if cfg["name"] not in tm.WITHDRAWN
            for f in cfg["features"]}
    unread = accepted - used - form_keys - {"age", "gender"}
    assert not unread, f"the API accepts fields nothing reads: {sorted(unread)}"
