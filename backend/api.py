"""
Oncovision API.

Two endpoints:
  POST /predict     a flat object of lab values and history, per domain risk out
  POST /parse-pdf   lab report PDFs in, extracted biomarker values out
  GET  /models      the trained model registry and its measured performance

Each domain model is a soft voting ensemble of XGBoost and Extra Trees, trained
by train_models.py on the features this application can actually collect. The
probability returned for a domain is the model's own output. Clinical
thresholds are reported alongside it as separate flags and never overwrite it.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import os
import pdfplumber
import io
import re

try:
    import shap
except ImportError:  # per-patient attribution degrades to global importance
    shap = None

# SHAP holds a second copy of every tree, which roughly doubles resident memory.
# On a constrained instance set ENABLE_SHAP=0 to fall back to global importance
# without redeploying code. Everything else keeps working.
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "1").strip().lower() not in ("0", "false", "no")

app = FastAPI(title="Oncovision AI")

# Who may call this service.
#
# This was allow_origins=["*"] with allow_credentials=True, which is both a
# permissive setting and a contradictory one: browsers refuse to send credentials
# to a wildcard origin, so the combination never did what it looked like it did.
# The deployed frontend and local development are the only callers that exist.
# ALLOWED_ORIGINS overrides without a redeploy, as a comma-separated list.
DEFAULT_ORIGINS = [
    "https://oncovision.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = ([o.strip() for o in _origins_env.split(",") if o.strip()]
                   if _origins_env else DEFAULT_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Upload limits. /parse-pdf hands whatever it is given to pdfplumber, which
# parses the whole document in memory. Without a cap, one large or deliberately
# malformed PDF exhausts a 512 MB instance, and the count limit that was here
# (files[:5]) bounded the number of files but not their size.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "40"))

# Request rate, per client address, as a fixed window. Deliberately simple: the
# service runs as a single instance, so an in-process counter is accurate here
# and a shared store would be pretending to a scale this does not have. If it
# ever runs multi-instance this must move to Redis, and the comment should stop
# being true rather than quietly becoming wrong.
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
RATE_WINDOW_SECONDS = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
_hits: dict = {}


def _rule_out_view(bundle, proba_pct: float) -> dict:
    """
    Whether this person falls below the panel's high-sensitivity cut.

    Reported with what the cut costs and buys, because a threshold with no
    consequences attached is not information. experiments/cost_model.py is the
    argument for offering this at all.
    """
    metrics = bundle.get("metrics") or {}
    ro = metrics.get("rule_out")
    if not ro:
        # A panel with no rule-out has a reason, and saying it is the difference
        # between a finding and an apparent omission.
        return {"rule_out": None,
                "no_rule_out_reason": metrics.get("no_rule_out_reason", "")}
    below = proba_pct < ro["threshold"] * 100.0
    return {
        "rule_out": {
            "below_cut": bool(below),
            "threshold_pct": round(ro["threshold"] * 100.0, 1),
            "sensitivity": ro["sensitivity"],
            "share_of_people_ruled_out": ro["share_ruled_out"],
            "cases_missed_per_100": ro["cases_missed_per_100"],
            "meaning": (
                "This panel would leave you out of further testing. At this "
                f"setting it caught {ro['sensitivity'] * 100:.0f} of every 100 "
                f"cases and excludes {ro['share_ruled_out'] * 100:.0f} percent of "
                "people, so it misses "
                f"{ro['cases_missed_per_100']} in 100. Those rates were measured "
                "on the cohort this cut was tuned on. Tested on a cohort from "
                "another decade the bowel panel caught slightly fewer cases and "
                "excluded slightly more people than promised, so treat the "
                "numbers as approximate rather than exact. It is a reason to "
                "feel less worried, not a clearance."
                if below else
                "This panel would NOT leave you out of further testing. That is "
                "not a prediction that you have cancer. It means there is not "
                "enough here to safely exclude you, which is the question worth "
                "asking before an expensive test."
            ),
        }
    }


@app.exception_handler(RequestValidationError)
async def readable_validation_error(request, exc):
    """
    A malformed field returned FastAPI's raw validation dump, which names
    pydantic internals and tells a patient nothing. Every field here is a
    number, so the only way to fail validation is to send something that is not
    one, and saying which field is both sufficient and readable.
    """
    fields = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        if loc:
            fields.append(DISPLAY_NAMES.get(loc[-1], loc[-1]))
    listed = ", ".join(dict.fromkeys(fields)) or "one of the fields"
    return JSONResponse(
        status_code=422,
        content={"status": "error",
                 "message": f"{listed} needs to be a number. Check that value and "
                            f"try again."},
    )


@app.middleware("http")
async def rate_limit(request, call_next):
    if request.method == "OPTIONS" or request.url.path in ("/", "/health"):
        return await call_next(request)
    now = time.time()
    who = request.client.host if request.client else "unknown"
    window = int(now // RATE_WINDOW_SECONDS)
    key = (who, window)
    # Drop counters from windows that have passed, so this cannot grow without
    # bound on a long-running instance.
    for k in [k for k in _hits if k[1] < window]:
        _hits.pop(k, None)
    _hits[key] = _hits.get(key, 0) + 1
    if _hits[key] > RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"status": "error",
                     "message": "Too many requests. Wait a minute and try again."},
        )
    return await call_next(request)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
models = {}

# Panels that only apply to one sex, using the app's coding: 0 female, 1 male.
# Without this a man's lab report came back carrying an ovarian and a cervical
# risk score.
SEX_SPECIFIC = {
    "ovarian": 0,
    "cervical": 0,
    "prostate": 1,
}

# The inputs without which a panel is not answering its own question. At least
# one of each panel's list must be present, or the panel is skipped.
#
# This exists because the coverage gate below counts every feature equally, and
# that is not how these panels work. The ovarian panel reads 27 values, of which
# 22 are a routine blood count and metabolic panel and 5 are the tumour markers
# ordered when a pelvic mass is being worked up. A woman entering ordinary
# bloodwork and nothing else reached 81 percent coverage, was labelled HIGH
# confidence with no caveat, and was told her ovarian malignancy risk was 94.9
# percent. Not one tumour marker had been supplied. The score came almost
# entirely from training medians standing in for the only features that
# discriminate, on a case-control cohort whose base rate is 49 percent.
#
# That is the cervical failure in a new place: a number that looks exactly as
# confident as a real one, built out of values the patient never gave. Coverage
# alone cannot catch it, because the missing features were a minority of the
# count and a majority of the information.
#
# Triage and interpretation panels are the ones at risk, because each is defined
# by a test that has already been performed. If that test is absent the panel
# has nothing to interpret, whatever else was entered.
DEFINING_INPUTS = {
    # A pelvic mass work-up. Without a tumour marker this is not that.
    "ovarian": ["ca125", "he4"],
    # CA 19-9 is the marker ordered when pancreatic cancer is suspected, and the
    # cohort separates adenocarcinoma from benign hepatobiliary disease on it.
    "pancreatic": ["plasma_ca19_9"],
    # An interpretation panel over a prostate MRI. No PI-RADS, no MRI.
    "prostate": ["pi_rads"],
}

# The share of a panel's inputs that has to be real, rather than filled in from
# a training median, before it will report a number.
#
# Two failures sit either side of this line. Demanding a complete panel makes
# the application useless, because a partial lab report is the normal case.
# Scoring off almost nothing produces a number that looks exactly as confident
# as a complete one, which is how the cervical panel came to report "Raised"
# from a single input out of fifteen.
#
# So the bar is half, and below it the panel says "Not enough data" and names
# how many values it needs, rather than either refusing silently or bluffing.
MIN_COVERAGE = 0.5

# Ranges outside which a value cannot belong to a living patient. Anything
# beyond these is treated as a typo and excluded rather than clamped.
BIOLOGICAL_BOUNDS = {
    "age": (1, 120), "bmi": (10, 70), "wbc": (0.0, 100.0), "rbc": (0.0, 10.0),
    "hemoglobin": (0.0, 25.0), "platelets": (0.0, 1000.0), "glucose": (0.0, 600.0),
    "calcium": (0.0, 20.0), "bun": (0.0, 100.0), "creatinine": (0.0, 15.0),
    "protein_total": (0.0, 12.0), "albumin": (0.0, 8.0), "ast": (0.0, 2000.0),
    "alt": (0.0, 2000.0), "bilirubin": (0.0, 50.0), "alkaline_phosphatase": (0.0, 2000.0),
    "alpha_fetoprotein_level": (0.0, 50000.0), "psa": (0.0, 1000.0),
    "plasma_ca19_9": (0.0, 50000.0), "radius_mean": (0.0, 40.0),
    "texture_mean": (0.0, 50.0), "perimeter_mean": (0.0, 250.0), "area_mean": (0.0, 2500.0),
    # History fields
    "gender": (0, 1), "smoking": (0, 2), "alcohol_intake": (0, 5), "hepatitis_b": (0, 1), "hepatitis_c": (0, 1), "diabetes": (0, 1),
    # Red cell and platelet indices, GGT, and the pelvic-mass tumour markers.
    "hematocrit": (0.0, 70.0), "mcv": (0.0, 150.0), "mch": (0.0, 60.0),
    "rdw": (0.0, 40.0), "mpv": (0.0, 20.0), "neutrophil_pct": (0.0, 100.0),
    "ggt": (0.0, 2000.0), "ca125": (0.0, 50000.0), "he4": (0.0, 50000.0),
    "cea": (0.0, 10000.0),
    # Menopausal status for the ovarian panel, pack-years for lung. The rest of
    # the reproductive and sexual history went when the cervical panel was
    # withdrawn: the service should not accept a sexual history that nothing
    # scores.
    "menopause": (0, 1), "smoking_packyears": (0, 200),
    # Tobacco exposure, inflammation, and the prostate work-up.
    "cotinine": (0.0, 2000.0), "crp": (0.0, 500.0),
    "prostate_volume": (1.0, 300.0), "psa_density": (0.0, 50.0), "pi_rads": (1, 5),
    # The remaining Wisconsin aspirate measurements.
    "smoothness_mean": (0.0, 300.0),
    "compactness_mean": (0.0, 300.0),
    "concavity_mean": (0.0, 300.0),
    "concave_points_mean": (0.0, 300.0),
    "symmetry_mean": (0.0, 300.0),
    "fractal_dimension_mean": (0.0, 300.0),
    "radius_se": (0.0, 300.0),
    "texture_se": (0.0, 300.0),
    "perimeter_se": (0.0, 300.0),
    "area_se": (0.0, 3000.0),
    "smoothness_se": (0.0, 300.0),
    "compactness_se": (0.0, 300.0),
    "concavity_se": (0.0, 300.0),
    "concave_points_se": (0.0, 300.0),
    "symmetry_se": (0.0, 300.0),
    "fractal_dimension_se": (0.0, 300.0),
    "radius_worst": (0.0, 300.0),
    "texture_worst": (0.0, 300.0),
    "perimeter_worst": (0.0, 300.0),
    "area_worst": (0.0, 3000.0),
    "smoothness_worst": (0.0, 300.0),
    "compactness_worst": (0.0, 300.0),
    "concavity_worst": (0.0, 300.0),
    "concave_points_worst": (0.0, 300.0),
    "symmetry_worst": (0.0, 300.0),
    "fractal_dimension_worst": (0.0, 300.0),
}

# Upper limit of the normal reference range for each measurable marker.
REFERENCE_RANGES = {
    "wbc": 11.0, "rbc": 5.5, "hemoglobin": 15.0, "platelets": 400.0,
    "glucose": 99.0, "calcium": 10.3, "bun": 20.0, "creatinine": 1.2,
    "protein_total": 8.0, "albumin": 5.0, "ast": 40.0, "alt": 40.0,
    "bilirubin": 1.2, "alkaline_phosphatase": 120.0, "alpha_fetoprotein_level": 10.0,
    "psa": 4.0, "plasma_ca19_9": 37.0, "radius_mean": 15.0, "texture_mean": 22.0,
    "perimeter_mean": 95.0, "area_mean": 600.0, "bmi": 25.0,
    "hematocrit": 46.0, "mcv": 100.0, "mch": 33.0, "rdw": 14.5, "mpv": 12.0,
    "neutrophil_pct": 70.0, "ggt": 50.0,
    # CA125 above 35 U/mL is the conventional cut. HE4 is assay dependent and
    # 140 pmol/L is the common premenopausal cut used in the ROMA index.
    "ca125": 35.0, "he4": 140.0, "cea": 5.0,
    "cotinine": 3.0, "crp": 3.0, "prostate_volume": 30.0, "psa_density": 0.15,
    "pi_rads": 2.0,
}

# Markers with an established clinical decision threshold. Crossing one is
# reported to the user as a flag on the relevant domain. It does not alter the
# model probability.
CLINICAL_THRESHOLDS = {
    "alpha_fetoprotein_level": ("AFP", 10.0, ["liver"]),
    "plasma_ca19_9": ("CA 19-9", 37.0, ["pancreatic"]),
    "psa": ("PSA", 4.0, ["prostate"]),
    "bilirubin": ("Total bilirubin", 1.2, ["liver", "pancreatic"]),
    "alkaline_phosphatase": ("Alkaline phosphatase", 120.0, ["liver", "pancreatic"]),
    "ast": ("AST", 40.0, ["liver"]),
    "alt": ("ALT", 40.0, ["liver"]),
    "wbc": ("WBC", 11.0, ["general"]),
    "calcium": ("Calcium", 10.3, ["general"]),
    "glucose": ("Glucose", 99.0, ["pancreatic"]),
    "radius_mean": ("Nuclear radius", 15.0, ["breast"]),
    "area_mean": ("Nuclear area", 600.0, ["breast"]),
    "perimeter_mean": ("Nuclear perimeter", 95.0, ["breast"]),
    "ca125": ("CA 125", 35.0, ["ovarian"]),
    "he4": ("HE4", 140.0, ["ovarian"]),
    "cea": ("CEA", 5.0, ["ovarian"]),
    "ggt": ("GGT", 50.0, ["liver", "ovarian"]),
    "cotinine": ("Serum cotinine", 3.0, ["lung"]),
    "crp": ("CRP", 3.0, ["lung"]),
    "psa_density": ("PSA density", 0.15, ["prostate"]),
    "pi_rads": ("PI-RADS", 2.0, ["prostate"]),
}

# History answers worth surfacing when they are positive.
HISTORY_FLAGS = {
    "hepatitis_b": ("Hepatitis B", ["liver"]),
    "hepatitis_c": ("Hepatitis C", ["liver"]),
    "diabetes": ("Diabetes", ["liver", "pancreatic"]),
    "menopause": ("Post-menopausal", ["ovarian"]),
}

# Coded history answers rendered back into words for the driver breakdown.
CODED_VALUES = {
    "gender": {0: "Female", 1: "Male"},
    "smoking": {0: "Never smoked", 1: "Former smoker", 2: "Current smoker"},
    "hepatitis_b": {0: "Negative", 1: "Positive"},
    "hepatitis_c": {0: "Negative", 1: "Positive"},
    "diabetes": {0: "No", 1: "Yes"},
    "menopause": {0: "Pre-menopausal", 1: "Post-menopausal"},
}

UNITS = {
    "bmi": "", "wbc": "K/uL", "rbc": "M/uL", "hemoglobin": "g/dL", "platelets": "K/uL",
    "glucose": "mg/dL", "calcium": "mg/dL", "bun": "mg/dL", "creatinine": "mg/dL",
    "protein_total": "g/dL", "albumin": "g/dL", "ast": "U/L", "alt": "U/L",
    "bilirubin": "mg/dL", "alkaline_phosphatase": "U/L",
    "alpha_fetoprotein_level": "ng/mL", "psa": "ng/mL", "plasma_ca19_9": "U/mL",
    "age": "years", "alcohol_intake": "of 5",
    "hematocrit": "%", "mcv": "fL", "mch": "pg", "rdw": "%", "mpv": "fL",
    "neutrophil_pct": "%", "ggt": "U/L",
    "ca125": "U/mL", "he4": "pmol/L", "cea": "ng/mL",
    "smoking_packyears": "pack-years",
    "cotinine": "ng/mL", "crp": "mg/L", "prostate_volume": "mL",
    "psa_density": "ng/mL/mL", "pi_rads": "of 5",
}

DISPLAY_NAMES = {
    "alpha_fetoprotein_level": "AFP", "plasma_ca19_9": "CA 19-9", "psa": "PSA",
    "wbc": "WBC", "rbc": "RBC", "bun": "BUN", "ast": "AST", "alt": "ALT",
    "alkaline_phosphatase": "ALP", "protein_total": "Total protein",
    "radius_mean": "Radius mean", "texture_mean": "Texture mean",
    "perimeter_mean": "Perimeter mean", "area_mean": "Area mean",
    "bmi": "BMI", "hemoglobin": "Hemoglobin", "platelets": "Platelets",
    "glucose": "Glucose", "calcium": "Calcium", "creatinine": "Creatinine",
    "albumin": "Albumin", "bilirubin": "Bilirubin", "age": "Age",
    "hematocrit": "Hematocrit", "mcv": "MCV", "mch": "MCH", "rdw": "RDW",
    "mpv": "MPV", "neutrophil_pct": "Neutrophils", "ggt": "GGT",
    "ca125": "CA 125", "he4": "HE4", "cea": "CEA",
    "menopause": "Menopausal status", "smoking_packyears": "Pack-years",
    "cotinine": "Serum cotinine", "crp": "CRP",
    "prostate_volume": "Prostate volume", "psa_density": "PSA density",
    "pi_rads": "PI-RADS score",
    "smoothness_mean": "Smoothness Mean",
    "compactness_mean": "Compactness Mean",
    "concavity_mean": "Concavity Mean",
    "concave_points_mean": "Concave Points Mean",
    "symmetry_mean": "Symmetry Mean",
    "fractal_dimension_mean": "Fractal Dimension Mean",
    "radius_se": "Radius Se",
    "texture_se": "Texture Se",
    "perimeter_se": "Perimeter Se",
    "area_se": "Area Se",
    "smoothness_se": "Smoothness Se",
    "compactness_se": "Compactness Se",
    "concavity_se": "Concavity Se",
    "concave_points_se": "Concave Points Se",
    "symmetry_se": "Symmetry Se",
    "fractal_dimension_se": "Fractal Dimension Se",
    "radius_worst": "Radius Worst",
    "texture_worst": "Texture Worst",
    "perimeter_worst": "Perimeter Worst",
    "area_worst": "Area Worst",
    "smoothness_worst": "Smoothness Worst",
    "compactness_worst": "Compactness Worst",
    "concavity_worst": "Concavity Worst",
    "concave_points_worst": "Concave Points Worst",
    "symmetry_worst": "Symmetry Worst",
    "fractal_dimension_worst": "Fractal Dimension Worst",
}


def pretty(key: str) -> str:
    return DISPLAY_NAMES.get(key, key.replace("_", " ").capitalize())


def readable(key: str, value: float) -> str:
    """Render a stored value the way a person would say it."""
    if key in CODED_VALUES:
        return CODED_VALUES[key].get(int(value), str(value))
    shown = int(value) if float(value).is_integer() else round(float(value), 2)
    unit = UNITS.get(key, "")
    return f"{shown} {unit}".strip()


@app.on_event("startup")
def load_models():
    if not os.path.isdir(MODEL_DIR):
        return
    for filename in sorted(os.listdir(MODEL_DIR)):
        if filename.endswith(".joblib"):
            name = filename.replace("model_", "").replace(".joblib", "")
            models[name] = joblib.load(os.path.join(MODEL_DIR, filename))


def inner_estimator(bundle):
    """Unwrap CalibratedClassifierCV to the model it actually calibrates."""
    model = bundle["model"]
    cal = getattr(model, "calibrated_classifiers_", None)
    if cal:
        return getattr(cal[0], "estimator", model)
    return model


def voting_members(bundle):
    """
    The fitted members of a soft-voting ensemble, or an empty list.

    Not every panel ships an ensemble. Model selection in train_models.py picks
    logistic regression where it beats the trees on cross-validated AUC, which
    is currently the case for liver and pancreatic, so this returns nothing for
    those and the linear paths below handle them instead.
    """
    return list(getattr(inner_estimator(bundle), "estimators_", []))


def linear_model(bundle):
    """The fitted LogisticRegression inside a scaling pipeline, or None."""
    inner = inner_estimator(bundle)
    steps = getattr(inner, "named_steps", None)
    if steps:
        for step in steps.values():
            if hasattr(step, "coef_"):
                return inner, step
    return (inner, inner) if hasattr(inner, "coef_") else (None, None)


def ensemble_importances(bundle) -> np.ndarray:
    """
    Global feature importance, however this panel's model expresses it.
    Tree ensembles report impurity importance; logistic regression reports the
    magnitude of its standardised coefficients.
    """
    parts = []
    for est in voting_members(bundle):
        imp = getattr(est, "feature_importances_", None)
        if imp is not None:
            total = float(np.sum(imp))
            if total > 0:
                parts.append(np.asarray(imp, dtype=float) / total)

    if not parts:
        _, lin = linear_model(bundle)
        if lin is not None:
            coef = np.abs(np.asarray(lin.coef_, dtype=float).reshape(-1))
            total = float(coef.sum())
            if total > 0:
                parts.append(coef / total)

    if not parts:
        n = len(bundle["feature_names"])
        return np.full(n, 1.0 / n)
    return np.mean(parts, axis=0)


_explainers: dict = {}


def get_explainers(name: str, bundle):
    """TreeExplainers for both ensemble members, built once and cached."""
    if name in _explainers:
        return _explainers[name]
    built = []
    if shap is not None and ENABLE_SHAP:
        for est in voting_members(bundle):
            try:
                built.append(shap.TreeExplainer(est))
            except Exception:
                pass
    _explainers[name] = built
    return built


def shap_attribution(name: str, bundle, frame: pd.DataFrame, supplied: set) -> list:
    """
    Per-patient SHAP attribution for one prediction.

    This is a real explanation of this patient's score, not a global importance
    ranking. Each ensemble member is explained separately and its vector is
    normalised to a share of that member's total absolute attribution, then the
    two are averaged. Normalising is necessary because XGBoost reports in log
    odds and Extra Trees in probability, so the raw magnitudes are not on a
    common scale. Sign is preserved, so a feature that pushed the score down
    stays negative.

    Only features the patient actually supplied are returned. A contribution
    from an imputed median describes the training set, not the person.
    """
    features = bundle["feature_names"]

    # Logistic panels get an exact analytic decomposition instead of SHAP.
    # For a linear model the log-odds contribution of each feature is simply
    # coefficient times standardised value, which is what SHAP would compute
    # anyway, so this is the same answer without the extra dependency.
    pipe, lin = linear_model(bundle)
    if lin is not None:
        try:
            scaler = None
            steps = getattr(pipe, "named_steps", {}) or {}
            for step in steps.values():
                if hasattr(step, "mean_") and hasattr(step, "scale_"):
                    scaler = step
                    break
            raw = frame.to_numpy(dtype=float).reshape(-1)
            z = (raw - scaler.mean_) / scaler.scale_ if scaler is not None else raw
            contrib = np.asarray(lin.coef_, dtype=float).reshape(-1) * z
            total = float(np.abs(contrib).sum())
            if total <= 0:
                return []
            share = contrib / total
            out = [
                {
                    "name": pretty(f),
                    "feature": f,
                    "share": round(float(abs(share[i])) * 100, 1),
                    "direction": "raises" if share[i] > 0 else "lowers",
                    "signed": round(float(share[i]) * 100, 1),
                }
                for i, f in enumerate(features) if f in supplied
            ]
            out.sort(key=lambda d: d["share"], reverse=True)
            return out[:8]
        except Exception:
            return []

    explainers = get_explainers(name, bundle)
    if not explainers:
        return []

    vectors = []
    for ex in explainers:
        try:
            raw = ex.shap_values(frame)
        except Exception:
            continue
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 3:          # (rows, features, classes) from Extra Trees
            arr = arr[:, :, 1]
        arr = arr.reshape(-1)[:len(features)]
        total = float(np.abs(arr).sum())
        if total > 0:
            vectors.append(arr / total)

    if not vectors:
        return []

    mean = np.mean(vectors, axis=0)
    out = []
    for idx, feat in enumerate(features):
        if feat not in supplied:
            continue
        out.append({
            "name": pretty(feat),
            "feature": feat,
            "share": round(float(abs(mean[idx])) * 100, 1),
            "direction": "raises" if mean[idx] > 0 else "lowers",
            "signed": round(float(mean[idx]) * 100, 1),
        })
    out.sort(key=lambda d: d["share"], reverse=True)
    return out[:8]


class PatientData(BaseModel):
    """
    The /predict request body, sent as a flat object.

    Every field is optional and defaults to None, because the interface asks the
    patient for whatever they happen to have and never requires a complete panel.
    A model scores only when it receives at least one of its own features, and
    anything left out is filled with that feature's training median.

    Unknown keys are ignored rather than rejected, so the frontend can post its
    whole form without filtering first.
    """

    model_config = ConfigDict(extra="ignore")

    # Body metrics
    age: Optional[float] = None
    bmi: Optional[float] = None

    # Complete blood count
    wbc: Optional[float] = None
    rbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    platelets: Optional[float] = None
    hematocrit: Optional[float] = None
    mcv: Optional[float] = None
    mch: Optional[float] = None
    rdw: Optional[float] = None
    mpv: Optional[float] = None
    neutrophil_pct: Optional[float] = None

    # Metabolic panel
    glucose: Optional[float] = None
    calcium: Optional[float] = None
    bun: Optional[float] = None
    creatinine: Optional[float] = None
    protein_total: Optional[float] = None
    albumin: Optional[float] = None

    # Liver panel
    ast: Optional[float] = None
    alt: Optional[float] = None
    bilirubin: Optional[float] = None
    alkaline_phosphatase: Optional[float] = None
    ggt: Optional[float] = None

    # Tumour markers
    alpha_fetoprotein_level: Optional[float] = None
    psa: Optional[float] = None
    plasma_ca19_9: Optional[float] = None
    cotinine: Optional[float] = None
    crp: Optional[float] = None
    prostate_volume: Optional[float] = None
    psa_density: Optional[float] = None
    pi_rads: Optional[float] = None
    ca125: Optional[float] = None
    he4: Optional[float] = None
    cea: Optional[float] = None

    # Breast mass morphology
    radius_mean: Optional[float] = None
    texture_mean: Optional[float] = None
    perimeter_mean: Optional[float] = None
    area_mean: Optional[float] = None
    smoothness_mean: Optional[float] = None
    compactness_mean: Optional[float] = None
    concavity_mean: Optional[float] = None
    concave_points_mean: Optional[float] = None
    symmetry_mean: Optional[float] = None
    fractal_dimension_mean: Optional[float] = None
    radius_se: Optional[float] = None
    texture_se: Optional[float] = None
    perimeter_se: Optional[float] = None
    area_se: Optional[float] = None
    smoothness_se: Optional[float] = None
    compactness_se: Optional[float] = None
    concavity_se: Optional[float] = None
    concave_points_se: Optional[float] = None
    symmetry_se: Optional[float] = None
    fractal_dimension_se: Optional[float] = None
    radius_worst: Optional[float] = None
    texture_worst: Optional[float] = None
    perimeter_worst: Optional[float] = None
    area_worst: Optional[float] = None
    smoothness_worst: Optional[float] = None
    compactness_worst: Optional[float] = None
    concavity_worst: Optional[float] = None
    concave_points_worst: Optional[float] = None
    symmetry_worst: Optional[float] = None
    fractal_dimension_worst: Optional[float] = None

    # Patient history
    gender: Optional[float] = None
    smoking: Optional[float] = None
    alcohol_intake: Optional[float] = None
    hepatitis_b: Optional[float] = None
    hepatitis_c: Optional[float] = None
    diabetes: Optional[float] = None

    # Menopausal status, read by the ovarian panel. Pack-years, read by lung.
    menopause: Optional[float] = None
    smoking_packyears: Optional[float] = None


@app.get("/")
def root():
    """Hitting the base URL wakes a sleeping free tier instance and confirms which build is live."""
    return {
        "service": "Oncovision AI",
        "status": "online",
        "models_loaded": sorted(models.keys()),
        "endpoints": ["/predict", "/parse-pdf", "/models", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if models else "degraded",
        "models_loaded": len(models),
        "shap_enabled": bool(shap is not None and ENABLE_SHAP),
    }


@app.get("/models")
def model_registry():
    return {
        "status": "success",
        "models": {
            name: {
                "label": b.get("label", name),
                "algorithm": b.get("algorithm", ""),
                "features": b.get("feature_names", []),
                "positive_means": b.get("positive_means", ""),
                "cohort_design": b.get("cohort_design", ""),
                "threshold": b.get("threshold"),
                "metrics": b.get("metrics", {}),
                "held_out": b.get("held_out", {}),
            }
            for name, b in models.items()
        },
    }


@app.post("/predict")
async def predict_risk(data: PatientData):
    values = {}
    ignored = {}

    # Pydantic has already dropped anything absent and coerced the rest to
    # float, so the only check left is whether a value is physiologically
    # possible. Anything outside the bounds is excluded and reported back
    # rather than clamped, so a typo cannot quietly move a score.
    for key, val in data.model_dump(exclude_none=True).items():
        low, high = BIOLOGICAL_BOUNDS.get(key, (-1e12, 1e12))
        if val < low or val > high:
            ignored[key] = f"outside {low} to {high}"
            continue
        values[key] = val

    if not values:
        return {"status": "error", "message": "No data entered."}

    results = {}
    skipped = {}

    for name, bundle in models.items():
        features = bundle["feature_names"]
        medians = bundle.get("feature_medians", {})

        # Anatomy gate. Reporting an ovarian or cervical risk score for a man is
        # not a rounding error, it is nonsense, and it happened: a male lab
        # report with a PSA on it came back with a "Raised" cervical panel.
        # Sex is only a gate when the patient actually told us their sex.
        required_sex = SEX_SPECIFIC.get(name)
        if required_sex is not None and "gender" not in values:
            # Sex-specific panels used to score whenever sex was simply absent,
            # on the reasoning that a gate should only fire on information the
            # patient actually gave. The result was worse than the gap it was
            # avoiding: a lab report PDF carries no sex, so uploading one
            # returned an ovarian risk AND a prostate risk for the same person.
            # One of those is always wrong and the interface had no way to say
            # which. Asking is one click and removes both.
            skipped[bundle.get("label", name)] = (
                "Needs your sex. This panel only applies to "
                + ("women" if required_sex == 0 else "men")
                + ", and a lab report does not say which you are. Answer the sex "
                  "question and run the analysis again."
            )
            continue
        if required_sex is not None and "gender" in values:
            if float(values["gender"]) != float(required_sex):
                skipped[bundle.get("label", name)] = (
                    "Not applicable. This panel only applies to "
                    + ("women." if required_sex == 0 else "men.")
                )
                continue

        # The defining-input gate, which runs before coverage because coverage
        # cannot see it. See DEFINING_INPUTS above for the failure this prevents.
        # A panel that cannot run its full version may still have a weaker one
        # that runs on a lab report. Refusing outright turns the application off
        # for exactly the person it was built for: a man with a PSA on his
        # annual bloodwork was getting nothing, because the full prostate panel
        # wants a PI-RADS score from an MRI he has not had. The reduced tier
        # reaches 0.676 against 0.825, which is weak and is not silence.
        reduced = bundle.get("reduced")
        needed = DEFINING_INPUTS.get(name)
        using_reduced = False
        if (needed and not any(f in values for f in needed) and reduced
                and any(f in values for f in reduced["feature_names"]
                        if f not in ("age", "gender", "bmi"))):
            using_reduced = True
            features = reduced["feature_names"]
            needed = None

        if needed and not any(f in values for f in needed):
            # Not named `pretty`: there is already a pretty() helper in this
            # scope, and shadowing it made every later call a NameError.
            needed_names = " or ".join(DISPLAY_NAMES.get(f, f) for f in needed)
            skipped[bundle.get("label", name)] = (
                f"Not applicable without {needed_names}. This panel reads a test that has "
                f"already been done, and without it there is nothing for it to "
                f"interpret. Routine bloodwork on its own cannot answer this one."
            )
            continue

        # Values are clipped to the range the panel was actually trained over.
        #
        # A tree model has no splits past the edge of its data, so beyond that
        # edge it returns whichever leaf it lands in, with undiminished
        # confidence. The liver panel scored a coherent acute-hepatitis pattern
        # (ALT 300, AST 260, GGT 200, bilirubin 2.5) at 3.0 percent, which is
        # LOWER than a completely normal patient and far below a mild
        # abnormality at 14.1 percent. Only 19 of 35,511 people in that cohort
        # have an ALT over 250 and none of the 1,436 cases exceeds 232, so the
        # model learned that a very high ALT means no liver disease: true of
        # NHANES, false of medicine.
        #
        # Clipping does not make the panel right about such a patient. It stops
        # it being confidently backwards, and `extreme` below says plainly that
        # the value was off the end of what the panel knows about.
        ranges = bundle.get("feature_ranges") or {}
        row, supplied, extreme = [], [], []
        for feat in features:
            if feat in values:
                val = values[feat]
                lo_hi = ranges.get(feat)
                if lo_hi:
                    lo, hi = lo_hi
                    if val > hi:
                        extreme.append((feat, val, hi, "above"))
                        val = hi
                    elif val < lo:
                        extreme.append((feat, val, lo, "below"))
                        val = lo
                row.append(val)
                supplied.append(feat)
            else:
                row.append(medians.get(feat, 0.0))

        # Nothing this panel reads was entered at all. It used to `continue`
        # here with no message, so the panel simply vanished from the response:
        # not scored, not skipped, not mentioned. A woman entering a routine
        # blood panel got seven results and no indication that a breast panel
        # existed, because breast reads only biopsy morphology and she had
        # supplied none of it. Silence is the one thing this application is not
        # allowed to do, so it says which values it would need.
        if not supplied:
            skipped[bundle.get("label", name)] = (
                f"No data for this panel. It reads {len(features)} values and none "
                f"of them were entered."
            )
            continue

        # Coverage. Every panel scores with whatever it was given, because
        # refusing to answer is useless to someone holding a partial lab report,
        # and the whole point of storing a training median per feature is to
        # cope with blanks.
        #
        # What coverage changes is how loudly the answer is qualified, not
        # whether there is one. The failure this replaces was a cervical panel
        # reporting "Raised" off a single input out of fifteen with no hint that
        # it had invented the other fourteen. The number was not the problem.
        # The silence about where it came from was.
        #
        # The one thing still refused is a score with nothing but age and sex
        # behind it, because that is not reading a lab report at all, it is
        # reciting the population average for someone's demographic.
        informative = [f for f in supplied if f not in ("age", "gender")]
        coverage = len(supplied) / len(features)
        need = max(1, int(np.ceil(MIN_COVERAGE * len(features))))

        # Not enough to answer accurately. Say that plainly instead of printing
        # a number built mostly out of training medians, which is what the
        # cervical panel used to do: it reported "Raised" off one input in
        # fifteen and looked exactly as confident as a complete panel.
        #
        # The bar is deliberately half, not everything. A partial lab report is
        # the normal case and still gets a real answer.
        if not informative:
            skipped[bundle.get("label", name)] = (
                f"Not enough data. Only age and sex were entered, and this panel reads "
                f"{len(features)} values. Enter at least {need} of them."
            )
            continue
        if coverage < MIN_COVERAGE:
            skipped[bundle.get("label", name)] = (
                f"Not enough data. You entered {len(supplied)} of the {len(features)} "
                f"values this panel reads, and it needs at least {need} to be accurate."
            )
            continue

        if coverage >= 0.75:
            confidence, caveat = "high", ""
        else:
            confidence, caveat = "moderate", (
                f"Based on {len(supplied)} of {len(features)} values. The rest were filled "
                f"with typical readings, which pulls this score toward an average patient."
            )

        try:
            frame = pd.DataFrame([row], columns=features)
            scorer = reduced["model"] if using_reduced else bundle["model"]
            proba = float(scorer.predict_proba(frame)[0][1]) * 100.0
            # Isotonic calibration maps the top bin to exactly 1.0, so an
            # extreme record can come back as a flat 100 percent. No model here
            # has earned certainty: the best panel's confidence interval still
            # runs to 0.993, and reporting 100 would claim a precision the
            # held-out test cannot support. Clipped at both ends for the same
            # reason.
            # The rule-out comparison must use the UNCLIPPED probability.
            #
            # Clipping to [0.1, 99.9] is right for display: no panel here has
            # earned a claim of certainty. But the bowel rule-out cut is 0.09
            # percent, which sits BELOW the 0.1 floor, so comparing the clipped
            # value made it unreachable and no patient could ever be ruled out.
            # The feature was silently inert until a healthy test patient came
            # back "cannot be ruled out" by every panel at once.
            raw_proba = proba
            proba = min(max(proba, 0.1), 99.9)
        except Exception:
            continue

        importances = ensemble_importances(bundle)

        contributors = []
        for idx, feat in enumerate(features):
            if feat not in values or feat in ("age", "gender"):
                continue
            limit = REFERENCE_RANGES.get(feat)
            if limit is None:
                continue
            val = values[feat]
            ratio = val / limit if limit else 1.0
            contributors.append({
                "name": pretty(feat),
                "impact": round(float(importances[idx]) * min(ratio, 4.0) * 100, 2),
                "value": val,
                "limit": limit,
                "over": val > limit,
            })
        contributors.sort(key=lambda c: c["impact"], reverse=True)

        # Every feature the patient supplied, ranked by how much this model
        # relies on it. Covers history answers that have no reference range and
        # therefore cannot appear in the chart above.
        drivers = []
        for idx, feat in enumerate(features):
            if feat not in values:
                continue
            limit = REFERENCE_RANGES.get(feat)
            val = values[feat]
            drivers.append({
                "name": pretty(feat),
                "reading": readable(feat, val),
                "weight": round(float(importances[idx]) * 100, 1),
                "abnormal": bool(limit is not None and val > limit)
                            or bool(feat in CODED_VALUES and feat != "gender" and val > 0),
            })
        drivers.sort(key=lambda d: d["weight"], reverse=True)

        flags = []
        for key, (label, limit, domains) in CLINICAL_THRESHOLDS.items():
            if name in domains and key in values and values[key] > limit:
                flags.append({
                    "label": f"{label} above {limit}",
                    "detail": f"{values[key]} recorded",
                })
        for key, (label, domains) in HISTORY_FLAGS.items():
            if name in domains and values.get(key, 0) == 1:
                flags.append({"label": label, "detail": "reported by patient"})

        # Banding is relative to this panel's own operating threshold, not to a
        # flat 50 percent. The models are calibrated against real prevalence, so
        # on a 4 percent condition a genuinely concerning result sits near 10
        # percent, and a fixed 50 percent cut would flag almost nobody. The
        # threshold is chosen on training data by Youden's J and frozen in the
        # bundle, so the interface, the evaluation and the prospective analysis
        # all use the same number.
        # The wording here is deliberately plain. "Above the operating threshold
        # selected by Youden's J" is precise and means nothing to the person
        # reading it, so the band says what it means in ordinary words and the
        # exact number stays on the card underneath, unchanged.
        threshold_pct = float(bundle.get("threshold", 0.5)) * 100.0
        if proba >= threshold_pct * 2:
            band = "Clearly raised"
            meaning = ("Well above the level this panel treats as worth a second look. "
                       "Worth showing to a doctor, alongside the rest of your results.")
        elif proba >= threshold_pct:
            band = "Raised"
            meaning = ("Above the level this panel treats as worth a second look. "
                       "That is a reason to ask a question, not a reason to panic.")
        elif proba >= threshold_pct * 0.5:
            band = "Borderline"
            meaning = ("Below the level this panel flags, but not by much. "
                       "Usually nothing, and worth mentioning if you have symptoms.")
        else:
            band = "Nothing unusual"
            meaning = ("Nothing in the values you entered looks unusual to this panel. "
                       "A normal result does not rule anything out.")

        metrics = bundle.get("metrics", {})
        held = bundle.get("held_out", {})
        stab = bundle.get("stability", {})
        fair = bundle.get("fairness", {})
        # A reduced answer gets its own label. The full one names a test the
        # user has not had, which reads as a mistake on their own result card.
        card_label = bundle.get("label", name)
        if using_reduced and (reduced or {}).get("label"):
            card_label = reduced["label"]
        results[card_label] = {
            "key": name,
            "risk": round(proba, 1),
            "band": band,
            "meaning": meaning,
            "confidence": confidence,
            "coverage_caveat": caveat,
            # Which tier of this panel answered. A reduced answer is a real
            # answer and a weaker one, and the difference belongs on the card
            # rather than buried in a log.
            "tier": "lab report only" if using_reduced else "full",
            # The rule-out call, which is the one that matters before an
            # expensive procedure. "Are you flagged" balances a false positive
            # against a false negative as though a colonoscopy and a missed
            # cancer cost the same. "Can you be left out" does not.
            **_rule_out_view(bundle, raw_proba),
            "reduced_note": (reduced or {}).get("note", "") if using_reduced else "",
            "reduced_auc": (reduced or {}).get("auc") if using_reduced else None,
            # Values that fell outside anything this panel was trained on. The
            # score for such a patient is a floor, not an estimate, and saying
            # so is the difference between a limitation and a wrong answer.
            "beyond_training_range": [
                {
                    "field": f,
                    "name": DISPLAY_NAMES.get(f, f),
                    "entered": v,
                    "furthest_seen": round(edge, 1),
                    "direction": d,
                }
                for f, v, edge, d in extreme
            ],
            "extreme_value_caveat": (
                "One or more of your values is further from normal than anything this "
                "panel has data on: "
                + ", ".join(f"{DISPLAY_NAMES.get(f, f)} {v:g}" for f, v, _, _ in extreme)
                + ". The panel cannot rank a value it has never seen, so this score is "
                  "a floor rather than an estimate, and the value itself matters more "
                  "than the percentage. Show it to a doctor."
                if extreme else ""
            ),
            "threshold": round(threshold_pct, 1),
            "above_threshold": bool(proba >= threshold_pct),
            "contributors": contributors[:6],
            "drivers": drivers[:8],
            "shap": shap_attribution(name, bundle, frame, set(supplied)),
            "flags": flags,
            "inputs_used": len(supplied),
            "inputs_total": len(features),
            "coverage": round(len(supplied) / len(features) * 100),
            "missing": [pretty(f) for f in features if f not in values],
            # Held-out test evidence. These are the numbers that belong in front
            # of a person: measured on a split cut before anything was fitted.
            "auc": held.get("auc", metrics.get("auc")),
            # The mean across repeated 80/20 splits, and where the one split
            # quoted above lands inside that distribution. A single split is one
            # draw, and on a small cohort the draw moves a lot: colorectal drew
            # a split where it scores below its own age-and-sex baseline while
            # over twenty repeats it beats that baseline by 0.038. Both numbers
            # are shown so neither can quietly stand in for the other.
            "stable_auc": stab.get("stable_auc"),
            "split_spread": stab.get("split_spread"),
            "split_percentile": stab.get("shipped_split_percentile"),
            # Per-group accuracy, where the cohort records race and ethnicity.
            # Shown because a panel that works measurably worse for one group
            # and stays quiet about it is claiming more than it earned. On bowel
            # and lung the weakest group is the one with the higher mortality
            # from that cancer, which is the opposite of a harmless gap.
            "fairness_groups": fair.get("groups"),
            "fairness_worst_group": fair.get("worst_group"),
            "fairness_worst_auc": fair.get("worst_auc"),
            "fairness_flagged": fair.get("materially_worse_groups"),
            # What kind of question this panel answers, and whether it is a
            # screening test at all. Four of the eight are not: two need a
            # biopsy or an MRI first, one runs only after a mass is found, and
            # three of the four true screening panels flag so many healthy
            # people per case that acting on them population-wide is not
            # defensible. Carried to the card so a demo cannot imply otherwise.
            "panel_kind": bundle.get("panel_kind", "screening"),
            "panel_kind_note": bundle.get("panel_kind_note", ""),
            # Panels with no confirmatory procedure to send a flagged person to.
            # A limit the reader needs, and one no amount of accuracy repairs.
            "no_action_note": bundle.get("no_action_note", ""),
            "screening_viable": bundle.get("screening_viable", True),
            # How much this panel adds over knowing age and sex. Surfaced
            # because a 0.75 AUC looks respectable right up until you learn
            # that age and sex alone reach 0.75 too.
            "gain_over_age_sex": bundle.get("gain_over_age_sex"),
            "barely_beats_demographics": bundle.get("barely_beats_demographics", False),
            "auc_ci": held.get("auc_ci"),
            "sensitivity": held.get("sensitivity", metrics.get("sensitivity")),
            "sensitivity_ci": held.get("sensitivity_ci"),
            "specificity": held.get("specificity", metrics.get("specificity")),
            "specificity_ci": held.get("specificity_ci"),
            "n_samples": metrics.get("n_samples"),
            "n_test": held.get("n_test"),
            # The number that decides whether this is usable as screening.
            "ppv_at_population_prevalence": held.get("ppv_at_population_prevalence"),
            "people_flagged_per_true_case": held.get("people_flagged_per_true_case"),
            "population_prevalence": held.get("population_prevalence"),
            "prevalence_source": held.get("prevalence_source"),
            "cohort_prevalence": held.get("cohort_prevalence"),
            "baseline_logistic_auc": held.get("baseline_logistic_auc"),
            "cohort_design": bundle.get("cohort_design", ""),
            "algorithm": bundle.get("algorithm", ""),
        }

    if not results:
        # Nothing scored, which in practice means age and sex were the only
        # things supplied. That is not an error the user caused, so it is
        # answered with the reason and the way forward rather than a bare
        # failure message.
        return {
            "status": "success",
            "predictions": {},
            "skipped": skipped,
            "ignored": ignored,
            "values_used": len(values),
            "message": (
                "Not enough data. Nothing was entered that any panel can read, so there is "
                "nothing to score. Enter whatever lab values you have and the relevant "
                "panels will use them."
            ),
        }

    top = max(r["risk"] for r in results.values())
    all_flags = sum(len(r["flags"]) for r in results.values())

    flagged_panels = [k for k, r in results.items() if r.get("above_threshold")]
    baseline = {
        "key": "benign",
        "risk": round(100 - top, 1),
        "band": "Nothing stood out" if not flagged_panels else "Something else stood out",
        "meaning": (
            "None of the panels found anything above the level it treats as worth a "
            "second look. That is reassuring, but it is not a clean bill of health: "
            "many cancers produce completely normal lab results early on."
            if not flagged_panels else
            "At least one panel found something worth a second look, so this "
            "healthy-baseline score is correspondingly lower. Read the flagged "
            "panel above rather than this number."
        ),
        "threshold": None,
        "above_threshold": False,
        "contributors": [],
        "drivers": [],
        "flags": [],
        "inputs_used": len(values),
        "inputs_total": len(values),
        "coverage": 100,
        "missing": [],
        "auc": None,
        "note": (
            f"This is simply 100 minus the highest panel score above. "
            f"{len(flagged_panels)} panel(s) came back raised, and "
            f"{all_flags} individual lab value(s) sit outside their normal range."
        ),
    }
    results["Nothing Flagged"] = baseline

    ordered = dict(sorted(results.items(), key=lambda kv: kv[1]["risk"], reverse=True))

    return {
        "status": "success",
        "predictions": ordered,
        # Panels that did not run, and why. Returned rather than dropped so the
        # interface can say "this needs more information" instead of quietly
        # showing one fewer card, which reads like something broke.
        "skipped": skipped,
        "ignored": ignored,
        "values_used": len(values),
    }


# Printed names for each analyte, longest-first within each entry. Matching is
# word-bounded, and on any given line the longest matching synonym wins, which is
# what keeps "CA 19-9" from being read as calcium and stops "ast" matching inside
# the word "fasting".
BIOMARKER_SYNONYMS = {
    "age": [r"age"],
    "bmi": [r"body mass index", r"bmi"],
    "wbc": [r"white blood cell count", r"white blood cells", r"white blood", r"leukocytes", r"wbc"],
    "rbc": [r"red blood cell count", r"red blood cells", r"red blood", r"erythrocytes", r"rbc"],
    "hemoglobin": [r"haemoglobin", r"hemoglobin", r"hgb", r"hb"],
    # "Platelet Count" is singular on most reports, which the old pattern missed.
    "platelets": [r"platelet count", r"platelets", r"platelet", r"thrombocytes", r"plt"],
    "glucose": [r"glucose, fasting", r"fasting glucose", r"blood glucose", r"glucose", r"glu"],
    # Bare "ca" is last, and safe only because the longest synonym on a line
    # wins: on a CA 19-9 line the longer ca-19-9 pattern matches first.
    "calcium": [r"calcium, total", r"total calcium", r"calcium", r"ca"],
    "bun": [r"blood urea nitrogen", r"urea nitrogen", r"bun", r"urea"],
    "creatinine": [r"creatinine, serum", r"creatinine", r"creat", r"crea"],
    "protein_total": [r"total protein", r"protein, total", r"serum protein", r"protein total", r"tprot", r"tp"],
    "albumin": [r"albumin, serum", r"albumin", r"alb"],
    # SGOT and SGPT come first so the bare three-letter codes are a last resort.
    "ast": [r"aspartate aminotransferase", r"ast \(sgot\)", r"sgot", r"ast"],
    "alt": [r"alanine aminotransferase", r"alt \(sgpt\)", r"sgpt", r"alt"],
    "bilirubin": [r"bilirubin, total", r"total bilirubin", r"bilirubin", r"tbili", r"tbil", r"bili"],
    "alkaline_phosphatase": [r"alkaline phosphatase", r"alk phos", r"alkphos", r"alp"],
    # The hyphenated spelling is the common one and the old pattern missed it.
    "alpha_fetoprotein_level": [r"alpha[- ]?fetoprotein", r"afp"],
    "psa": [r"prostate specific antigen", r"prostate-specific antigen", r"psa"],
    "plasma_ca19_9": [r"ca[ \-_]?19[\-_]?9", r"carbohydrate antigen 19"],
    # Longest-match-wins keeps these away from calcium's bare "ca".
    "ca125": [r"cancer antigen 125", r"ca[ \-_]?125"],
    "cotinine": [r"cotinine, serum", r"serum cotinine", r"cotinine"],
    "crp": [r"c[- ]?reactive protein", r"hs[- ]?crp", r"crp"],
    "prostate_volume": [r"prostate volume", r"prostatic volume"],
    "psa_density": [r"psa density", r"psad"],
    "pi_rads": [r"pi[- ]?rads score", r"pi[- ]?rads", r"pirads"],
    "he4": [r"human epididymis protein 4", r"human epididymis protein", r"he[ \-_]?4"],
    "cea": [r"carcinoembryonic antigen", r"cea"],
    # Word-bounded matching stops "mch" firing inside "mchc", which is a
    # different analyte that appears on the same line block.
    "hematocrit": [r"haematocrit", r"hematocrit", r"hct"],
    "mcv": [r"mean corpuscular volume", r"mcv"],
    "mch": [r"mean corpuscular hemoglobin", r"mean corpuscular haemoglobin", r"mch"],
    "rdw": [r"red cell distribution width", r"rdw[\-_ ]?cv", r"rdw"],
    "mpv": [r"mean platelet volume", r"mpv"],
    "neutrophil_pct": [r"neutrophils?[ ,]*%", r"neutrophil percent", r"neutrophils?", r"neut"],
    "ggt": [r"gamma[- ]?glutamyl transferase", r"gamma[- ]?gt", r"ggtp", r"ggt"],
    "radius_mean": [r"radius mean", r"mean radius"],
    "texture_mean": [r"texture mean", r"mean texture"],
    "perimeter_mean": [r"perimeter mean", r"mean perimeter"],
    "area_mean": [r"area mean", r"mean area"],
}

UNIT_NOISE = re.compile(
    # These were literal backspace bytes rather than word boundaries: a shell
    # escape turned the boundary marks into 0x08 when this file was written, so
    # the pattern could never match and unit text was never actually stripped.
    # These were literal backspace bytes rather than word boundaries: a shell
    # escape turned the boundary marks into 0x08 when this file was written, so
    # the pattern could never match and unit text was never actually stripped.
    r"\b(mg/dl|g/dl|u/l|iu/l|ng/ml|k/ul|m/ul|u/ml|mmol/l|umol/l|pmol/l|g/l|"
    r"mmhg|bpm|fl|pg|ratio)\b|%",
    re.IGNORECASE,
)

# A reference interval, in the forms reports actually print.
#   0.70 - 1.30      4.0-11.0      70 to 99      <1.2      0.2 – 1.2
RANGE_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"[<>]=?\s*\d+(?:\.\d+)?"),
]

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _strip_ranges(text: str) -> str:
    """
    Blank out reference intervals before looking for the patient's value.

    This is the single most important step. When a report prints the expected
    range before the result, a forward scan reaches the range first and returns
    its lower bound, so creatinine 1.05 gets read as 0.70. Removing ranges first
    fixes that, and it also fixes the commoner layout where the range trails the
    result, because in both cases what is left on the line is the value.
    """
    for pat in RANGE_PATTERNS:
        text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _match_analyte(line: str):
    """The analyte this line is about, by longest word-bounded synonym match."""
    best = None
    for key, synonyms in BIOMARKER_SYNONYMS.items():
        for syn in synonyms:
            m = re.search(r"(?<![a-z0-9])(?:" + syn + r")(?![a-z0-9])", line, re.IGNORECASE)
            if m and (best is None or len(m.group(0)) > best[2]):
                best = (key, m.end(), len(m.group(0)))
    return best


def parse_report_text(text: str) -> dict:
    """
    Extract analyte values from lab report text, one line at a time.

    Reports are line-oriented, so the previous approach of collapsing the whole
    document into a single string was the root problem: it let a value from one
    analyte be picked up by another, several lines away. Working per line keeps
    each value with its own analyte, and a one-line lookahead handles the layout
    that prints the value beneath the name.
    """
    found: dict = {}
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]

    for i, raw in enumerate(lines):
        hit = _match_analyte(raw)
        if not hit:
            continue
        key, name_end, _ = hit
        if key in found:
            continue

        # After the analyte name, with ranges and units removed.
        tail = _strip_ranges(UNIT_NOISE.sub(" ", raw[name_end:]))
        nums = NUMBER.findall(tail)

        # Value printed on the following line, as in a stacked single column.
        if not nums and i + 1 < len(lines):
            nxt = lines[i + 1]
            if not _match_analyte(nxt):
                nums = NUMBER.findall(_strip_ranges(UNIT_NOISE.sub(" ", nxt)))

        for candidate in nums:
            try:
                val = float(candidate)
            except ValueError:
                continue
            low, high = BIOLOGICAL_BOUNDS.get(key, (-1e12, 1e12))
            if low <= val <= high:
                found[key] = val
                break

    return found


@app.post("/parse-pdf")
async def parse_pdf(files: List[UploadFile] = File(...)):
    try:
        extracted: dict = {}
        budget = MAX_UPLOAD_BYTES
        for file in files[:MAX_UPLOAD_FILES]:
            contents = await file.read()
            # Size is checked after reading because Starlette streams the body to
            # a spooled temporary file before this runs, so the read itself is
            # bounded by the server's own limits rather than by us. What this
            # prevents is handing an oversized document to pdfplumber, which is
            # where the memory actually goes.
            if len(contents) > budget:
                return {
                    "status": "error",
                    "message": (f"That file is too large. The limit is "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB across "
                                f"all documents in one upload."),
                }
            budget -= len(contents)
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                pages = pdf.pages[:MAX_PDF_PAGES]
                text = "\n".join(p.extract_text() or "" for p in pages)
            for key, val in parse_report_text(text).items():
                extracted.setdefault(key, val)

        if not extracted:
            return {"status": "empty", "message": "No recognised biomarkers found in these documents."}
        return {"status": "success", "data": extracted}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process documents: {str(e)}"}
