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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    "gender": (0, 1), "smoking": (0, 2), "alcohol_intake": (0, 5),
    "physical_activity": (0, 10), "genetic_risk": (0, 2), "cancer_history": (0, 1),
    "family_history_cancer": (0, 1), "hepatitis_b": (0, 1), "hepatitis_c": (0, 1),
    "cirrhosis_history": (0, 1), "diabetes": (0, 1),
    # Red cell and platelet indices, GGT, and the pelvic-mass tumour markers.
    "hematocrit": (0.0, 70.0), "mcv": (0.0, 150.0), "mch": (0.0, 60.0),
    "rdw": (0.0, 40.0), "mpv": (0.0, 20.0), "neutrophil_pct": (0.0, 100.0),
    "ggt": (0.0, 2000.0), "ca125": (0.0, 50000.0), "he4": (0.0, 50000.0),
    "cea": (0.0, 10000.0),
    # Reproductive and sexual history, read by the cervical panel.
    "menopause": (0, 1), "sexual_partners": (0, 50),
    "first_intercourse_age": (8, 60), "pregnancies": (0, 20),
    "smokes": (0, 1), "smoking_years": (0, 70), "smoking_packyears": (0, 200),
    "hormonal_contraceptives": (0, 1), "hormonal_contraceptives_years": (0, 50),
    "iud": (0, 1), "iud_years": (0, 50),
    "stds": (0, 1), "stds_number": (0, 20), "stds_hpv": (0, 1),
    "stds_diagnoses": (0, 20),
    # Tobacco exposure, inflammation, and the prostate work-up.
    "cotinine": (0.0, 2000.0), "crp": (0.0, 500.0),
    "prostate_volume": (1.0, 300.0), "psa_density": (0.0, 50.0), "pi_rads": (1, 5),
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
    "cirrhosis_history": ("Cirrhosis", ["liver"]),
    "cancer_history": ("Prior cancer diagnosis", ["general"]),
    "family_history_cancer": ("Family history of cancer", ["liver", "general"]),
    "diabetes": ("Diabetes", ["liver", "pancreatic"]),
    "stds_hpv": ("HPV", ["cervical"]),
    "stds": ("Sexually transmitted infection history", ["cervical"]),
    "menopause": ("Post-menopausal", ["ovarian"]),
}

# Coded history answers rendered back into words for the driver breakdown.
CODED_VALUES = {
    "gender": {0: "Female", 1: "Male"},
    "smoking": {0: "Never smoked", 1: "Former smoker", 2: "Current smoker"},
    "genetic_risk": {0: "Low", 1: "Medium", 2: "High"},
    "cancer_history": {0: "No", 1: "Yes"},
    "family_history_cancer": {0: "No", 1: "Yes"},
    "hepatitis_b": {0: "Negative", 1: "Positive"},
    "hepatitis_c": {0: "Negative", 1: "Positive"},
    "cirrhosis_history": {0: "No", 1: "Yes"},
    "diabetes": {0: "No", 1: "Yes"},
    "menopause": {0: "Pre-menopausal", 1: "Post-menopausal"},
    "smokes": {0: "No", 1: "Yes"},
    "hormonal_contraceptives": {0: "No", 1: "Yes"},
    "iud": {0: "No", 1: "Yes"},
    "stds": {0: "No", 1: "Yes"},
    "stds_hpv": {0: "Negative", 1: "Positive"},
}

UNITS = {
    "bmi": "", "wbc": "K/uL", "rbc": "M/uL", "hemoglobin": "g/dL", "platelets": "K/uL",
    "glucose": "mg/dL", "calcium": "mg/dL", "bun": "mg/dL", "creatinine": "mg/dL",
    "protein_total": "g/dL", "albumin": "g/dL", "ast": "U/L", "alt": "U/L",
    "bilirubin": "mg/dL", "alkaline_phosphatase": "U/L",
    "alpha_fetoprotein_level": "ng/mL", "psa": "ng/mL", "plasma_ca19_9": "U/mL",
    "age": "years", "alcohol_intake": "of 5", "physical_activity": "hrs/week",
    "hematocrit": "%", "mcv": "fL", "mch": "pg", "rdw": "%", "mpv": "fL",
    "neutrophil_pct": "%", "ggt": "U/L",
    "ca125": "U/mL", "he4": "pmol/L", "cea": "ng/mL",
    "first_intercourse_age": "years", "smoking_years": "years",
    "smoking_packyears": "pack-years", "hormonal_contraceptives_years": "years",
    "iud_years": "years",
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
    "menopause": "Menopausal status", "sexual_partners": "Sexual partners",
    "first_intercourse_age": "Age at first intercourse",
    "pregnancies": "Pregnancies", "smokes": "Smokes",
    "smoking_years": "Years smoking", "smoking_packyears": "Pack-years",
    "hormonal_contraceptives": "Hormonal contraceptives",
    "hormonal_contraceptives_years": "Years on hormonal contraceptives",
    "iud": "IUD", "iud_years": "Years with an IUD",
    "stds": "STI history", "stds_number": "Number of STIs",
    "stds_hpv": "HPV", "stds_diagnoses": "STI diagnoses",
    "cotinine": "Serum cotinine", "crp": "CRP",
    "prostate_volume": "Prostate volume", "psa_density": "PSA density",
    "pi_rads": "PI-RADS score",
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

    # Patient history
    gender: Optional[float] = None
    smoking: Optional[float] = None
    alcohol_intake: Optional[float] = None
    physical_activity: Optional[float] = None
    genetic_risk: Optional[float] = None
    cancer_history: Optional[float] = None
    family_history_cancer: Optional[float] = None
    hepatitis_b: Optional[float] = None
    hepatitis_c: Optional[float] = None
    cirrhosis_history: Optional[float] = None
    diabetes: Optional[float] = None

    # Reproductive and sexual history. Read by the cervical panel, and
    # menopausal status is also read by the ovarian panel.
    menopause: Optional[float] = None
    sexual_partners: Optional[float] = None
    first_intercourse_age: Optional[float] = None
    pregnancies: Optional[float] = None
    smokes: Optional[float] = None
    smoking_years: Optional[float] = None
    smoking_packyears: Optional[float] = None
    hormonal_contraceptives: Optional[float] = None
    hormonal_contraceptives_years: Optional[float] = None
    iud: Optional[float] = None
    iud_years: Optional[float] = None
    stds: Optional[float] = None
    stds_number: Optional[float] = None
    stds_hpv: Optional[float] = None
    stds_diagnoses: Optional[float] = None


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
        if required_sex is not None and "gender" in values:
            if float(values["gender"]) != float(required_sex):
                skipped[bundle.get("label", name)] = (
                    "Not applicable. This panel only applies to "
                    + ("women." if required_sex == 0 else "men.")
                )
                continue

        row, supplied = [], []
        for feat in features:
            if feat in values:
                row.append(values[feat])
                supplied.append(feat)
            else:
                row.append(medians.get(feat, 0.0))

        if not supplied:
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
            proba = float(bundle["model"].predict_proba(frame)[0][1]) * 100.0
            # Isotonic calibration maps the top bin to exactly 1.0, so an
            # extreme record can come back as a flat 100 percent. No model here
            # has earned certainty: the best panel's confidence interval still
            # runs to 0.993, and reporting 100 would claim a precision the
            # held-out test cannot support. Clipped at both ends for the same
            # reason.
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
        results[bundle.get("label", name)] = {
            "key": name,
            "risk": round(proba, 1),
            "band": band,
            "meaning": meaning,
            "confidence": confidence,
            "coverage_caveat": caveat,
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
        for file in files[:5]:
            contents = await file.read()
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            for key, val in parse_report_text(text).items():
                extracted.setdefault(key, val)

        if not extracted:
            return {"status": "empty", "message": "No recognised biomarkers found in these documents."}
        return {"status": "success", "data": extracted}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process documents: {str(e)}"}
