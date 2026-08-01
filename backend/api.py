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
}

# Upper limit of the normal reference range for each measurable marker.
REFERENCE_RANGES = {
    "wbc": 11.0, "rbc": 5.5, "hemoglobin": 15.0, "platelets": 400.0,
    "glucose": 99.0, "calcium": 10.3, "bun": 20.0, "creatinine": 1.2,
    "protein_total": 8.0, "albumin": 5.0, "ast": 40.0, "alt": 40.0,
    "bilirubin": 1.2, "alkaline_phosphatase": 120.0, "alpha_fetoprotein_level": 10.0,
    "psa": 4.0, "plasma_ca19_9": 37.0, "radius_mean": 15.0, "texture_mean": 22.0,
    "perimeter_mean": 95.0, "area_mean": 600.0, "bmi": 25.0,
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
}

# History answers worth surfacing when they are positive.
HISTORY_FLAGS = {
    "hepatitis_b": ("Hepatitis B", ["liver"]),
    "hepatitis_c": ("Hepatitis C", ["liver"]),
    "cirrhosis_history": ("Cirrhosis", ["liver"]),
    "cancer_history": ("Prior cancer diagnosis", ["general"]),
    "family_history_cancer": ("Family history of cancer", ["liver", "general"]),
    "diabetes": ("Diabetes", ["liver", "pancreatic"]),
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
}

UNITS = {
    "bmi": "", "wbc": "K/uL", "rbc": "M/uL", "hemoglobin": "g/dL", "platelets": "K/uL",
    "glucose": "mg/dL", "calcium": "mg/dL", "bun": "mg/dL", "creatinine": "mg/dL",
    "protein_total": "g/dL", "albumin": "g/dL", "ast": "U/L", "alt": "U/L",
    "bilirubin": "mg/dL", "alkaline_phosphatase": "U/L",
    "alpha_fetoprotein_level": "ng/mL", "psa": "ng/mL", "plasma_ca19_9": "U/mL",
    "age": "years", "alcohol_intake": "of 5", "physical_activity": "hrs/week",
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


def ensemble_importances(bundle) -> np.ndarray:
    """Mean feature importance across the two members of the voting ensemble."""
    model = bundle["model"]
    parts = []
    for est in getattr(model, "estimators_", []):
        imp = getattr(est, "feature_importances_", None)
        if imp is not None:
            total = float(np.sum(imp))
            if total > 0:
                parts.append(np.asarray(imp, dtype=float) / total)
    if not parts:
        n = len(bundle["feature_names"])
        return np.full(n, 1.0 / n)
    return np.mean(parts, axis=0)


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

    # Tumour markers
    alpha_fetoprotein_level: Optional[float] = None
    psa: Optional[float] = None
    plasma_ca19_9: Optional[float] = None

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
    return {"status": "healthy" if models else "degraded", "models_loaded": len(models)}


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
                "metrics": b.get("metrics", {}),
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
        return {"status": "error", "message": "No usable values were provided."}

    results = {}

    for name, bundle in models.items():
        features = bundle["feature_names"]
        medians = bundle.get("feature_medians", {})

        row, supplied = [], []
        for feat in features:
            if feat in values:
                row.append(values[feat])
                supplied.append(feat)
            else:
                row.append(medians.get(feat, 0.0))

        if not supplied:
            continue

        try:
            frame = pd.DataFrame([row], columns=features)
            proba = float(bundle["model"].predict_proba(frame)[0][1]) * 100.0
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

        if proba >= 50:
            band = "High"
        elif proba >= 20:
            band = "Moderate"
        else:
            band = "Low"

        metrics = bundle.get("metrics", {})
        results[bundle.get("label", name)] = {
            "key": name,
            "risk": int(round(proba)),
            "band": band,
            "contributors": contributors[:6],
            "drivers": drivers[:8],
            "flags": flags,
            "inputs_used": len(supplied),
            "inputs_total": len(features),
            "coverage": round(len(supplied) / len(features) * 100),
            "missing": [pretty(f) for f in features if f not in values],
            "auc": metrics.get("auc"),
            "auc_std": metrics.get("auc_std"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "n_samples": metrics.get("n_samples"),
            "algorithm": bundle.get("algorithm", ""),
        }

    if not results:
        return {"status": "error", "message": "No model could score the values provided."}

    top = max(r["risk"] for r in results.values())
    all_flags = sum(len(r["flags"]) for r in results.values())

    baseline = {
        "key": "benign",
        "risk": int(round(100 - top)),
        "band": "High" if (100 - top) >= 50 else "Low",
        "contributors": [],
        "drivers": [],
        "flags": [],
        "inputs_used": len(values),
        "inputs_total": len(values),
        "coverage": 100,
        "missing": [],
        "auc": None,
        "note": (
            "Complement of the highest domain score. "
            f"{all_flags} clinical threshold(s) crossed across all panels."
        ),
    }
    results["No Cancer Detected (Benign)"] = baseline

    ordered = dict(sorted(results.items(), key=lambda kv: kv[1]["risk"], reverse=True))

    return {
        "status": "success",
        "predictions": ordered,
        "ignored": ignored,
        "values_used": len(values),
    }


@app.post("/parse-pdf")
async def parse_pdf(files: List[UploadFile] = File(...)):
    try:
        extracted = {}
        biomarker_map = {
            "age": [r"age"], "bmi": [r"bmi", r"body mass index"],
            "wbc": [r"wbc", r"white blood"], "rbc": [r"rbc", r"red blood"],
            "hemoglobin": [r"hemoglobin", r"hgb", r"hct"],
            "platelets": [r"platelets", r"plt"], "glucose": [r"glucose", r"glu"],
            "calcium": [r"calcium", r"ca"], "bun": [r"bun", r"urea nitrogen"],
            "creatinine": [r"creatinine", r"creat"], "protein_total": [r"total protein"],
            "albumin": [r"albumin", r"alb"], "ast": [r"ast", r"sgot"],
            "alt": [r"alt", r"sgpt"], "bilirubin": [r"bilirubin", r"bili"],
            "alkaline_phosphatase": [r"alkaline phosphatase", r"alp"],
            "alpha_fetoprotein_level": [r"afp", r"alpha fetoprotein"],
            "psa": [r"psa", r"prostate specific"],
            "plasma_ca19_9": [r"ca 19-9", r"ca19-9", r"ca19_9"],
            "radius_mean": [r"radius mean"], "texture_mean": [r"texture mean"],
            "perimeter_mean": [r"perimeter mean"], "area_mean": [r"area mean"],
        }

        for file in files[:5]:
            contents = await file.read()
            pdf = pdfplumber.open(io.BytesIO(contents))
            raw_text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            clean = re.sub(r'(mg/dl|g/dl|u/l|ng/ml|k/ul|m/ul|u/ml|mmhg|bpm)', '', raw_text, flags=re.IGNORECASE)
            clean = re.sub(r'\s+', ' ', clean).lower()

            for key, synonyms in biomarker_map.items():
                if key in extracted:
                    continue
                for syn in synonyms:
                    match = re.search(syn + r"[^0-9\.]{0,30}(\d+\.\d+|\d+)", clean)
                    if match:
                        val = float(match.group(1))
                        low, high = BIOLOGICAL_BOUNDS.get(key, (-1e12, 1e12))
                        if low <= val <= high:
                            extracted[key] = val
                        break

        if not extracted:
            return {"status": "empty", "message": "No recognised biomarkers found in these documents."}
        return {"status": "success", "data": extracted}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process documents: {str(e)}"}
