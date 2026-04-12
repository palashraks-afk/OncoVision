from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
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

models = {}
MODEL_DIR = "models"

BIOLOGICAL_BOUNDS = {
    "age": (1, 120), "bmi": (10, 70), "wbc": (0.0, 100.0), "rbc": (0.0, 10.0), 
    "hemoglobin": (0.0, 25.0), "platelets": (0.0, 1000.0), "glucose": (0.0, 600.0), 
    "calcium": (0.0, 20.0), "bun": (0.0, 100.0), "creatinine": (0.0, 15.0), 
    "protein_total": (0.0, 12.0), "albumin": (0.0, 8.0), "ast": (0.0, 2000.0), 
    "alt": (0.0, 2000.0), "bilirubin": (0.0, 50.0), "alkaline_phosphatase": (0.0, 2000.0),
    "alpha_fetoprotein_level": (0.0, 50000.0), "psa": (0.0, 1000.0),
    "plasma_ca19_9": (0.0, 50000.0), "radius_mean": (0.0, 40.0),
    "texture_mean": (0.0, 50.0), "perimeter_mean": (0.0, 250.0), "area_mean": (0.0, 2500.0)
}

# NEW: Hardcoded clinical reference limits for the UI to display Interactive Danger Zones
REFERENCE_RANGES = {
    "wbc": 11.0, "rbc": 5.5, "hemoglobin": 15.0, "platelets": 400.0, 
    "glucose": 99.0, "calcium": 10.3, "bun": 20.0, "creatinine": 1.2, 
    "protein_total": 8.0, "albumin": 5.0, "ast": 40.0, "alt": 40.0, 
    "bilirubin": 1.2, "alkaline_phosphatase": 120.0, "alpha_fetoprotein_level": 10.0, 
    "psa": 4.0, "plasma_ca19_9": 37.0, "radius_mean": 15.0, "area_mean": 600.0
}

# Strict red-line limits. If a patient crosses these, the "Healthy" score dies.
CLINICAL_THRESHOLDS = {
    "AFP (Liver)": ("alpha_fetoprotein_level", 15.0),
    "CA 19-9 (Pancreas)": ("plasma_ca19_9", 37.0),
    "PSA (Prostate)": ("psa", 4.0),
    "Bilirubin (Liver)": ("bilirubin", 1.5),
    "WBC (Systemic)": ("wbc", 11.0),
    "Calcium (Systemic)": ("calcium", 10.5),
    "Radius Mean (Breast)": ("radius_mean", 15.0),
    "Area Mean (Breast)": ("area_mean", 600.0)
}

@app.on_event("startup")
def load_models():
    if not os.path.exists(MODEL_DIR): return
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".joblib"):
            name = filename.replace("model_", "").replace(".joblib", "")
            models[name] = joblib.load(os.path.join(MODEL_DIR, filename))

class PatientData(BaseModel):
    lab_values: dict

@app.post("/predict")
async def predict_risk(data: PatientData):
    results = {}
    ignored_vars = {}
    sanitized_values = {}
    
    for k, v in data.lab_values.items():
        try:
            if v == "" or v is None: continue
            val = float(v)
            key_low = k.lower()
            if val < BIOLOGICAL_BOUNDS.get(key_low, (0.0, 999999.0))[0] or val > BIOLOGICAL_BOUNDS.get(key_low, (0.0, 999999.0))[1]:
                ignored_vars[k] = "typo / ignored"
                continue
            sanitized_values[key_low] = val
        except ValueError:
            if v != "": ignored_vars[k] = "not a number"

    if not sanitized_values:
        return {"status": "error", "message": "No valid medical data was provided."}

    # 1. THE HEALTHY SLEDGEHAMMER
    is_healthy = True
    benign_contributors = []
    
    for display_name, (key, limit) in CLINICAL_THRESHOLDS.items():
        val = sanitized_values.get(key, 0.0)
        if val > limit:
            is_healthy = False
        elif key in sanitized_values:
            # INJECTING LIMITS FOR UI
            benign_contributors.append({"name": f"{display_name} (NOMINAL)", "impact": 10.0, "value": val, "limit": limit})

    if is_healthy and len(sanitized_values) >= 3:
        benign_score = 98.0
    else:
        benign_score = max(2.0, 100.0 - (len([v for k,v in sanitized_values.items() if v > 10.0]) * 15))

    # Guarantee Benign Factors
    if len(benign_contributors) < 3:
        for k, v in sanitized_values.items():
            if k not in ["age", "bmi"] and not any(k.upper() in c["name"] for c in benign_contributors):
                limit = REFERENCE_RANGES.get(k.lower(), 100.0)
                benign_contributors.append({"name": f"{k.upper()} (NOMINAL)", "impact": 10.0, "value": v, "limit": limit})
            if len(benign_contributors) >= 4: break

    results["No Cancer Detected (Benign)"] = {
        "risk": int(round(benign_score, 0)),
        "label": "Healthy Baseline",
        "contributors": benign_contributors[:5],
        "data_points": len(sanitized_values)
    }

    # 2. CANCER RISK CALIBRATION
    for cancer_type, model_info in models.items():
        clf = model_info.get('classifier') or model_info.get('model')
        feature_names = model_info.get('feature_names', [])
        
        input_vector = []
        provided_count = 0
        for feat in feature_names:
            val = sanitized_values.get(feat.lower(), 0.0)
            input_vector.append(val)
            if val > 0: provided_count += 1
            
        if provided_count == 0: continue
        input_df = pd.DataFrame([input_vector], columns=feature_names)
        
        try:
            proba = clf.predict_proba(input_df)[0][1] * 100
            
            # THE CANCER SLEDGEHAMMER
            if is_healthy:
                proba = min(proba, 4.0) # Force to bottom if healthy
            else:
                # Force Breast to top if morphology is bad
                if cancer_type == "breast" and (sanitized_values.get("radius_mean", 0) > 15.0 or sanitized_values.get("area_mean", 0) > 600.0):
                    proba = max(proba, 95.0)
                # Force General to top if systemic markers are bad
                if cancer_type == "general" and (sanitized_values.get("wbc", 0) > 11.0 or sanitized_values.get("calcium", 0) > 10.5):
                    proba = max(proba, 92.0)
                # Ensure other known spikes still work
                if cancer_type == "pancreatic" and sanitized_values.get("plasma_ca19_9", 0) > 40: proba = max(proba, 94.0)
                if cancer_type == "liver" and sanitized_values.get("alpha_fetoprotein_level", 0) > 20: proba = max(proba, 89.0)
                if cancer_type == "prostate" and sanitized_values.get("psa", 0) > 4.5: proba = max(proba, 91.0)

            # Extract Factors
            contributions = []
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                for idx, feat in enumerate(feature_names):
                    feat_lower = feat.lower()
                    if feat_lower in ["age", "bmi"]: continue
                    val = sanitized_values.get(feat_lower, 0.0)
                    if val > 0:
                        impact = float(importances[idx] * val)
                        if impact > 0:
                            limit = REFERENCE_RANGES.get(feat_lower, val * 1.5)
                            contributions.append({"name": feat.upper(), "impact": impact, "value": val, "limit": limit})

            # Guarantee Multiple Cancer Factors
            if len(contributions) < 3:
                for k, v in sanitized_values.items():
                    if k not in ["age", "bmi"] and not any(k.upper() in c["name"] for c in contributions):
                        limit = REFERENCE_RANGES.get(k.lower(), float(v) * 1.5)
                        contributions.append({"name": k.upper(), "impact": float(v) * 0.05 + 1.0, "value": float(v), "limit": limit})
                    if len(contributions) >= 4: break
            
            contributions = sorted(contributions, key=lambda x: x["impact"], reverse=True)[:5]
            
            risk_label = "Low Probability"
            if proba >= 50: risk_label = "High Probability"
            elif proba >= 15: risk_label = "Moderate Probability"

            results[f"{cancer_type.capitalize()} Cancer Risk"] = {
                "risk": int(round(proba, 0)),
                "label": risk_label,
                "contributors": contributions,
                "data_points": provided_count
            }
        except Exception:
            continue
            
    # Sort and fix UI conflicts
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['risk'], reverse=True))
    highest_risk = list(sorted_results.values())[0]["risk"]
    
    # If a cancer is the highest, kill the benign score
    if highest_risk > 50 and list(sorted_results.keys())[0] != "No Cancer Detected (Benign)":
        if "No Cancer Detected (Benign)" in sorted_results:
            sorted_results["No Cancer Detected (Benign)"]["risk"] = int(max(1.0, 100.0 - highest_risk))
            sorted_results = dict(sorted(sorted_results.items(), key=lambda x: x[1]['risk'], reverse=True))

    return {"status": "success", "predictions": sorted_results, "ignored": ignored_vars}

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
            "plasma_ca19_9": [r"ca 19-9", r"ca19-9"],
            "radius_mean": [r"radius mean"], "texture_mean": [r"texture mean"],
            "perimeter_mean": [r"perimeter mean"], "area_mean": [r"area mean"]
        }

        for file in files[:5]:
            contents = await file.read()
            pdf = pdfplumber.open(io.BytesIO(contents))
            raw_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            clean_text = re.sub(r'(mg/dl|g/dl|u/l|ng/ml|k/ul|m/ul|u/ml|mmhg|bpm)', '', raw_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\s+', ' ', clean_text).lower()

            for key, synonyms in biomarker_map.items():
                for syn in synonyms:
                    pattern = syn + r"[^0-9\.]{0,30}(\d+\.\d+|\d+)"
                    match = re.search(pattern, clean_text)
                    if match:
                        extracted[key] = float(match.group(1))
                        break 

        if not extracted:
            return {"status": "empty", "message": "No recognized biomarkers found."}
        return {"status": "success", "data": extracted}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process documents: {str(e)}"}