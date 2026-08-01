"""
Oncovision model training pipeline.

Trains one soft-voting ensemble (XGBoost + Extra Trees) per cancer domain and
writes it to /models and /backend/models as a joblib bundle.

Design rule: a model may only be trained on features the application can
actually collect from a patient. Training on columns the app never supplies
would produce an AUC that does not describe how the model performs in use, so
every dataset column is either mapped onto the app's canonical input schema or
dropped. Each bundle therefore carries its own feature list, the training
median for every feature (used to impute values the patient did not provide)
and its measured cross-validated performance.

Run:  python train_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

DATA_DIR = "data"
MODEL_DIRS = ["models", os.path.join("backend", "models")]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Canonical application input schema.
# Every model feature must be one of these keys.
# ---------------------------------------------------------------------------
# Lab values, entered directly or parsed out of a PDF report.
LAB_FIELDS = [
    "age", "bmi", "wbc", "rbc", "hemoglobin", "platelets", "glucose", "calcium",
    "bun", "creatinine", "protein_total", "albumin", "ast", "alt", "bilirubin",
    "alkaline_phosphatase", "alpha_fetoprotein_level", "psa", "plasma_ca19_9",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
]

# Patient history, answered by the user rather than read off a lab report.
# gender             0 female, 1 male
# smoking            0 never, 1 former, 2 current
# alcohol_intake     0 none through 5 heavy
# physical_activity  hours of exercise per week, 0 to 10
# genetic_risk       0 low, 1 medium, 2 high
# remaining flags    0 no, 1 yes
HISTORY_FIELDS = [
    "gender", "smoking", "alcohol_intake", "physical_activity", "genetic_risk",
    "cancer_history", "family_history_cancer", "hepatitis_b", "hepatitis_c",
    "cirrhosis_history", "diabetes",
]

APP_FIELDS = LAB_FIELDS + HISTORY_FIELDS


# ---------------------------------------------------------------------------
# Per dataset column translation into the canonical schema
# ---------------------------------------------------------------------------
def _map(series, table, default=np.nan):
    return series.astype(str).str.strip().map(table).fillna(default)


DATASETS = [
    {
        "name": "general",
        "file": "The_Cancer_data_1500_V2.csv",
        "label": "General Cancer Risk",
        # Gender is documented as 0 male / 1 female, inverted to the app's convention.
        "features": {
            "age": lambda d: d["Age"],
            "bmi": lambda d: d["BMI"],
            "gender": lambda d: 1 - d["Gender"],
            "smoking": lambda d: d["Smoking"] * 2,
            "genetic_risk": lambda d: d["GeneticRisk"],
            "physical_activity": lambda d: d["PhysicalActivity"],
            "alcohol_intake": lambda d: d["AlcoholIntake"],
            "cancer_history": lambda d: d["CancerHistory"],
        },
        "target": lambda d: d["Diagnosis"].astype(int),
        "positive_means": "a recorded cancer diagnosis",
    },
    {
        "name": "breast",
        "file": "data.csv",
        "label": "Breast Cancer Risk",
        # Only the four nuclear morphology means the app collects. The other 26
        # columns in the Wisconsin set are deliberately left out.
        "features": {
            "radius_mean": lambda d: d["radius_mean"],
            "texture_mean": lambda d: d["texture_mean"],
            "perimeter_mean": lambda d: d["perimeter_mean"],
            "area_mean": lambda d: d["area_mean"],
        },
        "target": lambda d: (d["diagnosis"].astype(str).str.upper() == "M").astype(int),
        "positive_means": "a malignant fine needle aspirate",
    },
    {
        "name": "liver",
        "file": "synthetic_liver_cancer_dataset.csv",
        "label": "Liver Cancer Risk",
        # liver_function_score is excluded on purpose: it is a composite the
        # dataset generates and no patient can supply it, so training on it
        # would inflate the reported AUC relative to real use.
        "features": {
            "age": lambda d: d["age"],
            "bmi": lambda d: d["bmi"],
            "alpha_fetoprotein_level": lambda d: d["alpha_fetoprotein_level"],
            "gender": lambda d: _map(d["gender"], {"Male": 1, "Female": 0}),
            "smoking": lambda d: _map(d["smoking_status"], {"Never": 0, "Former": 1, "Current": 2}),
            "alcohol_intake": lambda d: _map(d["alcohol_consumption"], {"Never": 0.0, "Occasional": 1.5, "Regular": 3.5, "Heavy": 5.0}),
            "physical_activity": lambda d: _map(d["physical_activity_level"], {"Low": 2.0, "Moderate": 5.0, "High": 8.0}),
            "hepatitis_b": lambda d: d["hepatitis_b"],
            "hepatitis_c": lambda d: d["hepatitis_c"],
            "cirrhosis_history": lambda d: d["cirrhosis_history"],
            "family_history_cancer": lambda d: d["family_history_cancer"],
            "diabetes": lambda d: d["diabetes"],
        },
        "target": lambda d: d["liver_cancer"].astype(int),
        "positive_means": "a liver cancer diagnosis",
    },
    {
        "name": "pancreatic",
        "file": "pancreatic_600_risk123.csv",
        "label": "Pancreatic Cancer Risk",
        # diagnosis is three class: 1 control, 2 benign hepatobiliary disease,
        # 3 pancreatic ductal adenocarcinoma. Only class 3 is cancer.
        # The urinary panel (LYVE1, REG1B, TFF1, REG1A), CEA and the urine
        # chemistry are dropped because the app does not collect them, and
        # 'stage' is dropped because it is only known after diagnosis.
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: _map(d["sex"], {"M": 1, "F": 0}),
            "creatinine": lambda d: d["creatinine"],
            "plasma_ca19_9": lambda d: d["CA19_9"],
            "bilirubin": lambda d: d["bilirubin"],
            "glucose": lambda d: d["glucose"],
        },
        "target": lambda d: (d["diagnosis"].astype(int) == 3).astype(int),
        "positive_means": "pancreatic ductal adenocarcinoma, separated from both healthy controls and benign hepatobiliary disease",
    },
    {
        "name": "prostate",
        "file": "prostate.csv",
        "label": "Prostate Cancer Risk",
        # lpsa is the natural log of PSA, converted back to the ng/mL the app
        # collects. The remaining columns (tumour volume, capsular penetration,
        # seminal vesicle invasion) are surgical findings, not screening inputs.
        "features": {
            "age": lambda d: d["age"],
            "psa": lambda d: np.exp(d["lpsa"]),
        },
        "target": lambda d: (d["gleason"] >= 7).astype(int),
        "positive_means": "a Gleason score of 7 or above, the threshold for clinically significant disease",
    },
]


def build_ensemble(n_samples: int, pos_rate: float) -> VotingClassifier:
    """Soft-voting ensemble of XGBoost and Extra Trees, sized to the dataset."""
    small = n_samples < 300
    scale_pos = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0

    xgb = XGBClassifier(
        n_estimators=200 if small else 400,
        max_depth=3 if small else 4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        min_child_weight=2,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ext = ExtraTreesClassifier(
        n_estimators=300 if small else 600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgboost", xgb), ("extra_trees", ext)],
        voting="soft",
        weights=[1, 1],
    )


def evaluate(clf_factory, X: pd.DataFrame, y: pd.Series) -> dict:
    """Stratified 5 fold cross validated AUC plus a held out confusion matrix."""
    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=RANDOM_STATE)

    oof = cross_val_predict(clf_factory(), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    cv_auc = roc_auc_score(y, oof)

    # Per fold spread, so the number carries an honest error bar.
    fold_aucs = []
    for tr, te in cv.split(X, y):
        m = clf_factory().fit(X.iloc[tr], y.iloc[tr])
        fold_aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    holdout = clf_factory().fit(X_tr, y_tr)
    proba = holdout.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    return {
        "auc": round(float(cv_auc), 3),
        "auc_std": round(float(np.std(fold_aucs)), 3),
        "accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 3),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 3),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_features": int(X.shape[1]),
        "cv_folds": int(cv.get_n_splits()),
    }


def prepare(config: dict):
    path = os.path.join(DATA_DIR, config["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    X = pd.DataFrame({key: fn(df) for key, fn in config["features"].items()})
    X = X.apply(pd.to_numeric, errors="coerce")

    unknown = [c for c in X.columns if c not in APP_FIELDS]
    if unknown:
        raise ValueError(f"{config['name']}: features outside the app schema: {unknown}")

    y = config["target"](df)

    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    return X, y, medians


def main():
    for d in MODEL_DIRS:
        os.makedirs(d, exist_ok=True)

    summary = {}

    for config in DATASETS:
        name = config["name"]
        print(f"\n=== {name} ===")
        try:
            X, y, medians = prepare(config)
        except FileNotFoundError as e:
            print(f"  skipped, data file missing: {e}")
            continue

        pos_rate = float(y.mean())
        print(f"  {len(y)} rows, {int(y.sum())} positive ({pos_rate:.1%}), "
              f"{X.shape[1]} features: {', '.join(X.columns)}")

        factory = lambda: build_ensemble(len(y), pos_rate)
        metrics = evaluate(factory, X, y)
        print(f"  AUC {metrics['auc']} +/- {metrics['auc_std']}   "
              f"acc {metrics['accuracy']}   sens {metrics['sensitivity']}   "
              f"spec {metrics['specificity']}")

        model = factory().fit(X, y)

        bundle = {
            "model": model,
            "feature_names": list(X.columns),
            "feature_medians": {k: float(v) for k, v in medians.items()},
            "config_name": name,
            "label": config["label"],
            "positive_means": config["positive_means"],
            "metrics": metrics,
            "algorithm": "Soft-voting ensemble: XGBoost + Extra Trees",
        }
        # compress=3 keeps the bundles small enough to commit and deploy.
        for d in MODEL_DIRS:
            joblib.dump(bundle, os.path.join(d, f"model_{name}.joblib"), compress=3)
        print(f"  saved to {' and '.join(MODEL_DIRS)}")

        summary[name] = {
            "label": config["label"],
            "features": list(X.columns),
            **metrics,
        }

    with open(os.path.join("backend", "model_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote backend/model_metrics.json")

    print("\n" + "=" * 68)
    print(f"{'model':<13}{'AUC':<16}{'acc':<8}{'sens':<8}{'spec':<8}{'n':<7}{'feats'}")
    print("=" * 68)
    for name, m in summary.items():
        print(f"{name:<13}{str(m['auc']) + ' +/- ' + str(m['auc_std']):<16}"
              f"{m['accuracy']:<8}{m['sensitivity']:<8}{m['specificity']:<8}"
              f"{m['n_samples']:<7}{m['n_features']}")


if __name__ == "__main__":
    main()
