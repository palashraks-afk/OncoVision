"""
Train RandomForestClassifier models for cancer risk detection.
Reads Excel files from /data, trains one model per dataset, saves to /models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Excel files in /data and their config: (filename, target_column, drop_columns, optional preprocess)
DATA_DIR = "data"
MODELS_DIR = "models"

DATASET_CONFIG = [
    {
        "file": "The_Cancer_data_1500_V2.xlsx",
        "name": "general",
        "target": "Diagnosis",
        "drop": [],
    },
    {
        "file": "pancreatic_600_risk123.xlsx",
        "name": "pancreatic",
        "target": "diagnosis",
        "drop": ["sample_id", "patient_cc", "sample_o"],
    },
    {
        "file": "prostate.xlsx",
        "name": "prostate",
        "target": None,  # derived from gleason
        "drop": ["train", "lpsa"],
    },
    {
        "file": "synthetic_liver_cancer_dataset.xlsx",
        "name": "liver",
        "target": "liver_cancer",
        "drop": [],
    },
    {
        "file": "data.xlsx",
        "name": "breast",
        "target": "diagnosis",
        "drop": ["id", "Unnamed: 32"],
    },
]


def prepare_prostate_target(df: pd.DataFrame) -> pd.Series:
    """Binary risk from gleason: 1 if gleason >= 7 else 0."""
    if "gleason" not in df.columns:
        raise ValueError("Prostate data must contain 'gleason' column")
    return (df["gleason"] >= 7).astype(int)


def prepare_breast_target(series: pd.Series) -> pd.Series:
    """Encode M=1, B=0."""
    return (series.str.upper() == "M").astype(int)


def load_data(path: str) -> pd.DataFrame:
    """Load Excel file using pandas and openpyxl; fallback to CSV if not found."""
    if path.endswith(".xlsx") and os.path.isfile(path):
        return pd.read_excel(path, engine="openpyxl")
    csv_path = path.replace(".xlsx", ".csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Data file not found: {path} or {csv_path}")


def prepare_data(config: dict):
    # Returns (X, y, feature_names)
    """Load Excel, build X and y, return (X, y, feature_names)."""
    filepath = os.path.join(DATA_DIR, config["file"])
    df = load_data(filepath)

    # Drop specified columns
    drop = [c for c in config["drop"] if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    # Remove fully empty columns
    df = df.dropna(axis=1, how="all")

    if config["name"] == "prostate":
        y = prepare_prostate_target(df)
        # Features: all except gleason (and already dropped)
        feat_cols = [c for c in df.columns if c != "gleason"]
    else:
        target_col = config["target"]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in {list(df.columns)}")
        y_raw = df[target_col]
        if config["name"] == "breast":
            y = prepare_breast_target(y_raw.astype(str))
        else:
            y = pd.to_numeric(y_raw, errors="coerce").dropna()
            if y.isna().any():
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y_raw.astype(str).fillna("")), index=y_raw.index)
            else:
                y = y.astype(int)
        feat_cols = [c for c in df.columns if c != target_col]

    X = df[feat_cols].copy()

    # Encode object/string columns to numeric
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna("_nan_"))

    # Align X and y (drop rows where y is NaN)
    valid = y.notna()
    if not valid.all():
        y = y[valid]
        X = X.loc[valid]

    # Impute missing numeric values
    if X.isna().any().any():
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X, y, list(X.columns)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for config in DATASET_CONFIG:
        name = config["name"]
        print(f"Training {name} model from {config['file']}...")
        try:
            X, y, feature_names = prepare_data(config)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print(f"  {name}: test accuracy = {score:.4f}")

            out_path = os.path.join(MODELS_DIR, f"model_{name}.joblib")
            joblib.dump(
                {
                    "model": clf,
                    "feature_names": feature_names,
                    "config_name": name,
                },
                out_path,
            )
            print(f"  Saved to {out_path}")
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
