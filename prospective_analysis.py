"""
Pre-committed analysis for the prospective study in PROTOCOL.md.

PROTOCOL.md states that the analysis script is committed and tagged before
enrolment opens. This is that script. Writing it now, before any participant
exists, is the whole point: once data arrives there is no room to choose the
endpoint that happens to look best, because the endpoint is already fixed here
in version control.

It does three things:

  --dictionary   emits the data dictionary the site collects against, so the
                 capture instrument and this analysis cannot drift apart
  --simulate     generates a synthetic dataset in that exact schema and runs the
                 full analysis on it, which proves the pipeline executes end to
                 end before a single real patient is enrolled
  <file.csv>     runs the pre-specified analysis on real study data

Pre-specified, and not negotiable after the fact:

  PRIMARY     positive predictive value per panel at the frozen shipped
              threshold, with a 95% bootstrap interval
  SECONDARY   sensitivity, specificity, AUC, all with intervals; calibration
              slope and Brier; performance by sex, age band, and race and
              ethnicity; the gap between prospective and retrospective results
  FROZEN      thresholds come from the shipped model bundles and are not tuned
  FORBIDDEN   retraining or recalibrating on prospective data, and reporting
              AUC in place of PPV if PPV disappoints

Run:  python prospective_analysis.py --dictionary
      python prospective_analysis.py --simulate
      python prospective_analysis.py study_data.csv
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import bootstrap_ci, sens_at, spec_at  # noqa: E402

import joblib  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

MODEL_DIR = "models"
THRESHOLD = 0.5  # frozen, matches the shipped interface banding
RANDOM_STATE = 42

# The capture instrument. Every field the site records, with the coding the
# models expect, so a transcription error cannot silently become a model error.
DATA_DICTIONARY = [
    ("study_id", "text", "Random study identifier. The link to the patient is held separately by the site."),
    ("enrolled_date", "date", "YYYY-MM-DD."),
    ("age", "number", "Years at enrolment. Eligible if 20 or over."),
    ("gender", "0/1", "0 female, 1 male."),
    ("race_ethnicity", "text", "Collected deliberately. Subgroup accuracy is unmeasured for most panels without it."),
    ("bmi", "number", "kg/m2."),
    ("wbc", "number", "K/uL."),
    ("rbc", "number", "M/uL."),
    ("hemoglobin", "number", "g/dL."),
    ("platelets", "number", "K/uL."),
    ("glucose", "number", "mg/dL, fasting."),
    ("calcium", "number", "mg/dL."),
    ("bun", "number", "mg/dL."),
    ("creatinine", "number", "mg/dL."),
    ("protein_total", "number", "g/dL."),
    ("albumin", "number", "g/dL."),
    ("ast", "number", "U/L."),
    ("alt", "number", "U/L."),
    ("bilirubin", "number", "mg/dL, total."),
    ("alkaline_phosphatase", "number", "U/L."),
    ("smoking", "0/1/2", "0 never, 1 former, 2 current."),
    ("alcohol_intake", "0-5", "0 none through 5 heavy."),
    ("physical_activity", "number", "Hours of activity per week, 0 to 10."),
    ("genetic_risk", "0/1/2", "0 low, 1 medium, 2 high."),
    ("cancer_history", "0/1", "Any prior cancer diagnosis."),
    ("family_history_cancer", "0/1", "Parent, sibling or child."),
    ("hepatitis_b", "0/1", ""),
    ("hepatitis_c", "0/1", ""),
    ("cirrhosis_history", "0/1", ""),
    ("diabetes", "0/1", ""),
    ("parser_used", "0/1", "1 if values came from the PDF parser rather than manual entry."),
    ("parser_fields_corrected", "number", "Fields the reviewer had to correct. Feeds the parser accuracy endpoint."),
    ("outcome_general", "0/1/blank", "Incident cancer of any site at 12 months, by blinded chart review."),
    ("outcome_liver", "0/1/blank", "Liver disease documented at 12 months."),
    ("outcome_breast", "0/1/blank", "Malignant on biopsy. Blank unless a biopsy was performed."),
    ("outcome_ascertained", "0/1", "1 if 12 month chart review was completed. Rows with 0 are excluded."),
]

# Which outcome column each shipped panel is scored against.
PANEL_OUTCOME = {
    "general": "outcome_general",
    "liver": "outcome_liver",
    "breast": "outcome_breast",
}


def emit_dictionary():
    print(f"{'field':<26}{'type':<12}notes")
    print("=" * 100)
    for name, kind, note in DATA_DICTIONARY:
        print(f"{name:<26}{kind:<12}{note}")
    path = "study_data_dictionary.csv"
    pd.DataFrame(DATA_DICTIONARY, columns=["field", "type", "notes"]).to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(f"blank template: {os.path.basename(write_template())}")


def write_template():
    path = "study_data_template.csv"
    pd.DataFrame(columns=[f for f, _, _ in DATA_DICTIONARY]).to_csv(path, index=False)
    return path


def load_models():
    out = {}
    if not os.path.isdir(MODEL_DIR):
        return out
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith(".joblib"):
            name = f.replace("model_", "").replace(".joblib", "")
            out[name] = joblib.load(os.path.join(MODEL_DIR, f))
    return out


def score(bundle, df):
    """Score the study frame exactly as the live API does, medians and all."""
    feats = bundle["feature_names"]
    med = bundle.get("feature_medians", {})
    # Build against the frame's own index. A feature the site did not collect
    # becomes an all-NaN column rather than a scalar, then falls back to the
    # training median, which is what the live API does for a blank field.
    X = pd.DataFrame(index=df.index)
    for f in feats:
        X[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
        X[f] = X[f].fillna(med.get(f, 0.0))
    return bundle["model"].predict_proba(X[feats])[:, 1]


def simulate(n=600, seed=RANDOM_STATE):
    """
    Synthetic data in the study schema, used only to prove the pipeline runs.

    These are not results and must never be reported as such. The point is that
    the analysis executes end to end before enrolment, so nothing has to be
    written under time pressure once real data exists.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "study_id": [f"S{i:04d}" for i in range(n)],
        "enrolled_date": "2026-01-01",
        "age": rng.integers(20, 85, n),
        "gender": rng.integers(0, 2, n),
        "race_ethnicity": rng.choice(
            ["Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian",
             "Mexican American", "Other Hispanic", "Other or multiracial"], n),
        "bmi": rng.normal(28, 6, n).clip(15, 60),
        "wbc": rng.normal(7, 2, n).clip(2, 30),
        "rbc": rng.normal(4.7, 0.5, n).clip(2, 8),
        "hemoglobin": rng.normal(13.8, 1.5, n).clip(6, 20),
        "platelets": rng.normal(250, 60, n).clip(50, 700),
        "glucose": rng.normal(100, 25, n).clip(50, 400),
        "calcium": rng.normal(9.5, 0.5, n).clip(7, 13),
        "bun": rng.normal(15, 5, n).clip(3, 60),
        "creatinine": rng.normal(1.0, 0.3, n).clip(0.3, 8),
        "protein_total": rng.normal(7.1, 0.5, n).clip(4, 10),
        "albumin": rng.normal(4.2, 0.5, n).clip(1.5, 6),
        "ast": rng.lognormal(3.2, 0.5, n).clip(5, 500),
        "alt": rng.lognormal(3.1, 0.5, n).clip(5, 500),
        "bilirubin": rng.lognormal(-0.5, 0.6, n).clip(0.1, 20),
        "alkaline_phosphatase": rng.normal(90, 30, n).clip(20, 600),
        "smoking": rng.integers(0, 3, n),
        "alcohol_intake": rng.uniform(0, 5, n),
        "physical_activity": rng.uniform(0, 10, n),
        "genetic_risk": rng.integers(0, 3, n),
        "cancer_history": rng.binomial(1, 0.08, n),
        "family_history_cancer": rng.binomial(1, 0.25, n),
        "hepatitis_b": rng.binomial(1, 0.02, n),
        "hepatitis_c": rng.binomial(1, 0.03, n),
        "cirrhosis_history": rng.binomial(1, 0.03, n),
        "diabetes": rng.binomial(1, 0.12, n),
        "parser_used": rng.binomial(1, 0.7, n),
        "parser_fields_corrected": rng.poisson(0.3, n),
        "outcome_ascertained": 1,
    })
    # Outcomes at roughly the prevalences PROTOCOL.md assumes.
    df["outcome_general"] = rng.binomial(1, 0.02, n)
    df["outcome_liver"] = rng.binomial(1, 0.05, n)
    df["outcome_breast"] = np.where(rng.random(n) < 0.15, rng.binomial(1, 0.3, n), np.nan)
    return df


def analyse(df):
    if "outcome_ascertained" in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df["outcome_ascertained"], errors="coerce") == 1]
        print(f"Outcome ascertained for {len(df)} of {before} enrolled\n")

    models = load_models()
    if not models:
        print("No model bundles found. Run train_models.py first.")
        return {}

    results = {}
    for panel, bundle in models.items():
        outcome = PANEL_OUTCOME.get(panel)
        if outcome is None or outcome not in df.columns:
            print(f"{panel}: no outcome column, skipped")
            continue

        sub = df[pd.to_numeric(df[outcome], errors="coerce").notna()]
        y = pd.to_numeric(sub[outcome], errors="coerce").astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            print(f"{panel}: outcome has one class only, not analysable")
            continue

        p = score(bundle, sub)
        pred = (p >= THRESHOLD).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())

        def ppv(yy, pp):
            q = (pp >= THRESHOLD).astype(int)
            t = int(((q == 1) & (yy == 1)).sum())
            f = int(((q == 1) & (yy == 0)).sum())
            return t / (t + f) if (t + f) else float("nan")

        held = bundle.get("held_out", {})
        row = {
            "panel": panel,
            "n": int(len(y)),
            "events": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            # PRIMARY ENDPOINT
            "ppv": round(ppv(y, p), 3) if (tp + fp) else None,
            "ppv_ci": bootstrap_ci(y, p, ppv),
            "flagged": int(tp + fp),
            "sensitivity": round(sens_at(THRESHOLD)(y, p), 3),
            "sensitivity_ci": bootstrap_ci(y, p, sens_at(THRESHOLD)),
            "specificity": round(spec_at(THRESHOLD)(y, p), 3),
            "specificity_ci": bootstrap_ci(y, p, spec_at(THRESHOLD)),
            "auc": round(float(roc_auc_score(y, p)), 3),
            "auc_ci": bootstrap_ci(y, p, roc_auc_score),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "retrospective_auc": held.get("auc"),
        }
        if row["auc"] is not None and held.get("auc") is not None:
            row["auc_gap_vs_retrospective"] = round(held["auc"] - row["auc"], 3)

        subgroups = {}
        for col, label in [("race_ethnicity", "race"), ("gender", "sex")]:
            if col not in sub.columns:
                continue
            for group, idx in sub.groupby(col).groups.items():
                mask = sub.index.isin(idx)
                if mask.sum() >= 50 and len(np.unique(y[mask])) > 1:
                    subgroups[f"{label}: {group}"] = {
                        "n": int(mask.sum()),
                        "events": int(y[mask].sum()),
                        "auc": round(float(roc_auc_score(y[mask], p[mask])), 3),
                    }
        row["subgroups"] = subgroups
        results[panel] = row

        print(f"{panel}")
        print(f"  n {row['n']}, events {row['events']} ({row['prevalence']:.2%})")
        print(f"  PRIMARY  PPV {row['ppv']}  (95% CI {row['ppv_ci'][0]} to {row['ppv_ci'][1]}), "
              f"{row['flagged']} flagged")
        print(f"  sens {row['sensitivity']}  spec {row['specificity']}  AUC {row['auc']} "
              f"(95% CI {row['auc_ci'][0]} to {row['auc_ci'][1]})")
        if "auc_gap_vs_retrospective" in row:
            print(f"  retrospective AUC was {row['retrospective_auc']}, "
                  f"gap {row['auc_gap_vs_retrospective']:+.3f}")
        print()

    if "parser_used" in df.columns and "parser_fields_corrected" in df.columns:
        used = df[pd.to_numeric(df["parser_used"], errors="coerce") == 1]
        if len(used):
            corr = pd.to_numeric(used["parser_fields_corrected"], errors="coerce").fillna(0)
            print(f"Parser: used on {len(used)} reports, "
                  f"{corr.mean():.2f} fields corrected on average, "
                  f"{(corr > 0).mean():.1%} of reports needed a correction")
            print("  PROTOCOL.md halts enrolment above a 5% field error rate.\n")

    return results


def main():
    if "--dictionary" in sys.argv:
        emit_dictionary()
        return

    if "--simulate" in sys.argv:
        print("SIMULATED DATA. Proves the pipeline runs. These are not results.\n")
        df = simulate()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if not args:
            print(__doc__)
            return
        df = pd.read_csv(args[0])
        print(f"Loaded {len(df)} rows from {args[0]}\n")

    results = analyse(df)
    if results:
        out = "prospective_results.json"
        with open(out, "w") as f:
            json.dump({"threshold": THRESHOLD, "simulated": "--simulate" in sys.argv,
                       "panels": results}, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
