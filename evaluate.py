"""
Rigorous evaluation of the Oncovision models.

This exists because cross-validated AUC on its own is close to meaningless for a
screening claim. It answers the questions that actually decide whether a model
is usable:

  1. Held-out test set.       A 20% split is cut once, before anything is fitted,
                              and is not touched until the final measurement.
                              Model selection and calibration happen inside the
                              remaining 80% only.
  2. Confidence intervals.    Bootstrap percentile intervals on every operating
                              point metric, not just AUC. A sensitivity of 0.79
                              on 114 test rows is worth very little without one.
  3. Calibration.             The interface shows people a percentage, so the
                              percentages have to mean something. Brier score,
                              calibration slope and intercept, and a reliability
                              curve, before and after isotonic calibration.
  4. Baselines.               An ensemble is only worth its complexity if it
                              beats logistic regression and beats age and sex
                              alone. Both are measured.
  5. PPV and NPV at real      Case-control cohorts are enriched for disease. The
     population prevalence.   only honest way to report precision is to project
                              it onto the actual incidence of the disease, taken
                              from SEER. This is the number that decides whether
                              a screening tool is usable, and for most of these
                              panels it is brutal.
  6. Subgroup breakdown.      AUC by sex and by age band, so "accuracy varies
                              between groups" stops being a disclaimer and
                              becomes a measurement.

Writes evaluation.json and EVALUATION.md.

Run:  python evaluate.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_models as tm

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
rng = np.random.default_rng(RANDOM_STATE)

# Annual incidence per 100,000 from SEER, 2019 to 2023, age adjusted.
# Used as the prior for projecting precision onto a screening population.
SEER_INCIDENCE = {
    "general":    (450.7, "Cancer of any site, per 100,000 men and women per year"),
    "breast":     (132.5, "Female breast, per 100,000 women per year"),
    "prostate":   (123.2, "Prostate, per 100,000 men per year"),
    "pancreatic": (13.9,  "Pancreas, per 100,000 men and women per year"),
    # The liver panel detects liver disease, not liver cancer, so it is scored
    # against cirrhosis prevalence in US adults (3.1%, NHANES) rather than
    # liver cancer incidence. Using the cancer figure would understate the
    # panel's precision by roughly 300 times.
    "liver":      (3100.0, "Cirrhosis in US adults, NHANES, per 100,000"),
}

COHORT_DESIGN = {
    "general":    "Risk-factor cohort. Not a consecutive screening series.",
    "breast":     "Case-control, post-biopsy. Every record is an FNA already taken because a lesion was found.",
    "liver":      "583 real patients, India. Externally validated on 589 independent patients from Germany.",
    "pancreatic": "Case-control. Cases are confirmed PDAC, controls include benign hepatobiliary disease.",
    "prostate":   "Case-control, post-prostatectomy. Gleason grade comes from the surgical specimen.",
}


# ---------------------------------------------------------------- statistics
def bootstrap_ci(y, p, fn, n=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap CI for any metric fn(y_true, y_score)."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        try:
            out.append(fn(y[s], p[s]))
        except Exception:
            continue
    if not out:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 3), round(float(hi), 3))


def sens_at(threshold):
    def f(y, p):
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn_ = int(((pred == 0) & (y == 1)).sum())
        return tp / (tp + fn_) if (tp + fn_) else float("nan")
    return f


def spec_at(threshold):
    def f(y, p):
        pred = (p >= threshold).astype(int)
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        return tn / (tn + fp) if (tn + fp) else float("nan")
    return f


def projected_ppv_npv(sens, spec, prevalence):
    """Bayes. What precision looks like once the disease is actually rare."""
    if any(map(lambda v: v is None or np.isnan(v), [sens, spec])):
        return float("nan"), float("nan"), float("nan")
    tp = sens * prevalence
    fp = (1 - spec) * (1 - prevalence)
    fn_ = (1 - sens) * prevalence
    tn = spec * (1 - prevalence)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn_) if (tn + fn_) else float("nan")
    nnt = 1 / ppv if ppv else float("nan")  # people flagged per true case
    return ppv, npv, nnt


def calibration_slope_intercept(y, p):
    """
    Regress the outcome on the logit of the predicted probability.
    Slope 1 and intercept 0 is perfect. Slope below 1 means over-confident.
    """
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    try:
        slope = float(LogisticRegression(penalty=None, max_iter=2000).fit(logit, y).coef_[0][0])
        # Intercept with slope held at 1 (calibration-in-the-large).
        offset = LogisticRegression(penalty=None, max_iter=2000)
        offset.fit(np.zeros((len(y), 1)), y)
        intercept = float(offset.intercept_[0] - np.mean(logit))
    except Exception:
        slope, intercept = float("nan"), float("nan")
    return round(slope, 3), round(intercept, 3)


def reliability_curve(y, p, bins=5):
    """Observed frequency against predicted probability, for a reliability plot."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[i]:.1f} to {edges[i+1]:.1f}",
            "n": int(m.sum()),
            "predicted": round(float(p[m].mean()), 3),
            "observed": round(float(y[m].mean()), 3),
        })
    return rows


# ---------------------------------------------------------------- evaluation
def evaluate_domain(config):
    name = config["name"]
    X, y, _ = tm.prepare(config)
    prevalence_cohort = float(y.mean())

    # The test split is cut once, here, and is not seen again until the end.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    factory = lambda: tm.build_ensemble(len(y_tr), float(y_tr.mean()))

    # --- cross validated on the training portion only ------------------------
    folds = max(2, min(5, int(y_tr.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(factory(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_tr, oof)

    # --- fitted on train, measured on the untouched test set -----------------
    raw = factory().fit(X_tr, y_tr)
    p_raw = raw.predict_proba(X_te)[:, 1]

    # Isotonic calibration, fitted inside the training portion only.
    cal = CalibratedClassifierCV(factory(), method="isotonic", cv=cv)
    cal.fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    def block(p, label):
        auc = roc_auc_score(y_te, p)
        sens = sens_at(0.5)(y_te.values, p)
        spec = spec_at(0.5)(y_te.values, p)
        slope, intercept = calibration_slope_intercept(y_te.values, p)
        prev_rate, prev_desc = SEER_INCIDENCE[name]
        prev = prev_rate / 100_000.0
        ppv_pop, npv_pop, nnt = projected_ppv_npv(sens, spec, prev)
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_te.values == 1)).sum())
        fp = int(((pred == 1) & (y_te.values == 0)).sum())
        return {
            "variant": label,
            "auc": round(float(auc), 3),
            "auc_ci": bootstrap_ci(y_te.values, p, roc_auc_score),
            "sensitivity": round(float(sens), 3),
            "sensitivity_ci": bootstrap_ci(y_te.values, p, sens_at(0.5)),
            "specificity": round(float(spec), 3),
            "specificity_ci": bootstrap_ci(y_te.values, p, spec_at(0.5)),
            "ppv_in_cohort": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "brier": round(float(brier_score_loss(y_te, p)), 4),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "reliability": reliability_curve(y_te.values, p),
            "ppv_at_population_prevalence": round(float(ppv_pop), 5),
            "npv_at_population_prevalence": round(float(npv_pop), 5),
            "people_flagged_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
            "population_prevalence": prev,
            "prevalence_source": prev_desc,
        }

    # --- baselines -----------------------------------------------------------
    baselines = {}
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    baselines["prevalence_only"] = {
        "auc": round(float(roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])), 3)
        if len(np.unique(dummy.predict_proba(X_te)[:, 1])) > 1 else 0.5
    }

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    logit.fit(X_tr, y_tr)
    p_lr = logit.predict_proba(X_te)[:, 1]
    baselines["logistic_regression"] = {
        "auc": round(float(roc_auc_score(y_te, p_lr)), 3),
        "auc_ci": bootstrap_ci(y_te.values, p_lr, roc_auc_score),
    }

    simple_cols = [c for c in ["age", "gender"] if c in X.columns]
    if simple_cols:
        s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        s.fit(X_tr[simple_cols], y_tr)
        p_s = s.predict_proba(X_te[simple_cols])[:, 1]
        baselines["age_sex_only"] = {
            "features": simple_cols,
            "auc": round(float(roc_auc_score(y_te, p_s)), 3),
            "auc_ci": bootstrap_ci(y_te.values, p_s, roc_auc_score),
        }

    # --- subgroups -----------------------------------------------------------
    subgroups = {}
    if "gender" in X_te.columns and X_te["gender"].nunique() > 1:
        for g, lab in [(0, "female"), (1, "male")]:
            m = (X_te["gender"] == g).values
            if m.sum() >= 20 and len(np.unique(y_te.values[m])) > 1:
                subgroups[f"sex: {lab}"] = {
                    "n": int(m.sum()),
                    "positives": int(y_te.values[m].sum()),
                    "auc": round(float(roc_auc_score(y_te.values[m], p_cal[m])), 3),
                    "auc_ci": bootstrap_ci(y_te.values[m], p_cal[m], roc_auc_score),
                }
    if "age" in X_te.columns:
        med = float(X_te["age"].median())
        for lab, m in [
            (f"age under {med:.0f}", (X_te["age"] < med).values),
            (f"age {med:.0f} and over", (X_te["age"] >= med).values),
        ]:
            if m.sum() >= 20 and len(np.unique(y_te.values[m])) > 1:
                subgroups[lab] = {
                    "n": int(m.sum()),
                    "positives": int(y_te.values[m].sum()),
                    "auc": round(float(roc_auc_score(y_te.values[m], p_cal[m])), 3),
                    "auc_ci": bootstrap_ci(y_te.values[m], p_cal[m], roc_auc_score),
                }

    return {
        "domain": name,
        "cohort_design": COHORT_DESIGN[name],
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "cohort_prevalence": round(prevalence_cohort, 3),
        "cv_auc_train_only": round(float(cv_auc), 3),
        "uncalibrated": block(p_raw, "uncalibrated"),
        "calibrated": block(p_cal, "isotonic calibrated"),
        "baselines": baselines,
        "subgroups": subgroups,
    }


def main():
    results = {}
    for config in tm.DATASETS:
        print(f"evaluating {config['name']} ...", flush=True)
        results[config["name"]] = evaluate_domain(config)

    with open("evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    write_report(results)
    print("\nwrote evaluation.json and EVALUATION.md")

    print("\n" + "=" * 96)
    print(f"{'domain':<12}{'CV AUC':<9}{'test AUC':<11}{'LR base':<10}{'age+sex':<10}{'PPV@pop':<10}{'flagged/case'}")
    print("=" * 96)
    for k, r in results.items():
        c = r["calibrated"]
        lr = r["baselines"]["logistic_regression"]["auc"]
        ax = r["baselines"].get("age_sex_only", {}).get("auc", "n/a")
        print(f"{k:<12}{r['cv_auc_train_only']:<9}{c['auc']:<11}{lr:<10}{str(ax):<10}"
              f"{c['ppv_at_population_prevalence']*100:<9.2f}%{c['people_flagged_per_true_case']}")


def write_report(results):
    L = []
    A = L.append
    A("# Oncovision model evaluation\n")
    A("Generated by `evaluate.py`. Every number here comes from a 20% test split that was cut")
    A("before any model was fitted and was not used for training, model selection, or calibration.\n")
    A("## Headline\n")
    A("| Panel | Cohort AUC (test) | 95% CI | Logistic baseline | Age+sex baseline | Beats baseline? |")
    A("|---|---|---|---|---|---|")
    for k, r in results.items():
        c = r["calibrated"]
        lr = r["baselines"]["logistic_regression"]["auc"]
        ax = r["baselines"].get("age_sex_only", {}).get("auc")
        better = "yes" if c["auc"] > lr + 0.02 else ("marginal" if c["auc"] >= lr - 0.02 else "**no**")
        A(f"| {k} | {c['auc']} | {c['auc_ci'][0]} to {c['auc_ci'][1]} | {lr} | {ax if ax is not None else 'n/a'} | {better} |")

    A("\n## Precision at real population prevalence\n")
    A("Cohort AUC is measured on disease-enriched case-control data. Projecting the measured")
    A("sensitivity and specificity onto actual SEER incidence gives the number that decides")
    A("whether a screening tool is usable.\n")
    A("| Panel | SEER incidence | Sensitivity | Specificity | PPV at that prevalence | People flagged per true case |")
    A("|---|---|---|---|---|---|")
    for k, r in results.items():
        c = r["calibrated"]
        A(f"| {k} | {c['population_prevalence']*100:.4f}% | {c['sensitivity']} | {c['specificity']} "
          f"| **{c['ppv_at_population_prevalence']*100:.2f}%** | {c['people_flagged_per_true_case']} |")

    A("\n## Calibration\n")
    A("Slope 1.0 and intercept 0.0 is perfect. Slope below 1 means the model is over-confident.\n")
    A("| Panel | Brier (uncal) | Brier (calibrated) | Slope | Intercept |")
    A("|---|---|---|---|---|")
    for k, r in results.items():
        u, c = r["uncalibrated"], r["calibrated"]
        A(f"| {k} | {u['brier']} | {c['brier']} | {c['calibration_slope']} | {c['calibration_intercept']} |")

    A("\n## Per panel detail\n")
    for k, r in results.items():
        c = r["calibrated"]
        A(f"### {k}\n")
        A(f"- **Cohort design**: {r['cohort_design']}")
        A(f"- **Records**: {r['n_total']} total, {r['n_train']} train, {r['n_test']} held-out test")
        A(f"- **Features** ({r['n_features']}): {', '.join(r['features'])}")
        A(f"- **Cohort prevalence**: {r['cohort_prevalence']*100:.1f}% vs population {c['population_prevalence']*100:.4f}%")
        A(f"- **Test AUC**: {c['auc']} (95% CI {c['auc_ci'][0]} to {c['auc_ci'][1]})")
        A(f"- **Sensitivity**: {c['sensitivity']} (95% CI {c['sensitivity_ci'][0]} to {c['sensitivity_ci'][1]})")
        A(f"- **Specificity**: {c['specificity']} (95% CI {c['specificity_ci'][0]} to {c['specificity_ci'][1]})")
        A(f"- **PPV at SEER prevalence**: {c['ppv_at_population_prevalence']*100:.2f}%, "
          f"about {c['people_flagged_per_true_case']} people flagged per true case")
        if r["subgroups"]:
            A("\n  Subgroups:\n")
            A("  | Group | n | positives | AUC | 95% CI |")
            A("  |---|---|---|---|---|")
            for gk, g in r["subgroups"].items():
                A(f"  | {gk} | {g['n']} | {g['positives']} | {g['auc']} | {g['auc_ci'][0]} to {g['auc_ci'][1]} |")
        A("\n  Reliability (calibrated):\n")
        A("  | Predicted bin | n | mean predicted | observed rate |")
        A("  |---|---|---|---|")
        for b in c["reliability"]:
            A(f"  | {b['bin']} | {b['n']} | {b['predicted']} | {b['observed']} |")
        A("")

    with open("EVALUATION.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
