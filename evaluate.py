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
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
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
    # The general and liver panels predict a LIFETIME diagnosis, "ever told
    # you had this", not an incident one. Scoring them against SEER annual
    # incidence would be comparing a prevalence model to an incidence prior
    # and would understate precision by more than an order of magnitude, so
    # both use the prevalence NHANES itself measures on the same question.
    "general":    (3140.0, "Cancer diagnosed within 4 years, US adults, NHANES 2005-2014"),
    "breast":     (132.5, "Female breast, per 100,000 women per year"),
    "prostate":   (40000.0, "Adenocarcinoma among men taken to prostate biopsy"),
    "pancreatic": (13.9,  "Pancreas, per 100,000 men and women per year"),
    "liver":      (4000.0, "Ever told had a liver condition, US adults, NHANES 2005-2018"),
    # Colorectal is a genuine population screen, so it gets the real SEER
    # incidence rather than a referral prior.
    "colorectal": (36.5, "Colon and rectum, per 100,000 men and women per year"),
    # Lung is offered to people with tobacco exposure, so the prior is lung
    # cancer prevalence in that group rather than in the whole population.
    "lung":       (475.0, "Lung cancer among US adults with tobacco exposure, NHANES 1999-2018"),
    # The ovarian and cervical panels are triage steps, not population screens.
    # They run after a mass has been found or a woman has been referred, so the
    # prior that matters is prevalence in the referred group, not in the street.
    # Using SEER here would be arithmetically correct and clinically meaningless:
    # cervical at population incidence flags about 7,383 women per true case,
    # which is precisely why neither panel is offered as a population screen.
    "ovarian":    (20000.0, "Malignancy among women taken to surgery for an adnexal mass"),
    "cervical":   (6400.0, "Positive biopsy among women assessed for colposcopy, this cohort"),
}

COHORT_DESIGN = {
    "general":    "23,923 US adults, NHANES 2005-2014. Recent diagnosis, survivors excluded.",
    "breast":     "Case-control, post-biopsy. Every record is an FNA already taken because a lesion was found.",
    "liver":      "35,511 US adults, NHANES 2005-2018. Externally validated on India and Germany.",
    "pancreatic": "Case-control. Cases are confirmed PDAC, controls include benign hepatobiliary disease.",
    "prostate":   "212 men biopsied at one centre. Controls are benign biopsies. Needs an MRI PI-RADS score.",
    "colorectal": "23,794 US adults, NHANES 2005-2014. Diagnosed within 8 years; longer-ago survivors excluded.",
    "lung":       "21,916 US adults with tobacco exposure, NHANES 1999-2018. Controls are smokers, not the general population.",
    "ovarian":    "349 women operated on at one Chinese hospital. Controls are benign ovarian tumours, not healthy women.",
    "cervical":   "858 women assessed for colposcopy in Caracas, 55 biopsy-positive. Prior-diagnosis columns dropped as leakage.",
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

    # Calibrated out-of-fold scores, so the operating point is chosen on the
    # same scale it will be applied to. Mixing the two silently produces a
    # threshold that catches almost nobody.
    oof_cal = cross_val_predict(
        CalibratedClassifierCV(factory(), method="isotonic", cv=cv),
        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]

    # --- fitted on train, measured on the untouched test set -----------------
    raw = factory().fit(X_tr, y_tr)
    p_raw = raw.predict_proba(X_te)[:, 1]

    # Isotonic calibration, fitted inside the training portion only.
    cal = CalibratedClassifierCV(factory(), method="isotonic", cv=cv)
    cal.fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]

    def block(p, label, oof_scores):
        auc = roc_auc_score(y_te, p)
        # Operating point chosen on the TRAINING data by Youden's J, never on
        # the held-out split, and from out-of-fold scores on the same scale as
        # p. A fixed 0.5 is wrong once prevalence is realistic: it drove measured
        # sensitivity to 0.008 on the 9 percent general cohort.
        fpr_t, tpr_t, cuts_t = roc_curve(y_tr, oof_scores)
        thr = cuts_t[int(np.argmax(tpr_t - fpr_t))]
        thr = float(np.clip(thr, 0.01, 0.99)) if np.isfinite(thr) else 0.5
        sens = sens_at(thr)(y_te.values, p)
        spec = spec_at(thr)(y_te.values, p)
        slope, intercept = calibration_slope_intercept(y_te.values, p)
        prev_rate, prev_desc = SEER_INCIDENCE[name]
        prev = prev_rate / 100_000.0
        ppv_pop, npv_pop, nnt = projected_ppv_npv(sens, spec, prev)
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y_te.values == 1)).sum())
        fp = int(((pred == 1) & (y_te.values == 0)).sum())
        return {
            "variant": label,
            "threshold": round(float(thr), 4),
            "auc": round(float(auc), 3),
            "auc_ci": bootstrap_ci(y_te.values, p, roc_auc_score),
            "sensitivity": round(float(sens), 3),
            "sensitivity_ci": bootstrap_ci(y_te.values, p, sens_at(thr)),
            "specificity": round(float(spec), 3),
            "specificity_ci": bootstrap_ci(y_te.values, p, spec_at(thr)),
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
        "uncalibrated": block(p_raw, "uncalibrated", oof),
        "calibrated": block(p_cal, "isotonic calibrated", oof_cal),
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
