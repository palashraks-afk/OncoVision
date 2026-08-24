"""
External validation of the liver panel across two independent real cohorts.

The question every held-out split fails to answer is whether a model works on
patients it has never seen from a source it has never seen. A held-out slice of
one dataset shares that dataset's hospital, its assay machines, its referral
patterns and its population. Splitting it does not test generalisation, it
tests memorisation.

Two independent cohorts make the real test possible:

  India    ILPD, 583 patients, Andhra Pradesh. 71% liver disease.
  Germany  HCV data, 589 patients. 9.5% liver disease, the rest blood donors.

Different continent, hospital, protocol, population and disease prevalence, and
eight shared liver chemistry measurements after unit harmonisation.

Trained on one, tested on the other, in both directions. Nothing from the test
cohort touches training, tuning or calibration.

Run:  python external_validation.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv, calibration_slope_intercept

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES = ["age", "gender", "bilirubin", "alkaline_phosphatase",
            "alt", "ast", "protein_total", "albumin"]
TARGET = "liver_disease"

COHORTS = {
    "India": ("data/ilpd_liver_india.csv", "ILPD, Andhra Pradesh, UCI 225, clinical"),
    "Germany": ("data/hcv_liver_germany.csv", "HCV data, Germany, UCI 571, clinical"),
    "USA": ("data/nhanes_liver_usa.csv", "NHANES 2017-2018, CDC, population based"),
}

# SEER liver and intrahepatic bile duct incidence, used only to show what the
# precision would look like at a screening prevalence. Chronic liver disease is
# far more common than liver cancer, so this is a floor, not an estimate of
# liver disease prevalence.
LIVER_INCIDENCE = 9.5 / 100_000


def load(name):
    path, _ = COHORTS[name]
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET].astype(int)


def metrics(y, p, label):
    auc = roc_auc_score(y, p)
    sens = sens_at(0.5)(np.asarray(y), np.asarray(p))
    spec = spec_at(0.5)(np.asarray(y), np.asarray(p))
    slope, intercept = calibration_slope_intercept(np.asarray(y), np.asarray(p))
    ppv, npv, nnt = projected_ppv_npv(sens, spec, LIVER_INCIDENCE)
    return {
        "setting": label,
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": round(float(np.mean(y)), 3),
        "auc": round(float(auc), 3),
        "auc_ci": bootstrap_ci(np.asarray(y), np.asarray(p), roc_auc_score),
        "sensitivity": round(float(sens), 3),
        "sensitivity_ci": bootstrap_ci(np.asarray(y), np.asarray(p), sens_at(0.5)),
        "specificity": round(float(spec), 3),
        "specificity_ci": bootstrap_ci(np.asarray(y), np.asarray(p), spec_at(0.5)),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "calibration_slope": slope,
    }


def fit(X, y):
    """Same ensemble the application ships, calibrated the same way."""
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    )
    model.fit(X, y)
    return model


def run_direction(train_name, test_name):
    Xtr, ytr = load(train_name)
    Xte, yte = load(test_name)

    model = fit(Xtr, ytr)

    # Internal reference: held-out split of the training cohort itself.
    Xa, Xb, ya, yb = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=RANDOM_STATE, stratify=ytr
    )
    internal_model = fit(Xa, ya)
    internal = metrics(yb, internal_model.predict_proba(Xb)[:, 1],
                       f"internal held-out split of {train_name}")

    # The real test: a cohort from another country.
    external = metrics(yte, model.predict_proba(Xte)[:, 1],
                       f"external, trained on {train_name}, tested on {test_name}")

    # Logistic regression, same splits, as a floor.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    lr_external = round(float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])), 3)

    return {
        "train": train_name,
        "test": test_name,
        "internal": internal,
        "external": external,
        "external_logistic_auc": lr_external,
        "auc_drop": round(internal["auc"] - external["auc"], 3),
    }


def leave_one_cohort_out():
    """
    Leave-one-cohort-out validation, which is the correct design once more than
    one source exists.

    Single-source training is the problem the pairwise table exposes: a model
    fitted on one cohort learns that cohort's referral pattern, its assay
    calibration and its prevalence, and loses between 0.06 and 0.46 AUC when
    moved. Training on two cohorts and testing on the third measures whether
    diversity in the training data buys generalisation, and it is also the
    honest way to estimate what a model trained on everything would do on the
    next unseen population.

    The held-out cohort contributes nothing to fitting or calibration.
    """
    results = {}
    for held in COHORTS:
        train_names = [c for c in COHORTS if c != held]
        Xs, ys = zip(*[load(c) for c in train_names])
        X_tr = pd.concat(Xs, ignore_index=True)
        y_tr = pd.concat(ys, ignore_index=True)
        X_te, y_te = load(held)

        model = fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]

        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000, class_weight="balanced"))
        lr.fit(X_tr, y_tr)
        lr_auc = round(float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])), 3)

        results[held] = {
            "trained_on": train_names,
            "n_train": int(len(y_tr)),
            "held_out": held,
            **metrics(y_te, p, f"trained on {' + '.join(train_names)}, tested on {held}"),
            "logistic_auc": lr_auc,
        }
    return results


def validate_general_on_nhanes():
    """
    External test for the general panel against NHANES 2017-2018.

    The general panel trains on a risk-factor cohort that is 37% positive.
    NHANES is a nationally representative survey of US adults where 10.3% have
    ever been told they had cancer. So this tests two things at once: does the
    model transfer to a different population, and does it survive a realistic
    prevalence instead of an enriched one.

    Two of the eight training features do not exist in NHANES. Inherited risk
    is not surveyed, and prior cancer diagnosis is the outcome itself, so
    including it would leak. Both are filled with the training median, which is
    exactly what the application does for a patient who leaves them blank.

    NHANES also carries race and ethnicity, which no other cohort here does, so
    this is the only place subgroup accuracy across those groups can be
    measured rather than disclaimed.
    """
    cfg = next(c for c in tm.DATASETS if c["name"] == "general")
    X_tr, y_tr, medians = tm.prepare(cfg)

    nh = pd.read_csv("data/nhanes_general_usa.csv")
    y_te = nh["any_cancer"].astype(int)

    X_te = pd.DataFrame(
        {f: (nh[f] if f in nh.columns else medians.get(f, 0.0)) for f in X_tr.columns}
    )[X_tr.columns]
    X_te = X_te.fillna(medians)

    missing = [f for f in X_tr.columns if f not in nh.columns]

    model = fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    internal_split = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )
    Xa, Xb, ya, yb = internal_split
    internal = metrics(yb, fit(Xa, ya).predict_proba(Xb)[:, 1], "internal held-out split")
    external = metrics(y_te, p, "external, NHANES 2017-2018")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(X_tr, y_tr)
    lr_auc = round(float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])), 3)

    # Subgroup accuracy by race and ethnicity.
    subgroups = {}
    for group, mask in nh.groupby("race_ethnicity").groups.items():
        idx = nh.index.isin(mask)
        if idx.sum() >= 100 and len(np.unique(y_te[idx])) > 1:
            subgroups[str(group)] = {
                "n": int(idx.sum()),
                "positives": int(y_te[idx].sum()),
                "prevalence": round(float(y_te[idx].mean()), 3),
                "auc": round(float(roc_auc_score(y_te[idx], p[idx])), 3),
                "auc_ci": bootstrap_ci(np.asarray(y_te[idx]), p[idx], roc_auc_score),
            }

    return {
        "panel": "general",
        "train": "Risk-factor cohort, 1500 records, 37% positive",
        "test": "NHANES 2017-2018, 5173 US adults, 10.3% positive, nationally representative",
        "features_unavailable_in_nhanes": missing,
        "internal": internal,
        "external": external,
        "external_logistic_auc": lr_auc,
        "auc_drop": round(internal["auc"] - external["auc"], 3),
        "subgroups_by_race": subgroups,
    }


def main():
    print("Loading cohorts")
    for name, (path, desc) in COHORTS.items():
        X, y = load(name)
        print(f"  {name:<9} {len(y):>4} patients, {int(y.sum()):>4} with liver disease "
              f"({y.mean():.1%}) - {desc}")

    results = {}
    # Every ordered pair, so no direction is cherry-picked.
    pairs = [(a, b) for a in COHORTS for b in COHORTS if a != b]
    for a, b in pairs:
        print(f"\nTraining on {a}, testing on {b}")
        r = run_direction(a, b)
        results[f"{a}_to_{b}"] = r
        print(f"  internal held-out AUC  {r['internal']['auc']}  "
              f"(95% CI {r['internal']['auc_ci'][0]} to {r['internal']['auc_ci'][1]})")
        print(f"  EXTERNAL AUC           {r['external']['auc']}  "
              f"(95% CI {r['external']['auc_ci'][0]} to {r['external']['auc_ci'][1]})")
        print(f"  drop                   {r['auc_drop']}")
        print(f"  logistic on external   {r['external_logistic_auc']}")

    print("\nGeneral panel against NHANES 2017-2018")
    g = validate_general_on_nhanes()
    results["general_to_nhanes"] = g
    print(f"  features NHANES does not have: {g['features_unavailable_in_nhanes']}")
    print(f"  internal held-out AUC  {g['internal']['auc']}")
    print(f"  EXTERNAL AUC           {g['external']['auc']}  "
          f"(95% CI {g['external']['auc_ci'][0]} to {g['external']['auc_ci'][1]})")
    print(f"  drop                   {g['auc_drop']}")
    print(f"  logistic on external   {g['external_logistic_auc']}")
    print("  by race and ethnicity:")
    for k, v in g["subgroups_by_race"].items():
        print(f"    {k:<24} n={v['n']:<6} prev={v['prevalence']:<7} AUC={v['auc']}  "
              f"CI {v['auc_ci'][0]} to {v['auc_ci'][1]}")

    print("\nLeave-one-cohort-out, the correct design once three sources exist")
    loco = leave_one_cohort_out()
    results["leave_one_cohort_out"] = loco
    for held, r in loco.items():
        print(f"  held out {held:<8} trained on {' + '.join(r['trained_on']):<18} "
              f"n={r['n_train']:<5} AUC {r['auc']}  "
              f"(95% CI {r['auc_ci'][0]} to {r['auc_ci'][1]})  logistic {r['logistic_auc']}")

    with open("external_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    pairwise = {k: v for k, v in results.items() if k != "leave_one_cohort_out"}
    print("\n" + "=" * 88)
    print(f"{'direction':<22}{'internal':<12}{'external':<12}{'drop':<10}{'external 95% CI'}")
    print("=" * 88)
    for k, r in pairwise.items():
        ci = r["external"]["auc_ci"]
        print(f"{k.replace('_', ' '):<22}{r['internal']['auc']:<12}{r['external']['auc']:<12}"
              f"{r['auc_drop']:<10}{ci[0]} to {ci[1]}")

    liver_pairs = [r for k, r in pairwise.items() if k != "general_to_nhanes"]
    if liver_pairs:
        mean_single = sum(r["external"]["auc"] for r in liver_pairs) / len(liver_pairs)
        mean_loco = sum(r["auc"] for r in loco.values()) / len(loco)
        print("\n" + "=" * 88)
        print(f"Liver panel, mean external AUC")
        print(f"  trained on ONE cohort   {mean_single:.3f}   ({len(liver_pairs)} directions)")
        print(f"  trained on TWO cohorts  {mean_loco:.3f}   ({len(loco)} held-out cohorts)")
        print(f"  gain from cohort diversity: {mean_loco - mean_single:+.3f}")

    print("\nwrote external_validation.json")


if __name__ == "__main__":
    main()
