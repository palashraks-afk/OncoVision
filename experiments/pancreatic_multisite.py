"""
Multi-site validation of the pancreatic panel, and a search for a prostate one.

Background
----------
Both panels were withdrawn for having no external test. That was true of public
cohorts sharing their feature sets, but it skipped something: the pancreatic
dataset is itself multi-site, and the site column was being thrown away as an
identifier.

  sample_o   BPTB  365 samples,  74 confirmed adenocarcinoma
             CPTB  141 samples,  34
             UPTB   94 samples,  22

Three independent tissue banks. Training on two and testing on the third is a
real external test: different institution, different collection protocol,
different referral population. It is weaker than a different country, and it is
far stronger than a random split of one cohort.

For prostate, NHANES 2005 to 2010 measured serum PSA (LBXP1) on 4,697 men across
three cycles. That looked like the answer until the cases were counted: NHANES
excludes men with a prostate cancer history from the PSA subsample, which is
correct for a screening measurement and leaves only 17 cases. Far below the
roughly 96 events needed, so the prostate withdrawal stands on measured evidence
rather than on absence of search.

Run:  python experiments/pancreatic_multisite.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci, sens_at, spec_at, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES = ["age", "gender", "creatinine", "plasma_ca19_9", "bilirubin", "glucose"]
PANCREAS_INCIDENCE = 13.9 / 100_000

SITE_NAMES = {
    "BPTB": "Barts Pancreas Tissue Bank",
    "CPTB": "Second tissue bank",
    "UPTB": "Third tissue bank",
}


def load():
    d = pd.read_csv("data/pancreatic_600_risk123.csv")
    X = pd.DataFrame({
        "age": pd.to_numeric(d["age"], errors="coerce"),
        "gender": (d["sex"].astype(str).str.strip().str.upper() == "M").astype(int),
        "creatinine": pd.to_numeric(d["creatinine"], errors="coerce"),
        "plasma_ca19_9": pd.to_numeric(d["CA19_9"], errors="coerce"),
        "bilirubin": pd.to_numeric(d["bilirubin"], errors="coerce"),
        "glucose": pd.to_numeric(d["glucose"], errors="coerce"),
    })
    # Only class 3 is adenocarcinoma. 1 is control, 2 is benign hepatobiliary.
    y = (pd.to_numeric(d["diagnosis"], errors="coerce") == 3).astype(int)
    site = d["sample_o"].astype(str)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), site[keep].reset_index(drop=True)


def fit(X, y):
    folds = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    )
    return model.fit(X, y)


def main():
    X, y, site = load()
    print(f"Pancreatic cohort: {len(y)} samples, {int(y.sum())} adenocarcinoma "
          f"({y.mean():.1%})\n")
    print("By collection site:")
    for s in sorted(site.unique()):
        m = site == s
        print(f"  {s:<6} {SITE_NAMES.get(s, s):<30} n={int(m.sum()):<5} "
              f"cases={int(y[m].sum()):<4} ({y[m].mean():.1%})")

    # Internal reference: the random split the panel was originally judged on.
    Xa, Xb, ya, yb = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    p_int = fit(Xa, ya).predict_proba(Xb)[:, 1]
    internal = roc_auc_score(yb, p_int)
    print(f"\nInternal random split AUC: {internal:.3f}   "
          f"(this is what the panel was judged on before)")

    print("\nLeave-one-site-out, the real test")
    results = {}
    for held in sorted(site.unique()):
        te = site == held
        tr = ~te
        if y[tr].nunique() < 2 or y[te].nunique() < 2:
            continue

        model = fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        ci = bootstrap_ci(np.asarray(y[te]), p, roc_auc_score)

        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000, class_weight="balanced"))
        lr.fit(X[tr], y[tr])
        lr_auc = roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1])

        sens = sens_at(0.5)(np.asarray(y[te]), p)
        spec = spec_at(0.5)(np.asarray(y[te]), p)
        ppv, _npv, nnt = projected_ppv_npv(sens, spec, PANCREAS_INCIDENCE)

        results[held] = {
            "site": held,
            "site_name": SITE_NAMES.get(held, held),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "cases_test": int(y[te].sum()),
            "auc": round(float(auc), 3),
            "auc_ci": ci,
            "logistic_auc": round(float(lr_auc), 3),
            "sensitivity": round(float(sens), 3),
            "specificity": round(float(spec), 3),
            "ppv_at_population_prevalence": round(float(ppv), 5),
            "people_flagged_per_true_case": round(float(nnt), 1) if np.isfinite(nnt) else None,
        }
        print(f"  held out {held:<6} train n={int(tr.sum()):<4} test n={int(te.sum()):<4} "
              f"cases={int(y[te].sum()):<3} AUC {auc:.3f} "
              f"(95% CI {ci[0]} to {ci[1]})  logistic {lr_auc:.3f}")

    mean_auc = float(np.mean([r["auc"] for r in results.values()]))
    worst_lo = min(r["auc_ci"][0] for r in results.values())
    beats_chance = worst_lo > 0.5

    print(f"\nMean leave-one-site-out AUC: {mean_auc:.3f}")
    print(f"Lowest CI bound across sites: {worst_lo}")
    print(f"Drop from internal random split: {internal - mean_auc:+.3f}")

    print()
    if beats_chance:
        print("VERDICT: every site's interval excludes chance. The panel does transfer "
              "across institutions, so the withdrawal should be reconsidered on this "
              "evidence, with the caveat that all three sites are pancreatic tissue "
              "banks and none is a screening population.")
    else:
        print("VERDICT: at least one site's interval includes chance. Multi-site "
              "transfer is not demonstrated and the withdrawal stands.")

    out = {
        "question": "Does the pancreatic panel transfer across the three collection sites in its own cohort?",
        "internal_random_split_auc": round(float(internal), 3),
        "mean_leave_one_site_out_auc": round(mean_auc, 3),
        "drop": round(float(internal - mean_auc), 3),
        "lowest_ci_bound": worst_lo,
        "every_site_beats_chance": bool(beats_chance),
        "per_site": results,
        "prostate_note": ("NHANES 2005 to 2010 measured serum PSA on 4,697 men but contains only "
                          "17 prostate cancer cases, because men with a prostate cancer history "
                          "are excluded from the PSA subsample. Below the ~96 events needed, so "
                          "no external prostate validation is possible from it."),
        "caveat": ("All three sites are pancreatic tissue banks with 22 to 28 percent case "
                   "prevalence. This tests transfer between institutions, not transfer to a "
                   "screening population, and says nothing about the 525-per-true-case precision "
                   "problem at real incidence."),
    }
    with open("experiments/pancreatic_multisite_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/pancreatic_multisite_result.json")


if __name__ == "__main__":
    main()
