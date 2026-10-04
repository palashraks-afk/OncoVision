"""
Is there ANY cancer-specific information in routine bloodwork?

Why this is the right next question
-----------------------------------
mortality_cause_specificity.py showed the bloodwork panel predicts death from
other causes as well as death from cancer. The remaining question is not "how do
we improve it" but whether the information it would need exists at all: among
people who die, can routine labs tell a cancer death from another death better
than age and sex can? Age and sex cannot (they are the same in both groups to a
first approximation), so any gain is cancer-specific by construction. That is
what makes the decedent population the clean place to look.

What is searched
----------------
    horizon     5, 10 and 15 years. Longer horizons mean more cancer deaths
                (power), and they ask a more useful question: whether blood
                drawn long before death says which way it will go. A horizon
                only uses cohorts that COULD have been followed that long,
                because follow-up ends in 2019 and a later cycle simply cannot
                contribute a ten-year death. Without that restriction cycle
                composition would drift with the horizon.
    inputs      age and sex (the baseline); plus the 22 routine labs both
                surveys carry; plus BMI, smoking and alcohol, which the
                continuous survey has and NHANES III does not.
    comparison  cancer against all other deaths, and cancer against deaths from
                heart disease alone, which is the largest competitor.

Pre-registered, written before running
--------------------------------------
Cancer-specific signal in routine labs is FOUND only if, for the same inputs and
horizon, BOTH hold: (1) inside the continuous survey, repeated paired
cross-validation gives a gain over the stronger age-and-sex model with a 95%
interval excluding zero; and (2) trained on the continuous survey and applied
unchanged to NHANES III, the gain again has an interval excluding zero. The
lifestyle arm cannot be checked externally, so at best it is "unconfirmed".
Twelve comparisons are made, so a lone positive is expected by chance and is
treated as a lead to confirm, not a finding.

Run:  python experiments/cancer_specific_signal_search.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/cancer_specific_signal_search_result.json"
TRAIN = "data/nhanes_mortality_full.csv"
EXTERNAL = "data/nhanes3_mortality_full.csv"
CANCER, HEART = 2, 1
REPEATS, FOLDS, N_BOOT = 5, 5, 2000
MIN_EVENTS = 40

# Cohorts that could have been followed for the horizon, given follow-up ends
# in December 2019. A 2009-2010 exam has at most about ten years.
CYCLES_FOR = {
    5: None,
    10: ("1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"),
    15: ("1999-2000", "2001-2002", "2003-2004"),
}
DEMO = ["age", "gender"]
LIFESTYLE = ["bmi", "smoking", "alcohol_intake"]


def decedents(df, horizon, versus):
    """Rows for people who died within the horizon, labelled cancer or not."""
    d = df[(df["died"] == 1) & (df["followup_months"] <= horizon * 12)].copy()
    if versus == "heart":
        d = d[d["ucod_leading"].isin([CANCER, HEART])]
    d["label"] = (d["ucod_leading"] == CANCER).astype(int)
    return d


def paired_boot(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def cv_gain(X, base, y):
    """Repeated paired 5-fold gain, each model the stronger of two kinds."""
    gains = []
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        aucs = {}
        for name, M in (("panel", X), ("base", base)):
            best = -1.0
            for kind in ("logistic", "ensemble"):
                preds = np.zeros(len(y))
                for tr, te in cv.split(M, y):
                    m = tm.model_factory(kind, len(tr), float(y.iloc[tr].mean()))
                    m.fit(M.iloc[tr], y.iloc[tr])
                    preds[te] = m.predict_proba(M.iloc[te])[:, 1]
                best = max(best, roc_auc_score(y, preds))
            aucs[name] = best
        gains.append(aucs["panel"] - aucs["base"])
    g = np.array(gains)
    return (round(float(g.mean()), 4),
            [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)])


def external_gain(Xtr, ytr, Xte, yte, cols):
    best_p, best_b = None, None
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(ytr), float(ytr.mean())).fit(Xtr[cols], ytr)
        b = tm.model_factory(kind, len(ytr), float(ytr.mean())).fit(Xtr[DEMO], ytr)
        p, bb = m.predict_proba(Xte[cols])[:, 1], b.predict_proba(Xte[DEMO])[:, 1]
        if best_p is None or roc_auc_score(yte, p) > roc_auc_score(yte, best_p):
            best_p = p
        if best_b is None or roc_auc_score(yte, bb) > roc_auc_score(yte, best_b):
            best_b = bb
    auc, bauc = float(roc_auc_score(yte, best_p)), float(roc_auc_score(yte, best_b))
    return (round(auc, 4), round(bauc, 4), round(auc - bauc, 4),
            paired_boot(yte, best_p, best_b))


def main():
    for p in (TRAIN, EXTERNAL):
        if not os.path.exists(p):
            print(f"{p} is missing. Run the --full fetchers first.")
            return 1
    tr, te = pd.read_csv(TRAIN), pd.read_csv(EXTERNAL)
    skip = {"cancer_death", "followup_months", "cycle", "race_ethnicity", "died",
            "ucod_leading", "label", "smoking", "alcohol_intake", "bmi"}
    labs = [c for c in tr.columns if c in te.columns and c not in skip and c not in DEMO
            and pd.api.types.is_numeric_dtype(tr[c])]
    print(f"{len(labs)} routine labs shared by both surveys\n")

    arms = {"labs": DEMO + labs, "labs_and_lifestyle": DEMO + labs + LIFESTYLE}
    results = {"labs": labs, "rows": []}
    found = []
    print(f"{'horizon':>7} {'versus':<7} {'inputs':<19} {'events':>6}/{'n':<5} "
          f"{'internal gain':>14}  {'NHANES III gain':>16}")
    for horizon, cycles in CYCLES_FOR.items():
        sub = tr if cycles is None else tr[tr["cycle"].isin(cycles)]
        for versus in ("other", "heart"):
            d_tr = decedents(sub, horizon, versus)
            d_te = decedents(te, horizon, versus)
            ev_tr, ev_te = int(d_tr["label"].sum()), int(d_te["label"].sum())
            if ev_tr < MIN_EVENTS or ev_te < MIN_EVENTS or d_tr["label"].nunique() < 2:
                results["rows"].append({"horizon": horizon, "versus": versus,
                                        "skipped": "too few events",
                                        "events_train": ev_tr, "events_test": ev_te})
                print(f"{horizon:>6}y {versus:<7} too few events ({ev_tr} / {ev_te})")
                continue
            med = d_tr[DEMO + labs + LIFESTYLE].median()
            for arm, cols in arms.items():
                Xtr = d_tr[cols].fillna(med[[c for c in cols if c in med.index]]).fillna(0.0)
                ytr = d_tr["label"].reset_index(drop=True)
                Xtr = Xtr.reset_index(drop=True)
                g, gci = cv_gain(Xtr[cols], Xtr[DEMO], ytr)
                internal_ok = bool(gci[0] > 0)
                row = {"horizon": horizon, "versus": versus, "inputs": arm,
                       "n_train": int(len(d_tr)), "events_train": ev_tr,
                       "internal_gain": g, "internal_gain_ci": gci,
                       "internal_ok": internal_ok}
                ext_txt = "cannot be checked"
                if arm == "labs":
                    Xte = d_te[cols].fillna(med[cols]).fillna(0.0).reset_index(drop=True)
                    yte = d_te["label"].reset_index(drop=True)
                    a, ba, eg, egci = external_gain(Xtr, ytr, Xte, yte, cols)
                    row.update({"n_test": int(len(d_te)), "events_test": ev_te,
                                "external_auc": a, "external_age_sex_auc": ba,
                                "external_gain": eg, "external_gain_ci": egci,
                                "external_ok": bool(egci[0] > 0)})
                    ext_txt = f"{eg:+.4f} ({egci[0]:+.4f} to {egci[1]:+.4f})"
                    row["found"] = bool(internal_ok and row["external_ok"])
                    if row["found"]:
                        found.append((horizon, versus))
                else:
                    row["found"] = False
                results["rows"].append(row)
                print(f"{horizon:>6}y {versus:<7} {arm:<19} {ev_tr:>6}/{len(d_tr):<5} "
                      f"{g:>+7.4f} {'(' + format(gci[0], '+.3f') + ' to ' + format(gci[1], '+.3f') + ')':>20}"
                      f"  {ext_txt}")

    results["found"] = [{"horizon": h, "versus": v} for h, v in found]
    results["comparisons"] = sum(1 for r in results["rows"] if "skipped" not in r)
    results["any_found"] = bool(found)
    print(f"\n  {results['comparisons']} comparisons made. "
          + ("CANCER-SPECIFIC SIGNAL FOUND (internal and on NHANES III) at: "
             + ", ".join(f"{h}y vs {v}" for h, v in found) + ". Treat as a lead to confirm."
             if found else
             "No cancer-specific signal in routine labs survives both tests."))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
