"""
Should the prospective panel ship, and should it carry a cut?

Why this is the one worth asking
--------------------------------
Every panel that reads routine bloodwork for a named cancer has been withdrawn:
bowel, general and lung, each because the lab values added nothing to age and
sex on data the model had not seen. The reason is a design flaw those cohorts
share -- the blood and the answer were recorded at the same visit, and the
"answer" was a survivor being asked whether they had ever had cancer.

The NHANES-NDI linkage does not share it. Blood is drawn, years pass, and a
death certificate arrives later from a different agency. 33,834 adults with no
cancer diagnosis at the draw, 339 dead of cancer within five years. NHANES III
gives the same design in a different decade: 14,630 adults, 254 deaths.

This has only ever been run as evidence in the paper. It has never been asked
the question that decides a product: does it clear the bars every shipped panel
here has to clear, and if so, what may it say to a person?

Pre-registered, written before this was run
-------------------------------------------
SHIP THE PANEL only if, on NHANES III, its gain over the STRONGER of a logistic
and an ensemble age-and-sex model has a 95% interval excluding zero. Features
are restricted to those present in both cohorts, so the shipped model is the one
that was externally tested -- not a larger one that shares its name.

SHIP A RULE-OUT CUT only if, on NHANES III, at the same share of deaths caught,
that cut excludes materially more adults than a cut on age and sex, interval
excluding zero. This is the test that withdrew bowel and general. Failing it
does not withdraw the panel; it means the panel ships with no cut, as liver and
lung already do, and may not tell anyone they are safe.

Also measured, and reported whatever they say: precision at the cohort's own
event rate, and accuracy by race and ethnicity.

What this panel may never claim
-------------------------------
The outcome is DEATH FROM CANCER within five years, not a diagnosis. A person
who is diagnosed and successfully treated is a negative here. So this cannot say
"you have cancer" or "you will be diagnosed": at best it says this person's
routine bloodwork resembles that of people who went on to die of a cancer nobody
had found yet. The only honest action attached to that is the screening they are
already eligible for.

Run:  python experiments/prospective_panel_ship.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/prospective_panel_ship_result.json"
TRAIN = "data/nhanes_cancer_mortality.csv"
EXTERNAL = "data/nhanes3_cancer_mortality.csv"
TARGET = "cancer_death"
CATCH = 0.95
N_BOOT = 2000
MIN_GROUP_EVENTS = 20

DEMO = ["age", "gender"]


def shared_features(tr, te):
    """Only what both cohorts carry. The shipped panel is the tested panel."""
    skip = {TARGET, "followup_months", "cycle", "race_ethnicity"}
    return [c for c in tr.columns
            if c in te.columns and c not in skip and pd.api.types.is_numeric_dtype(tr[c])]


def rule_out_share(y, p, catch=CATCH):
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    total = ys.sum()
    if total == 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(ys), (1 - catch) * total, side="right"))
    return k / len(ys)


def paired_boot(y, a, b, fn, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].min() == y[i].max():
            continue
        out.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    return (round(float(np.mean(out)), 4),
            [round(float(np.percentile(out, 2.5)), 4),
             round(float(np.percentile(out, 97.5)), 4)])


def fit_best(X_tr, y_tr, X_te):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best = None
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(y_tr), float(y_tr.mean()))
        a = float(np.mean(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")))
        if best is None or a > best[0]:
            fitted = tm.model_factory(kind, len(y_tr), float(y_tr.mean()))
            fitted.fit(X_tr, y_tr)
            best = (a, kind, fitted.predict_proba(X_te)[:, 1], fitted)
    return best


def flagged_per_case(y, p, catch=0.5):
    """People flagged per true case, at the cut catching `catch` of events."""
    thr = np.quantile(p[np.asarray(y) == 1], 1 - catch)
    flagged = p >= thr
    caught = int((flagged & (np.asarray(y) == 1)).sum())
    return (round(float(flagged.sum() / max(caught, 1)), 1), int(flagged.sum()), caught)


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(EXTERNAL)
    feats = shared_features(tr, te)
    y_tr = tr[TARGET].astype(int)
    y_te = te[TARGET].astype(int).to_numpy()
    med = tr[feats].median()

    X_tr = tr[feats].fillna(med)
    X_te = te[feats].fillna(med)

    print(f"Train  NHANES 1999-2014: {len(tr):,} adults, {int(y_tr.sum())} cancer deaths "
          f"within 5 years ({y_tr.mean():.2%})")
    print(f"Test   NHANES III 1988-1994: {len(te):,} adults, {int(y_te.sum())} deaths "
          f"({y_te.mean():.2%})")
    print(f"Shared features ({len(feats)}): {', '.join(feats)}\n")

    cv_panel, kind_panel, p, model = fit_best(X_tr, y_tr, X_te)
    cv_base, kind_base, b, _ = fit_best(X_tr[DEMO], y_tr, X_te[DEMO])

    auc, base_auc = float(roc_auc_score(y_te, p)), float(roc_auc_score(y_te, b))
    gain, gain_ci = paired_boot(y_te, p, b, roc_auc_score)
    share, share_ci = paired_boot(y_te, p, b, rule_out_share, seed=1)
    ship = bool(gain_ci[0] > 0)
    cut_ok = bool(share_ci[0] > 0)

    print(f"  panel      {len(feats):>2} inputs  {kind_panel:<9} internal CV {cv_panel:.3f}   "
          f"NHANES III {auc:.3f} {bootstrap_ci(y_te, p, roc_auc_score)}")
    print(f"  age, sex    2 inputs  {kind_base:<9} internal CV {cv_base:.3f}   "
          f"NHANES III {base_auc:.3f}")
    print(f"\n  BAR 1, ship the panel: gain {gain:+.4f}  95% CI {gain_ci[0]:+.4f} to "
          f"{gain_ci[1]:+.4f}  -> {'PASS' if ship else 'FAIL'}")
    print(f"  BAR 2, ship a cut: at {CATCH:.0%} of deaths caught it excludes {share:+.1%} "
          f"more adults than a cut on age and sex ({share_ci[0]:+.1%} to {share_ci[1]:+.1%}) "
          f"-> {'PASS' if cut_ok else 'FAIL, the panel ships with no cut'}")

    ratio, flagged, caught = flagged_per_case(y_te, p)
    print(f"\n  precision: catching half the deaths flags {flagged:,} of {len(y_te):,} adults, "
          f"{ratio} people per death, at this cohort's {y_te.mean():.2%} rate")

    groups = {}
    if "race_ethnicity" in tr.columns:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        from sklearn.model_selection import cross_val_predict
        oof = cross_val_predict(tm.model_factory(kind_panel, len(y_tr), float(y_tr.mean())),
                                X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        overall = float(roc_auc_score(y_tr, oof))
        print(f"\n  by race and ethnicity, out of fold (overall {overall:.3f}):")
        for g, part in tr.groupby("race_ethnicity"):
            idx = part.index.to_numpy()
            yg = y_tr.to_numpy()[idx]
            if yg.sum() < MIN_GROUP_EVENTS or yg.min() == yg.max():
                groups[g] = {"events": int(yg.sum()), "auc": None}
                print(f"    {g:<26} {int(yg.sum()):>4} deaths  too few to measure")
                continue
            a = float(roc_auc_score(yg, oof[idx]))
            groups[g] = {"events": int(yg.sum()), "auc": round(a, 3),
                         "materially_worse": bool(a < overall - 0.05)}
            print(f"    {g:<26} {int(yg.sum()):>4} deaths  AUC {a:.3f}"
                  f"{'   <-- more than 0.05 below overall' if a < overall - 0.05 else ''}")

    result = {
        "n_train": int(len(tr)), "events_train": int(y_tr.sum()),
        "n_test": int(len(te)), "events_test": int(y_te.sum()),
        "features": feats, "model_kind": kind_panel,
        "internal_cv_auc": round(cv_panel, 3),
        "external_auc": round(auc, 3),
        "external_auc_ci": bootstrap_ci(y_te, p, roc_auc_score),
        "age_sex_external_auc": round(base_auc, 3),
        "external_gain": gain, "external_gain_ci": gain_ci,
        "ship_the_panel": ship,
        "rule_out_extra_share": share, "rule_out_extra_share_ci": share_ci,
        "ship_a_cut": cut_ok,
        "people_flagged_per_death_at_half_caught": ratio,
        "fairness_groups": groups,
        "outcome": "death from any cancer within 60 months, in adults with no cancer "
                   "diagnosis when the blood was drawn",
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  VERDICT: "
          + ("ship the panel" if ship else "do not ship")
          + (", with a rule-out cut" if ship and cut_ok else
             ", with no cut and no reassurance" if ship else ""))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
