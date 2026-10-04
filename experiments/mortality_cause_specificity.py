"""
Is the cancer-mortality panel reading cancer, or reading "this person is unwell"?

The suspicion
-------------
The panel ships as a cancer panel. Its outcome is death from cancer within five
years, and its cohort was built by keeping two kinds of people: those who died
of cancer inside the window, and those who lived past it. Anyone who died of
something else inside the window was DROPPED (fetch_nhanes_mortality.py,
`keep = positive | followed`).

That makes every early death in the training file a cancer death. A model can
then score well by recognising who looks close to dying -- low albumin, a high
RDW, a raised white count are textbook markers of that, and they predict death
from heart disease or infection just as well. The external cohort was built the
same way, so testing on it could not catch this: the bias is in the design, and
it travels with the data. A passing external result proved the panel transfers.
It did not prove the panel is about cancer.

Four measurements, all on cohorts that keep everybody
-----------------------------------------------------
    original     the shipped definition: cancer death vs people who lived past
                 five years. Reproduced, so the rest is comparable to it.
    honest       cancer death vs EVERYONE else, including people who died of
                 other causes inside the window. This is the question the card
                 claims to answer.
    all-cause    any death within five years vs people who lived past it. If
                 the panel does as well here as on cancer, it is a mortality
                 score with a cancer label.
    decedents    among ONLY the people who died within five years, cancer vs
                 any other cause. Age and sex cannot separate these two groups,
                 so any gain here is cancer-specific information. This is the
                 cleanest test, and the one that cannot be explained away.

Pre-registered, written before the cohorts finished downloading
---------------------------------------------------------------
The panel stays a CANCER panel only if BOTH hold on NHANES III, with the model
trained on the continuous survey and nothing else changed:

  1. honest: its gain over the stronger of a logistic and an ensemble model on
     age and sex has a 95% interval excluding zero; and
  2. decedents: among people who died, its gain over age and sex at telling a
     cancer death from another death has a 95% interval excluding zero.

If either fails the cancer claim is not supported and the panel is withdrawn,
not relabelled quietly. Whether a relabelled all-cause mortality score is worth
shipping is a different question, decided separately and not by this script.

Run:  python fetch_nhanes_mortality.py --full
      python fetch_nhanes3_external.py --full
      python experiments/mortality_cause_specificity.py
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

warnings.filterwarnings("ignore")

OUT = "experiments/mortality_cause_specificity_result.json"
TRAIN = "data/nhanes_mortality_full.csv"
EXTERNAL = "data/nhanes3_mortality_full.csv"
HORIZON = 60
CANCER = 2
N_BOOT = 2000
MIN_EVENTS = 20

DEMO = ["age", "gender"]


def outcomes(df):
    """Each outcome as (label series, row mask). NaN labels are excluded."""
    months = df["followup_months"]
    died = df["died"] == 1
    cancer = died & (df["ucod_leading"] == CANCER)
    died_in = died & (months <= HORIZON)
    cancer_in = cancer & (months <= HORIZON)
    other_in = died_in & ~cancer_in
    # Alive, or died after the horizon: the outcome at five years is known.
    past = months >= HORIZON
    # Alive and followed for less than five years: unknown, never a negative.
    known = died_in | past

    out = {}
    out["original"] = (cancer_in.astype(int), cancer_in | (past & ~cancer_in))
    out["honest"] = (cancer_in.astype(int), known)
    out["all_cause"] = (died_in.astype(int), known)
    # Among decedents only. Positive is a cancer death, negative any other death.
    out["decedents"] = (cancer_in.astype(int), died_in)
    out["other_cause_only"] = (other_in.astype(int), other_in | (past & ~cancer_in))
    return out


def paired_boot(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return (round(float(np.mean(out)), 4),
            [round(float(np.percentile(out, 2.5)), 4),
             round(float(np.percentile(out, 97.5)), 4)])


def auc_ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        v.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def fit_best(X_tr, y_tr):
    """The stronger of the two model kinds by cross-validated AUC, refitted."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best = None
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(y_tr), float(y_tr.mean()))
        a = float(np.mean(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")))
        if best is None or a > best[0]:
            best = (a, kind)
    m = tm.model_factory(best[1], len(y_tr), float(y_tr.mean()))
    m.fit(X_tr, y_tr)
    return best[0], best[1], m


def main():
    for p in (TRAIN, EXTERNAL):
        if not os.path.exists(p):
            print(f"{p} is missing. Run the --full fetchers first.")
            return 1
    tr, te = pd.read_csv(TRAIN), pd.read_csv(EXTERNAL)
    skip = {"cancer_death", "followup_months", "cycle", "race_ethnicity", "died",
            "ucod_leading"}
    feats = [c for c in tr.columns if c in te.columns and c not in skip
             and pd.api.types.is_numeric_dtype(tr[c])]
    med = tr[feats].median()
    Xtr_all, Xte_all = tr[feats].fillna(med), te[feats].fillna(med)
    o_tr, o_te = outcomes(tr), outcomes(te)

    print(f"Train: continuous NHANES, {len(tr):,} adults kept "
          f"({int((tr['died'] == 1).sum()):,} died by 2019)")
    print(f"Test:  NHANES III, {len(te):,} adults kept "
          f"({int((te['died'] == 1).sum()):,} died by 2019)")
    print(f"{len(feats)} shared inputs\n")

    result = {"n_train": int(len(tr)), "n_test": int(len(te)),
              "features": feats, "outcomes": {}}

    # 1. The shipped definition, fitted once. Reused below to ask what ELSE that
    # model predicts.
    y_o, m_o = o_tr["original"]
    shipped_cv, shipped_kind, shipped = fit_best(Xtr_all[m_o.values], y_o[m_o.values])

    for name in ("original", "honest", "all_cause", "decedents"):
        y_tr, m_tr = o_tr[name]
        y_te, m_te = o_te[name]
        Xa, ya = Xtr_all[m_tr.values], y_tr[m_tr.values]
        Xb, yb = Xte_all[m_te.values], y_te[m_te.values].to_numpy()
        if ya.sum() < MIN_EVENTS or yb.sum() < MIN_EVENTS:
            result["outcomes"][name] = {"skipped": "too few events",
                                        "train_events": int(ya.sum()),
                                        "test_events": int(yb.sum())}
            print(f"  {name}: too few events ({int(ya.sum())} / {int(yb.sum())})")
            continue
        cv_p, kind_p, panel = fit_best(Xa, ya)
        cv_b, kind_b, base = fit_best(Xa[DEMO], ya)
        p = panel.predict_proba(Xb)[:, 1]
        b = base.predict_proba(Xb[DEMO])[:, 1]
        auc, bauc = float(roc_auc_score(yb, p)), float(roc_auc_score(yb, b))
        gain, gci = paired_boot(yb, p, b)
        entry = {
            "train_n": int(len(ya)), "train_events": int(ya.sum()),
            "test_n": int(len(yb)), "test_events": int(yb.sum()),
            "panel_kind": kind_p, "panel_auc": round(auc, 4),
            "panel_auc_ci": auc_ci(yb, p),
            "age_sex_auc": round(bauc, 4), "gain": gain, "gain_ci": gci,
            "beats_age_sex": bool(gci[0] > 0)}
        result["outcomes"][name] = entry
        print(f"  {name:<10} train {len(ya):>6,} ({int(ya.sum())} events)  "
              f"NHANES III {len(yb):>6,} ({int(yb.sum())} events)\n"
              f"             panel {auc:.4f} {entry['panel_auc_ci']}   age+sex {bauc:.4f}   "
              f"gain {gain:+.4f} ({gci[0]:+.4f} to {gci[1]:+.4f}) -> "
              f"{'BEATS age and sex' if gci[0] > 0 else 'not shown'}")

    # 2. What the SHIPPED model predicts, scored against other outcomes on the
    # same external people. A cancer panel should do best on cancer.
    print("\n  The shipped model (fitted on the original definition), scored on "
          "NHANES III against each outcome:")
    cross = {}
    for name in ("original", "honest", "all_cause", "other_cause_only"):
        y_te, m_te = o_te[name]
        yb = y_te[m_te.values].to_numpy()
        if yb.sum() < MIN_EVENTS:
            continue
        p = shipped.predict_proba(Xte_all[m_te.values])[:, 1]
        a = float(roc_auc_score(yb, p))
        cross[name] = {"auc": round(a, 4), "auc_ci": auc_ci(yb, p),
                       "events": int(yb.sum())}
        print(f"    {name:<18} AUC {a:.4f} {cross[name]['auc_ci']}  ({int(yb.sum())} events)")
    result["shipped_model_cross_outcome"] = cross
    result["shipped_model_kind"] = shipped_kind

    # 3. The decision, by the bar written above.
    h = result["outcomes"].get("honest", {})
    d = result["outcomes"].get("decedents", {})
    honest_ok = bool(h.get("beats_age_sex"))
    decedents_ok = bool(d.get("beats_age_sex"))
    result["honest_passes"] = honest_ok
    result["decedents_pass"] = decedents_ok
    result["stays_a_cancer_panel"] = bool(honest_ok and decedents_ok)
    inflated = None
    if "original" in result["outcomes"] and "honest" in result["outcomes"]:
        inflated = round(result["outcomes"]["original"].get("panel_auc", 0)
                         - result["outcomes"]["honest"].get("panel_auc", 0), 4)
    result["auc_lost_when_other_deaths_are_kept"] = inflated
    print(f"\n  BAR 1, honest outcome beats age and sex externally: "
          f"{'PASS' if honest_ok else 'FAIL'}")
    print(f"  BAR 2, among people who died, tells cancer from other causes better "
          f"than age and sex: {'PASS' if decedents_ok else 'FAIL'}")
    if inflated is not None:
        print(f"  AUC lost by keeping the people the original file dropped: {inflated:+.4f}")
    print(f"\n  VERDICT: " + ("it stays a cancer panel." if result["stays_a_cancer_panel"]
          else "the cancer claim is NOT supported. Withdraw it as a cancer panel."))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
