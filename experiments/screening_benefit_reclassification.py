"""
Do routine labs change who is likely to live long enough to benefit from screening?

The idea
--------
Every panel in this project tried to detect cancer and failed for solid tumours.
What the same labs are good at is the OTHER side of the ledger: they identify the
heart, kidney and metabolic disease that cause the other deaths. That is exactly
the input a different question needs. Guidelines tell clinicians not to screen for
cancer when a person is unlikely to live ten more years, because screening takes
about that long to pay off and overdiagnosis and procedure harms arrive at once.
The available tools for estimating that (the Lee-Schonberg index is the standard
one) use age, sex, BMI, function and disease history, and no laboratory values. A
lab report is something nearly every older adult already holds.

So the question is not "does this person have cancer". It is: if screening is
withheld from people whose chance of living ten years is low, do the labs find a
different set of people than age and sex alone would?

What is tested
--------------
Train on continuous NHANES 1999-2008 (ten years of follow-up are available) and
apply unchanged to NHANES III, 1988-1994, a different decade, laboratory contract
and population. Outcome: death from any cause within ten years. Nobody is dropped
for how they died.

Calibration is expected to differ: mortality at a given age was higher in the
early 1990s than in the 2000s, so absolute risks will read too low. A decision
tool needs a model fitted to the population it is used on, and this says so
rather than hiding it. The decision question is therefore asked on RANK, not on
absolute risk: in adults 65 to 84, each model flags the same share of people as
least likely to benefit, and the observed ten-year mortality in each flagged group
is compared.

Pre-registered, written before running
--------------------------------------
Routine labs are useful for this only if BOTH hold on NHANES III:
  1. discrimination: labs plus age and sex beat age and sex alone for ten-year
     death, 95% interval on the paired AUC gain excluding zero;
  2. decision zone (ages 65 to 84, flagging the same share under each model): the
     people the lab model flags have higher observed ten-year mortality than the
     people age and sex flag, AND the people it leaves unflagged have lower, each
     with a 95% interval excluding zero.
Condition 2 is the one that matters. A better AUC across all ages can be entirely
a matter of ranking the old above the young and move nobody in the age range where
the screening decision is actually made.

Run:  python experiments/screening_benefit_reclassification.py
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

OUT = "experiments/screening_benefit_reclassification_result.json"
TRAIN, EXTERNAL = "data/nhanes_mortality_full.csv", "data/nhanes3_mortality_full.csv"
HORIZON = 120
CYCLES = ("1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008")
ZONE = (65, 84)
FLAG_SHARE = 0.30
N_BOOT = 2000
DEMO = ["age", "gender"]


def label(df):
    months, died = df["followup_months"], df["died"] == 1
    died_in = died & (months <= HORIZON)
    known = died_in | (months >= HORIZON)
    return died_in.astype(int)[known], known


def fit_best(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best = None
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(y), float(y.mean()))
        a = float(np.mean(cross_val_score(m, X, y, cv=cv, scoring="roc_auc")))
        if best is None or a > best[0]:
            best = (a, kind)
    m = tm.model_factory(best[1], len(y), float(y.mean())).fit(X, y)
    return best[1], m


def paired_auc(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    g = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]


def flagged(p, share):
    return p >= np.quantile(p, 1 - share)


def main():
    tr, te = pd.read_csv(TRAIN), pd.read_csv(EXTERNAL)
    tr = tr[tr["cycle"].isin(CYCLES)]
    skip = {"cancer_death", "followup_months", "cycle", "race_ethnicity", "died",
            "ucod_leading", "smoking", "alcohol_intake", "bmi"}
    labs = [c for c in tr.columns if c in te.columns and c not in skip and c not in DEMO
            and pd.api.types.is_numeric_dtype(tr[c])]
    y_tr, k_tr = label(tr)
    y_te, k_te = label(te)
    tr, te = tr[k_tr].reset_index(drop=True), te[k_te].reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)
    med = tr[DEMO + labs].median()
    Xtr, Xte = tr[DEMO + labs].fillna(med), te[DEMO + labs].fillna(med)
    print(f"Train: NHANES 1999-2008, {len(y_tr):,} adults, {int(y_tr.sum()):,} died within 10 years "
          f"({y_tr.mean():.1%})")
    print(f"Test:  NHANES III 1988-1994, {len(y_te):,} adults, {int(y_te.sum()):,} died "
          f"({y_te.mean():.1%})\n")

    k_a, m_a = fit_best(Xtr[DEMO], y_tr)
    k_l, m_l = fit_best(Xtr, y_tr)
    p_a, p_l = m_a.predict_proba(Xte[DEMO])[:, 1], m_l.predict_proba(Xte)[:, 1]
    auc_a, auc_l = float(roc_auc_score(y_te, p_a)), float(roc_auc_score(y_te, p_l))
    gain_ci = paired_auc(y_te, p_l, p_a)
    ok1 = bool(gain_ci[0] > 0)
    print(f"1. DISCRIMINATION, all ages")
    print(f"   age+sex ({k_a}) {auc_a:.4f}   +labs ({k_l}) {auc_l:.4f}   gain {auc_l - auc_a:+.4f} "
          f"{gain_ci} -> {'PASS' if ok1 else 'FAIL'}")
    print(f"   mean predicted {p_l.mean():.1%} against observed {y_te.mean():.1%}: these models are "
          f"trained with class balancing, so their probabilities are not risks and only RANK is "
          f"used below. A deployed tool would need calibration to the population it serves.\n")

    zone = ((te["age"] >= ZONE[0]) & (te["age"] <= ZONE[1])).to_numpy()
    yz = y_te[zone].to_numpy()
    pa, pl = p_a[zone], p_l[zone]
    # Inside the zone the age range is narrow, so an AUC here is a harder and fairer test
    # than the all-ages figure, which is helped by ranking the old above the young.
    zone_auc_a, zone_auc_l = float(roc_auc_score(yz, pa)), float(roc_auc_score(yz, pl))
    zone_gain_ci = paired_auc(yz, pl, pa, seed=3)
    print(f"   AUC inside ages {ZONE[0]}-{ZONE[1]} only: age+sex {zone_auc_a:.4f}  +labs "
          f"{zone_auc_l:.4f}  gain {zone_auc_l - zone_auc_a:+.4f} {zone_gain_ci}")
    fa, fl = flagged(pa, FLAG_SHARE), flagged(pl, FLAG_SHARE)
    both, only_a, only_l = (fa & fl).sum(), (fa & ~fl).sum(), (~fa & fl).sum()
    print(f"2. DECISION ZONE, ages {ZONE[0]}-{ZONE[1]}: {int(zone.sum()):,} adults, observed ten-year "
          f"mortality {yz.mean():.1%}")
    print(f"   each model flags its highest-risk {FLAG_SHARE:.0%} as least likely to benefit")
    print(f"   flagged by both {both:,}; by age+sex only {only_a:,}; by labs only {only_l:,} "
          f"-> {(only_a + only_l) / max(zone.sum(), 1):.1%} of this age group are classified differently\n")

    def rate(mask):
        return float(yz[mask].mean()) if mask.sum() else float("nan")

    rng = np.random.default_rng(1)
    diffs_f, diffs_u = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yz), len(yz))
        yy, a_, l_ = yz[i], fa[i], fl[i]
        if a_.sum() < 5 or l_.sum() < 5 or (~a_).sum() < 5 or (~l_).sum() < 5:
            continue
        diffs_f.append(yy[l_].mean() - yy[a_].mean())
        diffs_u.append(yy[~l_].mean() - yy[~a_].mean())
    ci_f = [round(float(np.percentile(diffs_f, 2.5)), 4), round(float(np.percentile(diffs_f, 97.5)), 4)]
    ci_u = [round(float(np.percentile(diffs_u, 2.5)), 4), round(float(np.percentile(diffs_u, 97.5)), 4)]
    rf_a, rf_l, ru_a, ru_l = rate(fa), rate(fl), rate(~fa), rate(~fl)
    ok2 = bool(ci_f[0] > 0 and ci_u[1] < 0)
    print(f"   observed ten-year mortality in the FLAGGED group:    age+sex {rf_a:.1%}   labs {rf_l:.1%}   "
          f"difference {rf_l - rf_a:+.1%} {[round(x * 100, 1) for x in ci_f]}")
    print(f"   observed ten-year mortality in the UNFLAGGED group:  age+sex {ru_a:.1%}   labs {ru_l:.1%}   "
          f"difference {ru_l - ru_a:+.1%} {[round(x * 100, 1) for x in ci_u]}")
    print(f"   -> {'PASS' if ok2 else 'FAIL'}")

    # What the people only the labs flag, or only age flags, actually did.
    print(f"\n   people flagged ONLY by labs:      {rate(~fa & fl):.1%} died within 10 years ({int(only_l)} people)")
    print(f"   people flagged ONLY by age+sex:   {rate(fa & ~fl):.1%} died within 10 years ({int(only_a)} people)")

    result = {
        "train_n": int(len(y_tr)), "test_n": int(len(y_te)),
        "discrimination": {"age_sex_auc": round(auc_a, 4), "with_labs_auc": round(auc_l, 4),
                           "gain": round(auc_l - auc_a, 4), "gain_ci": gain_ci, "passes": ok1,
                           "mean_predicted": round(float(p_l.mean()), 4),
                           "observed": round(float(y_te.mean()), 4)},
        "decision_zone": {"ages": list(ZONE), "n": int(zone.sum()), "flag_share": FLAG_SHARE,
                          "zone_auc_age_sex": round(zone_auc_a, 4), "zone_auc_labs": round(zone_auc_l, 4),
                          "zone_auc_gain": round(zone_auc_l - zone_auc_a, 4),
                          "zone_auc_gain_ci": zone_gain_ci,
                          "observed_mortality": round(float(yz.mean()), 4),
                          "classified_differently": round(float((only_a + only_l) / max(zone.sum(), 1)), 4),
                          "flagged_mortality_age_sex": round(rf_a, 4), "flagged_mortality_labs": round(rf_l, 4),
                          "flagged_difference_ci": ci_f,
                          "unflagged_mortality_age_sex": round(ru_a, 4),
                          "unflagged_mortality_labs": round(ru_l, 4), "unflagged_difference_ci": ci_u,
                          "only_labs_n": int(only_l), "only_labs_mortality": round(rate(~fa & fl), 4),
                          "only_age_n": int(only_a), "only_age_mortality": round(rate(fa & ~fl), 4),
                          "passes": ok2},
        "useful_for_screening_benefit": bool(ok1 and ok2)}
    print(f"\n  VERDICT: routine labs are "
          + ("useful for deciding who is unlikely to benefit from screening."
             if ok1 and ok2 else "not shown to change that decision."))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
