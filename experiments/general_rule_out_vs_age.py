"""
Does the general panel's rule-out cut beat a cut on age and sex alone?

The test that decided two other panels
--------------------------------------
Bowel failed it: at matched sensitivity, age and sex alone excluded as many
people and missed fewer cancers, and the panel was withdrawn. The mammogram
breast panel passed it by 9.8 points. The general panel is the only other panel
that ships a rule-out call, and it is the weakest on every other measure: its
gain over age and sex is about +0.002, and on NHANES III its cut caught 90.5%
of cancers against 95.1% promised.

Two settings, because an in-survey result has flattered this project before:

    in-survey   out-of-fold scores on the 28,711-adult training cohort
    external    fitted on that cohort, applied to 15,896 NHANES III adults

In each, the question is the same: at the SAME share of cancers caught, which
cut excludes more people? Matching sensitivity stops either cut looking better
only because it sits at a different point on its curve.

The age-and-sex arm is the better of logistic regression and the ensemble, the
strongest baseline, as everywhere else now. The panel keeps its rule-out call
only if it excludes at least one more person in a hundred than age and sex do,
with an interval above zero, in BOTH settings. Set before the numbers were known.

Run:  python experiments/general_rule_out_vs_age.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/general_rule_out_vs_age_result.json"
NAME = "general"
TEST = "data/nhanes3_general.csv"
TARGET = "recent_cancer"
BASE = ["age", "gender"]
MATERIAL = 0.01
BOOT = 500


def exclusion_at_sensitivity(y, p, target):
    df = pd.DataFrame({"p": p, "pos": (y == 1).astype(float), "all": 1.0})
    g = df.groupby("p", sort=True)[["pos", "all"]].sum()
    ok = g["pos"].cumsum() <= (1.0 - target) * g["pos"].sum() + 1e-9
    return float(g["all"].cumsum()[ok].iloc[-1] / g["all"].sum()) if ok.any() else 0.0


def calibrated(kind, n, rate, seed=0):
    return CalibratedClassifierCV(tm.model_factory(kind, n, rate), method="isotonic",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def compare(y, p_panel, p_base, target, label):
    rng = np.random.default_rng(0)
    m_panel = exclusion_at_sensitivity(y, p_panel, target)
    m_base = exclusion_at_sensitivity(y, p_base, target)
    diffs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        diffs.append(exclusion_at_sensitivity(y[i], p_panel[i], target)
                     - exclusion_at_sensitivity(y[i], p_base[i], target))
    ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    diff = m_panel - m_base
    print(f"  {label}: at {target:.1%} of cancers caught, panel excludes {m_panel:.1%}, "
          f"age and sex {m_base:.1%}  ->  {diff:+.1%} (95% CI {ci[0]:+.1%} to {ci[1]:+.1%})",
          flush=True)
    return {"matched_sensitivity": round(target, 3),
            "panel_excluded": round(m_panel, 4), "age_sex_excluded": round(m_base, 4),
            "difference": round(diff, 4), "difference_ci": ci,
            "panel_beats_age_sex": bool(ci[0] > 0 and diff >= MATERIAL),
            "panel_auc": round(float(roc_auc_score(y, p_panel)), 3),
            "age_sex_auc": round(float(roc_auc_score(y, p_base)), 3)}


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    X, y, med = tm.prepare(cfg)
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True).fillna(med)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    yv, rate = y.to_numpy(), float(y.mean())
    kind, _ = tm.select_model(X, y, rate)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p_panel = cross_val_predict(calibrated(kind, len(y), rate), X, y, cv=cv,
                                method="predict_proba")[:, 1]
    base_oof = {k: cross_val_predict(calibrated(k, len(y), rate), X[BASE], y, cv=cv,
                                     method="predict_proba")[:, 1]
                for k in ("logistic", "ensemble")}
    base_kind = max(base_oof, key=lambda k: roc_auc_score(yv, base_oof[k]))
    p_base = base_oof[base_kind]

    panel_ro = tm.rule_out_threshold(yv, p_panel)
    target_in = panel_ro["sensitivity"] if panel_ro else tm.RULE_OUT_SENSITIVITY
    print(f"general panel ({kind}) against age and sex ({base_kind}), "
          f"{len(y):,} adults, {int(y.sum())} cancers\n")
    inside = compare(yv, p_panel, p_base, target_in, "in-survey")

    te = pd.read_csv(TEST)
    yt = te[TARGET].astype(int).to_numpy()

    def frame(cols):
        return pd.DataFrame({c: (pd.to_numeric(te[c], errors="coerce") if c in te.columns
                                 else np.nan) for c in cols}).fillna(med[cols])

    panel_fit = calibrated(kind, len(y), rate).fit(X, y)
    base_fit = calibrated(base_kind, len(y), rate).fit(X[BASE], y)
    pt_panel = panel_fit.predict_proba(frame(list(X.columns)))[:, 1]
    pt_base = base_fit.predict_proba(frame(BASE))[:, 1]
    # Matched at the sensitivity the panel's in-survey cut promised.
    external = compare(yt, pt_panel, pt_base, target_in, "NHANES III")

    keeps = inside["panel_beats_age_sex"] and external["panel_beats_age_sex"]
    print(f"\n  -> {'the general panel earns its rule-out call' if keeps else 'THE GENERAL PANEL DOES NOT BEAT AGE AND SEX AT ITS RULE-OUT CUT'}")

    with open(OUT, "w") as f:
        json.dump({"panel_model": kind, "baseline_model": base_kind,
                   "in_survey": inside, "external": external,
                   "material_threshold": MATERIAL,
                   "panel_beats_age_sex_in_both": bool(keeps)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
