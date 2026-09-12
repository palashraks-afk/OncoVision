"""
Does a panel still work on a later survey, and does its rule-out cut still keep
its promise there?

Why this exists
---------------
Section 4.2 of the paper is the reason. Resampling inside one survey said the
prospective panel scored 0.837; an actual external cohort reversed the sign of
the effect. So a number that has only ever been measured by splitting one
dataset is not evidence, and three of the shipped operating points had never
been measured any other way.

The strongest test is another SOURCE -- another country, another laboratory,
another referral pattern. The bowel panel has one, NHANES III, and the liver
panel has India and Germany. For the rest there is no public cohort that
carries the same measurements against the same outcome.

The always-available substitute is time. NHANES runs in two-year cycles with
the same protocol but different laboratory runs, different instrument lots, a
different sample of the country and a slowly moving population. Withholding the
most recent cycle from training gives a test set that shares the protocol and
shares nothing else -- no rows, no calibration, no threshold. TRIPOD calls this
temporal validation and counts it as external validation of the weaker kind.

It is the weaker kind and it is not written up as anything else. A panel that
passes here has shown it survives a change of laboratory run and two years of
drift. It has NOT shown it survives a change of country, and the liver panel is
the standing proof that those are different questions: it scores 0.442 on the
German cohort, below chance.

What is withheld
----------------
train_models.TEMPORAL_HOLDOUT, currently the 2017-2018 cycle for liver and
lung. Those rows are removed in prepare(), so they touch neither fitting nor
calibration nor threshold selection. The bowel and general panels cannot join:
CDC dropped the MCQ240 age-at-diagnosis series after 2015-2016, so their
screening-window target cannot be built on later data at all.

Judged on
---------
    discrimination   AUC on the withheld cycle, against the shipped figure
    gain             AUC over an age-and-sex model fitted on the same training
                     rows, because raw AUC mostly measures how old people are
    rule-out         the cut is applied UNCHANGED and asked whether it caught
                     the share of cases it promised

Run:  python experiments/temporal_validation.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/temporal_validation_result.json"

# How far the delivered sensitivity may fall below the promised one before the
# cut counts as broken. Set here, before the numbers are known, and the same
# 5-point tolerance the bowel cut was held to on NHANES III.
TOLERANCE = 0.05


def age_sex_baseline(X_tr, y_tr, X_te):
    """An age-and-sex model fitted on the training rows only."""
    cols = [c for c in ("age", "gender") if c in X_tr.columns]
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(X_tr[cols], y_tr)
    return lr.predict_proba(X_te[cols])[:, 1]


def score(bundle, X):
    """Score exactly the way the service does, clipping included.

    The application refuses to rank a value outside the range it has evidence
    about, because tree models cannot extrapolate and were found rating
    fulminant hepatitis safer than health. Scoring the holdout without that
    clipping would measure a model the users never meet.
    """
    X = X.copy()
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    X = X[bundle["feature_names"]]
    return bundle["model"].predict_proba(X)[:, 1]


def run_panel(name):
    cfg = next(c for c in tm.DATASETS if c["name"] == name)
    df = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    train_df, held_df = tm.split_temporal(df, name)
    if held_df is None or held_df.empty:
        return None

    bundle = joblib.load(f"backend/models/model_{name}.joblib")
    medians = pd.Series(bundle["feature_medians"])

    def matrix(frame):
        X = tm.build_features(cfg, frame)
        return X.fillna(medians).fillna(0.0)

    X_tr, y_tr = matrix(train_df), cfg["target"](train_df)
    keep_tr = y_tr.notna()
    X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr].astype(int)

    X_te, y_te = matrix(held_df), cfg["target"](held_df)
    keep = y_te.notna()
    X_te, y_te = X_te[keep], y_te[keep].astype(int).reset_index(drop=True)

    if y_te.sum() < 5:
        return {"panel": name, "too_few_events": int(y_te.sum())}

    p = score(bundle, X_te)
    base = age_sex_baseline(X_tr, y_tr, X_te)

    auc = float(roc_auc_score(y_te, p))
    base_auc = float(roc_auc_score(y_te, base))
    shipped = bundle.get("held_out", {}).get("auc") or bundle["metrics"]["auc"]

    # The gain is the claim, so the gain is what needs an interval. Resampling
    # the two scores TOGETHER on the same patients keeps the comparison paired,
    # which matters on a panel with thirteen events where the two curves are
    # ranking the same handful of people.
    rng = np.random.default_rng(0)
    yv, pv, bv = np.asarray(y_te), np.asarray(p), np.asarray(base)
    gains = []
    for _ in range(2000):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        gains.append(roc_auc_score(yv[idx], pv[idx]) - roc_auc_score(yv[idx], bv[idx]))
    lo, hi = (round(float(np.percentile(gains, 2.5)), 3),
              round(float(np.percentile(gains, 97.5)), 3)) if gains else (None, None)

    out = {
        "panel": name,
        "held_out_cycle": tm.TEMPORAL_HOLDOUT[name],
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "events_test": int(y_te.sum()),
        "prevalence_test": round(float(y_te.mean()), 4),
        "auc": round(auc, 3),
        "auc_ci": bootstrap_ci(np.asarray(y_te), np.asarray(p), roc_auc_score),
        "age_sex_auc": round(base_auc, 3),
        "gain": round(auc - base_auc, 3),
        "gain_ci": [lo, hi],
        "gain_beats_demographics": bool(lo is not None and lo > 0),
        "shipped_auc": round(float(shipped), 3),
        "drop_vs_shipped": round(float(shipped) - auc, 3),
    }

    # The cut is applied unchanged. Nothing about the holdout is allowed to
    # move it, which is the only way the promise means anything.
    ro = (bundle.get("metrics") or {}).get("rule_out")
    if ro:
        cut = ro["threshold"]
        ruled_out = p < cut
        missed = int((ruled_out & (y_te == 1)).sum())
        delivered_sens = float(1 - missed / max(int(y_te.sum()), 1))
        out["rule_out"] = {
            "threshold": cut,
            "promised_sensitivity": ro["sensitivity"],
            "delivered_sensitivity": round(delivered_sens, 3),
            "promised_share_ruled_out": ro["share_ruled_out"],
            "delivered_share_ruled_out": round(float(ruled_out.mean()), 3),
            "cases_wrongly_ruled_out": missed,
            "within_tolerance": bool(delivered_sens >= ro["sensitivity"] - TOLERANCE),
        }
    else:
        out["rule_out"] = None
        out["no_rule_out_reason"] = (bundle.get("metrics") or {}).get("no_rule_out_reason", "")
    return out


def main():
    print("Temporal validation: the withheld later cycle\n")
    results = {}
    for name in tm.TEMPORAL_HOLDOUT:
        r = run_panel(name)
        if r is None:
            continue
        results[name] = r
        if "too_few_events" in r:
            print(f"  {name}: {r['too_few_events']} events, too few to measure")
            continue
        print(f"  {name:<10} cycle {r['held_out_cycle']}  n={r['n_test']}  "
              f"events={r['events_test']}")
        print(f"      AUC {r['auc']:.3f} {r['auc_ci']}  age/sex {r['age_sex_auc']:.3f}  "
              f"gain {r['gain']:+.3f} {tuple(r['gain_ci'])}")
        print(f"      shipped {r['shipped_auc']:.3f}, drop {r['drop_vs_shipped']:+.3f}")
        ro = r.get("rule_out")
        if ro:
            verdict = "within tolerance" if ro["within_tolerance"] else "BROKEN"
            print(f"      rule-out promised {ro['promised_sensitivity']:.3f} sens / "
                  f"{ro['promised_share_ruled_out']:.1%} excluded, delivered "
                  f"{ro['delivered_sensitivity']:.3f} / "
                  f"{ro['delivered_share_ruled_out']:.1%} -> {verdict}")
        else:
            print("      no rule-out cut ships for this panel")
        print()

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
