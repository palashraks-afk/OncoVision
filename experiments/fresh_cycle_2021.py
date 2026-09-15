"""
The lung and liver panels on NHANES 2021-2023, a cohort nothing here has seen.

Why this exists
---------------
Two open problems. Lung's gain over age and sex was +0.030 on leave-one-cycle-out
with an interval touching zero, and only 13 events in the withheld 2017-2018
cycle. And both panels' published held-out AUCs were lucky draws, at the top of
their split distributions. A later cycle, collected after the pandemic pause on a
new laboratory contract, answers both without anyone choosing a split.

Pre-registered, written before the 2021-2023 numbers were seen
--------------------------------------------------------------
PRIMARY (lung): the shipped model, trained on 1999-2016 only, scores the
2017-2018 and 2021-2023 participants pooled. Neither touched fitting, calibration
or threshold choice. Its gain over the STRONGER of two age-and-sex models
(logistic and ensemble, both fitted on the same training rows) is resampled
paired. The gain counts as confirmed only if the lower 95% bound is above zero.
Choosing the stronger baseline on the test rows is deliberately conservative: it
can only shrink the gain.

SECONDARY: each cycle alone; the lung gain over the questionnaire (age, sex,
smoking, pack-years); liver AUC and gain; each shipped rule-out cut applied
unchanged, held to the same 5-point tolerance as temporal_validation.py.

Run:  python experiments/fresh_cycle_2021.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/fresh_cycle_2021_result.json"
FRESH = {"lung": "data/nhanes_lung_smokers_2021.csv", "liver": "data/nhanes_liver_2021.csv"}
TOLERANCE = 0.05
N_BOOT = 2000


def score(bundle, X):
    X = X.copy()
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    return bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]


def paired_gain_ci(y, a, b):
    rng = np.random.default_rng(0)
    y, a, b = map(np.asarray, (y, a, b))
    gains = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(gains, 2.5)), 3), round(float(np.percentile(gains, 97.5)), 3)]


def fit_baselines(X_tr, y_tr, cols):
    rate = float(y_tr.mean())
    out = {}
    for kind in ("logistic", "ensemble"):
        m = tm.model_factory(kind, len(y_tr), rate)
        m.fit(X_tr[cols], y_tr)
        out[kind] = m
    return out


def evaluate_set(label, y, p, baselines, X, bundle):
    base_scores = {k: m.predict_proba(X[c])[:, 1] for k, (m, c) in baselines.items()}
    r = {"cycle": label, "n": int(len(y)), "events": int(y.sum()),
         "auc": round(float(roc_auc_score(y, p)), 3),
         "auc_ci": bootstrap_ci(np.asarray(y), np.asarray(p), roc_auc_score)}
    demo = {k: v for k, v in base_scores.items() if k.startswith("age_sex")}
    best = max(demo, key=lambda k: roc_auc_score(y, demo[k]))
    r["age_sex_baseline"] = best
    r["age_sex_auc"] = round(float(roc_auc_score(y, demo[best])), 3)
    r["gain"] = round(r["auc"] - r["age_sex_auc"], 3)
    r["gain_ci"] = paired_gain_ci(y, p, demo[best])
    if "questionnaire" in base_scores:
        q = base_scores["questionnaire"]
        r["questionnaire_auc"] = round(float(roc_auc_score(y, q)), 3)
        r["gain_over_questionnaire"] = round(r["auc"] - r["questionnaire_auc"], 3)
        r["gain_over_questionnaire_ci"] = paired_gain_ci(y, p, q)
    ro = (bundle.get("metrics") or {}).get("rule_out")
    if ro:
        out = p < ro["threshold"]
        missed = int((out & (np.asarray(y) == 1)).sum())
        sens = 1 - missed / max(int(y.sum()), 1)
        r["rule_out"] = {"promised_sensitivity": ro["sensitivity"],
                         "delivered_sensitivity": round(float(sens), 3),
                         "promised_share_ruled_out": ro["share_ruled_out"],
                         "delivered_share_ruled_out": round(float(out.mean()), 3),
                         "within_tolerance": bool(sens >= ro["sensitivity"] - TOLERANCE)}
    return r


def run(name, stability):
    cfg = next(c for c in tm.DATASETS if c["name"] == name)
    train_df, held_df = tm.split_temporal(pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"])), name)
    fresh_df = pd.read_csv(FRESH[name])
    bundle = joblib.load(f"backend/models/model_{name}.joblib")
    med = pd.Series(bundle["feature_medians"])

    def xy(frame):
        X = tm.build_features(cfg, frame).fillna(med).fillna(0.0)
        y = cfg["target"](frame)
        return X.reset_index(drop=True), pd.Series(np.asarray(y), dtype=int)

    X_tr, y_tr = xy(train_df)
    sets = {"2017-2018": xy(held_df), "2021-2023": xy(fresh_df)}
    sets["pooled"] = (pd.concat([sets["2017-2018"][0], sets["2021-2023"][0]], ignore_index=True),
                      pd.concat([sets["2017-2018"][1], sets["2021-2023"][1]], ignore_index=True))

    demo_cols = ["age", "gender"]
    fitted = fit_baselines(X_tr, y_tr, demo_cols)
    baselines = {f"age_sex_{k}": (m, demo_cols) for k, m in fitted.items()}
    if name == "lung":
        qcols = ["age", "gender", "smoking", "smoking_packyears"]
        q = tm.model_factory("logistic", len(y_tr), float(y_tr.mean())).fit(X_tr[qcols], y_tr)
        baselines["questionnaire"] = (q, qcols)

    res = {"panel": name, "model_kind": bundle.get("model_kind") or bundle.get("kind"),
           "n_train": int(len(y_tr))}
    for label, (X, y) in sets.items():
        res[label] = evaluate_set(label, y, score(bundle, X), baselines, X, bundle)
    s = stability.get(name) or {}
    res["split_distribution"] = {"median": s.get("median_auc"), "range": s.get("split_range_2p5_97p5"),
                                 "published_split": s.get("shipped_split_auc")}
    f = res["2021-2023"]["auc"]
    rng = s.get("split_range_2p5_97p5")
    res["fresh_inside_split_range"] = bool(rng and rng[0] <= f <= rng[1])
    if name == "lung":
        lo = res["pooled"]["gain_ci"][0]
        res["primary_confirmed"] = bool(lo > 0)
    return res


def main():
    stab = json.load(open("experiments/split_stability_result.json"))
    stab = stab.get("panels", stab)
    results = {}
    for name in ("lung", "liver"):
        r = run(name, stab)
        results[name] = r
        print(f"\n{name}  (trained on {r['n_train']:,} rows, 1999-2016 only)")
        for label in ("2017-2018", "2021-2023", "pooled"):
            e = r[label]
            line = (f"  {label:<10} n={e['n']:>6,} events={e['events']:>4}  AUC {e['auc']:.3f} "
                    f"{e['auc_ci']}  age/sex({e['age_sex_baseline'][8:]}) {e['age_sex_auc']:.3f}  "
                    f"gain {e['gain']:+.3f} {e['gain_ci']}")
            if "questionnaire_auc" in e:
                line += (f"  | vs questionnaire {e['gain_over_questionnaire']:+.3f} "
                         f"{e['gain_over_questionnaire_ci']}")
            print(line)
            if e.get("rule_out"):
                ro = e["rule_out"]
                print(f"             rule-out promised {ro['promised_sensitivity']} / "
                      f"{ro['promised_share_ruled_out']:.1%}, delivered {ro['delivered_sensitivity']} / "
                      f"{ro['delivered_share_ruled_out']:.1%} -> "
                      f"{'held' if ro['within_tolerance'] else 'BROKEN'}")
        sd = r["split_distribution"]
        print(f"  split distribution median {sd['median']} range {sd['range']}, published {sd['published_split']}; "
              f"2021-2023 inside range: {r['fresh_inside_split_range']}")
        if name == "lung":
            print(f"  PRIMARY (pooled gain lower bound > 0): "
                  f"{'CONFIRMED' if r['primary_confirmed'] else 'NOT CONFIRMED'}")
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
