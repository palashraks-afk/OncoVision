"""
Free-data ladder for multi-cancer risk: questions (Q), routine checkup (C), phone/watch-type activity (W).
Questions and bars: docs/FREE_DATA_LADDER_PREREG.md (committed before this was run).

Run:  python experiments/free_data_ladder.py     needs data/nhanes_wide.csv.gz and data/nhanes_activity.csv
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "free_data_ladder_result.json")
RNG = np.random.default_rng(20261013)
LABS = ["LBXHGB", "LBXWBCSI", "LBXPLTSI", "LBXRDW", "LBXSAL", "LBXSAPSI", "LBXSGL", "LBXSCR", "LBXSCH"]
BP = ["BPXSY1", "BPXDI1"]
ACT = ["log_tac", "mvpa_min", "sed_frac", "frag"]


def load():
    w = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide.csv.gz"))
    a = pd.read_csv(os.path.join(ROOT, "data", "nhanes_activity.csv"))
    d = w.merge(a.drop(columns=["cycle"]), on="SEQN", how="inner")
    d = d[(d["age"] >= 40) & (d["age"] <= 84) & d["followup_months"].notna()].copy()
    d = d[(d["followup_months"] >= 120) | (d["died"] == 1)]
    d["y"] = ((d["died"] == 1) & (d["ucod_leading"] == 2) & (d["followup_months"] <= 120)).astype(int)
    d["y_other"] = ((d["died"] == 1) & (d["ucod_leading"] != 2) & (d["followup_months"] <= 120)).astype(int)
    d["male"] = (d["gender"] == 1).astype(float)
    ever, now = d["SMQ020"], d["SMQ040"]
    d["former"] = ((ever == 1) & (now == 3)).astype(float)
    d["current"] = ((ever == 1) & now.isin([1, 2])).astype(float)
    d = d[ever.isin([1, 2])]
    d["bmi"] = d["BMXBMI"].clip(15, 60)
    wn, w1 = d["WHD020"].where(d["WHD020"].between(50, 600)), d["WHD050"].where(d["WHD050"].between(50, 600))
    d["wt_change"] = ((wn - w1) / w1).clip(-0.3, 0.3)
    d["unint_loss"] = ((d["wt_change"] <= -0.05) & (d["WHQ070"] != 1)).astype(float)
    d["wt_change"] = d["wt_change"].fillna(0)
    d["log_tac"] = np.log(d["tac"].clip(lower=1000))
    d["mvpa_min"] = d["mvpa_min"].clip(0, 180)
    for c in LABS + BP:
        d[c] = d[c].fillna(d[c].median())
    d = d.dropna(subset=["bmi"]).reset_index(drop=True)
    return d


def X(d, layers, st):
    cols = [st.transform(d[["age"]].to_numpy()), d[["male", "former", "current", "bmi", "wt_change", "unint_loss"]].to_numpy()]
    if "C" in layers:
        lab = d[LABS + BP].copy()
        lab["LBXSAPSI"] = np.log(lab["LBXSAPSI"].clip(lower=10))
        lab["LBXSGL"] = np.log(lab["LBXSGL"].clip(lower=40))
        lab["LBXSCR"] = np.log(lab["LBXSCR"].clip(lower=0.2))
        cols.append(lab.to_numpy())
    if "W" in layers:
        cols.append(d[ACT].to_numpy())
    return np.column_stack(cols)


def oof(d, layers, outcome="y", repeats=20):
    """Out-of-fold predictions averaged over repeated 5-fold splits."""
    y = d[outcome].to_numpy()
    acc = np.zeros(len(d))
    for r in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=1000 + r)
        for tr, te in skf.split(d, y):
            st = SplineTransformer(degree=3, n_knots=4, knots="quantile", extrapolation="constant", include_bias=False).fit(d[["age"]].iloc[tr])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X(d.iloc[tr], layers, st), y[tr])
            acc[te] += m.predict_proba(X(d.iloc[te], layers, st))[:, 1]
    return acc / repeats


def gain(y, a, b, n=1000):
    g = roc_auc_score(y, b) - roc_auc_score(y, a)
    bs = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        bs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return {"gain": round(float(g), 4), "ci": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]}


def slope(y, p):
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    return round(float(LogisticRegression(C=1e6).fit(lp, y).coef_[0, 0]), 3)


def main():
    d = load()
    y = d["y"].to_numpy()
    print(f"NHANES 2003-2006 with valid activity data, ages 40-84: {len(d):,} adults, {y.sum()} cancer deaths and {int(d['y_other'].sum())} other deaths within ten years")
    print(f"unintentional weight loss flag: {d['unint_loss'].mean():.1%} of adults")
    res = {"n": int(len(d)), "cancer_deaths": int(y.sum()), "other_deaths": int(d["y_other"].sum()), "models": {}, "gains": {}}
    P = {}
    for L in ("Q", "QC", "QW", "QCW"):
        P[L] = oof(d, L)
        res["models"][L] = {"auc": round(float(roc_auc_score(y, P[L])), 4), "slope": slope(y, P[L])}
        print(f"  {L:<4} AUC {res['models'][L]['auc']:.4f}  slope {res['models'][L]['slope']}")
    for a, b in (("Q", "QW"), ("Q", "QC"), ("QC", "QCW")):
        res["gains"][f"{b} vs {a}"] = gain(y, P[a], P[b])
        print(f"  {b} vs {a}: {res['gains'][f'{b} vs {a}']}")
    # specificity check: other-cause death
    yo = d["y_other"].to_numpy()
    Po = {L: oof(d, L, "y_other", 10) for L in ("Q", "QW")}
    res["other_cause"] = {"auc_Q": round(float(roc_auc_score(yo, Po["Q"])), 4), "auc_QW": round(float(roc_auc_score(yo, Po["QW"])), 4), "gain": gain(yo, Po["Q"], Po["QW"])}
    print("  other-cause death, QW vs Q:", res["other_cause"])
    g_c, g_o = res["gains"]["QW vs Q"]["gain"], res["other_cause"]["gain"]["gain"]
    res["bars"] = {"1_W_adds": bool(g_c >= 0.01 and res["gains"]["QW vs Q"]["ci"][0] > 0),
                   "2_C_adds": bool(res["gains"]["QC vs Q"]["gain"] >= 0.01 and res["gains"]["QC vs Q"]["ci"][0] > 0),
                   "3_cancer_not_just_frailty": bool(g_c >= 0.5 * g_o) if g_o > 0 else True}
    # exploratory: 65+, single features, weight-loss flag
    old = d["age"].to_numpy() >= 65
    res["age65plus"] = {"n": int(old.sum()), "cancer_deaths": int(y[old].sum()), "auc_Q": round(float(roc_auc_score(y[old], P["Q"][old])), 4), "auc_QW": round(float(roc_auc_score(y[old], P["QW"][old])), 4)}
    res["single_W"] = {}
    for f in ACT:
        z = (d[f] - d[f].mean()) / d[f].std()
        q1, q4 = z <= z.quantile(0.25), z >= z.quantile(0.75)
        res["single_W"][f] = {"cancer_rate_bottom_quarter": round(float(y[q1].mean()), 4), "cancer_rate_top_quarter": round(float(y[q4].mean()), 4)}
    u = d["unint_loss"].to_numpy() == 1
    res["unintentional_weight_loss"] = {"n": int(u.sum()), "cancer_rate": round(float(y[u].mean()), 4), "rate_others": round(float(y[~u].mean()), 4)}
    print("  65+:", res["age65plus"]); print("  single W:", res["single_W"]); print("  weight loss:", res["unintentional_weight_loss"])
    print("bars:", res["bars"])
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
