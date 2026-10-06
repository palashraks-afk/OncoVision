"""
Risk-ranking study. Questions, feature sets, split and bars are in
docs/RISK_RANKING_PREREG.md and were committed before this was run.

Does everything a survey knows about a person rank who will die of cancer within five
years better than the obvious baseline (age, sex, smoking, body mass index), on an era
the model never saw?

Run:  WIDE_CYCLES=8 python fetch_nhanes_wide.py   (once)
      python experiments/risk_ranking.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "risk_ranking_result.json")
RNG = np.random.default_rng(20261006)
CANCER = 2
META = {"SEQN", "died", "ucod_leading", "followup_months", "cycle"}
ROUTINE = ["LBXHGB", "LBXHCT", "LBXRBCSI", "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMPSI", "LBXWBCSI",
           "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT"]


def logreg(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)


def boot_gain(y, a, b, n=1000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    g = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        g.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return [round(float(x), 4) for x in np.percentile(g, [2.5, 97.5])]


def slope(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return round(float(m.coef_[0][0]), 3), round(float(m.intercept_[0]), 3)


def capture(y, s, frac):
    k = int(round(frac * len(y)))
    top = np.argsort(-s)[:k]
    return round(float(y[top].sum() / y.sum()), 4)


def main():
    path = os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz")
    d = pd.read_csv(path)
    d["year0"] = d["cycle"].str[:4].astype(int)
    d = d[d["age"] >= 40]
    d = d[(d["followup_months"] >= 60) | (d["died"] == 1)].reset_index(drop=True)
    d["y"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= 60)).astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan)) if "SMQ020" in d else np.nan
    d["male"] = (d["gender"] == 1).astype(float)
    tr, te = d[d["year0"] <= 2005].reset_index(drop=True), d[d["year0"] >= 2007].reset_index(drop=True)
    print(f"{len(d):,} adults 40+;  fit {len(tr):,} ({int(tr['y'].sum())} cancer deaths, 1999-2006);  "
          f"test {len(te):,} ({int(te['y'].sum())} cancer deaths, 2007-2014)\n")

    allcols = [c for c in d.columns if c not in META | {"y", "year0", "smoked", "male", "gender"} and d[c].dtype != object]
    labs = [c for c in allcols if c in ROUTINE or c.startswith("LBXS")]
    sets = {
        "S0 age+sex": ["age", "male"],
        "S1 +smoking, BMI": ["age", "male", "smoked", "BMXBMI"],
        "S2 +22 routine labs": ["age", "male", "smoked", "BMXBMI"] + labs,
        "S3 whole picture": ["age", "male"] + [c for c in allcols if c != "age"] + ["smoked"],
    }
    # Post hoc, labelled as such in the paper: the whole picture without the missing-value
    # indicators, which can encode which survey cycle someone was in.
    sets["S3b whole picture, no missing flags"] = [c for c in sets["S3 whole picture"] if not c.endswith("__missing")]
    sets = {k: list(dict.fromkeys(v)) for k, v in sets.items()}
    for k, v in sets.items():
        print(f"   {k:<22} {len(v):>4} variables")
    print()

    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "cancer_train": int(ytr.sum()), "cancer_test": int(yte.sum()),
           "variables": {k: len(v) for k, v in sets.items()}, "models": {}}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores, chosen = {}, {}
    print("   cross-validated AUC within 1999-2006, then the held-out 2007-2014 AUC")
    for name, cols in sets.items():
        cands = {"ridge": logreg(1.0 if len(cols) < 10 else 0.05)}
        if len(cols) > 4:
            cands["boosting"] = hgb()
        best, best_cv = None, -1
        for mname, model in cands.items():
            p = cross_val_predict(model, tr[cols], ytr, cv=cv, method="predict_proba")[:, 1]
            a = roc_auc_score(ytr, p)
            if a > best_cv:
                best, best_cv, best_name = model, a, mname
        best.fit(tr[cols], ytr)
        s = best.predict_proba(te[cols])[:, 1]
        scores[name] = s
        chosen[name] = best
        res["models"][name] = {"chosen": best_name, "cv_auc_train_era": round(float(best_cv), 4),
                               "test_auc": round(float(roc_auc_score(yte, s)), 4)}
        print(f"   {name:<22} {best_name:<9} CV {best_cv:.3f}   held-out {roc_auc_score(yte, s):.3f}")

    # R1
    s1, s3 = scores["S1 +smoking, BMI"], scores["S3 whole picture"]
    gain = roc_auc_score(yte, s3) - roc_auc_score(yte, s1)
    gci = boot_gain(yte, s1, s3)
    res["R1"] = {"auc_S1": round(float(roc_auc_score(yte, s1)), 4), "auc_S3": round(float(roc_auc_score(yte, s3)), 4),
                 "gain": round(float(gain), 4), "gain_ci": gci, "bar": "gain >= 0.02 and CI above 0",
                 "passes": bool(gain >= 0.02 and gci[0] > 0)}
    steps = {}
    for a, b in (("S0 age+sex", "S1 +smoking, BMI"), ("S1 +smoking, BMI", "S2 +22 routine labs"), ("S2 +22 routine labs", "S3 whole picture")):
        steps[f"{a} -> {b}"] = {"gain": round(float(roc_auc_score(yte, scores[b]) - roc_auc_score(yte, scores[a])), 4),
                                "ci": boot_gain(yte, scores[a], scores[b], 500)}
    res["R1"]["steps"] = steps
    print(f"\nR1  S3 vs strong baseline S1: AUC {res['R1']['auc_S3']:.3f} vs {res['R1']['auc_S1']:.3f}, gain {gain:+.4f} {gci}  ->  "
          f"{'PASSES' if res['R1']['passes'] else 'FAILS'}")
    for k, v in steps.items():
        print(f"      {k}: {v['gain']:+.4f} {v['ci']}")

    s3b = scores["S3b whole picture, no missing flags"]
    gb = roc_auc_score(yte, s3b) - roc_auc_score(yte, s1)
    gbci = boot_gain(yte, s1, s3b)
    sl3b, _ = slope(yte, s3b)
    res["R1b_post_hoc"] = {"auc_S3b": round(float(roc_auc_score(yte, s3b)), 4), "gain_vs_S1": round(float(gb), 4), "gain_ci": gbci, "slope": sl3b,
                           "note": "Post hoc. Not pre-registered. Removes missing-value indicators, which can encode survey cycle."}
    print(f"R1b post hoc, no missing-flags: AUC {roc_auc_score(yte, s3b):.3f}, gain over S1 {gb:+.4f} {gbci}, slope {sl3b}")

    # R2: among test-era people who died within five years, cancer versus other
    dm = ((te["died"] == 1) & (te["followup_months"] <= 60)).to_numpy()
    yd = te["y"].to_numpy()[dm]
    a1, a3 = roc_auc_score(yd, s1[dm]), roc_auc_score(yd, s3[dm])
    r2ci = boot_gain(yd, s1[dm], s3[dm])
    res["R2"] = {"decedents": int(dm.sum()), "cancer_deaths": int(yd.sum()), "auc_S1": round(float(a1), 4), "auc_S3": round(float(a3), 4),
                 "gain": round(float(a3 - a1), 4), "gain_ci": r2ci, "bar": "decedent AUC gain CI above 0", "passes": bool(r2ci[0] > 0)}
    print(f"R2  among {int(dm.sum())} test-era decedents ({int(yd.sum())} cancer): AUC S1 {a1:.3f}, S3 {a3:.3f}, gain {a3 - a1:+.4f} {r2ci}  ->  "
          f"{'PASSES' if res['R2']['passes'] else 'FAILS'}")

    # R3
    sl, ic = slope(yte, s3)
    res["R3"] = {"slope": sl, "intercept": ic, "bar": "slope 0.8 to 1.2", "passes": bool(0.8 <= sl <= 1.2)}
    sl1, _ = slope(yte, s1)
    res["R3"]["slope_S1"] = sl1
    print(f"R3  calibration slope of S3 {sl} (S1 {sl1}), intercept {ic}  ->  {'PASSES' if res['R3']['passes'] else 'FAILS'}")

    # R4 learning curve
    lc = {}
    for frac in (0.25, 0.5, 1.0):
        row = {"S1": [], "S3": []}
        for rep in range(3 if frac < 1 else 1):
            idx = RNG.choice(len(tr), int(frac * len(tr)), replace=False) if frac < 1 else np.arange(len(tr))
            sub = tr.iloc[idx]
            m1 = logreg(1.0).fit(sub[sets["S1 +smoking, BMI"]], sub["y"])
            row["S1"].append(roc_auc_score(yte, m1.predict_proba(te[sets["S1 +smoking, BMI"]])[:, 1]))
            m3 = (hgb() if res["models"]["S3 whole picture"]["chosen"] == "boosting" else logreg(0.05)).fit(sub[sets["S3 whole picture"]], sub["y"])
            row["S3"].append(roc_auc_score(yte, m3.predict_proba(te[sets["S3 whole picture"]])[:, 1]))
        lc[str(frac)] = {k: round(float(np.mean(v)), 4) for k, v in row.items()} | {"n_train": int(frac * len(tr))}
        print(f"R4  fit on {int(frac * 100):>3}% ({lc[str(frac)]['n_train']:,} people): S1 {lc[str(frac)]['S1']:.3f}  S3 {lc[str(frac)]['S3']:.3f}")
    res["R4"] = lc

    # R5 triage
    cap = {}
    for nm, s in (("age+sex", scores["S0 age+sex"]), ("S1", s1), ("S2", scores["S2 +22 routine labs"]), ("S3", s3)):
        cap[nm] = {"top10": capture(yte, s, 0.10), "top20": capture(yte, s, 0.20), "top30": capture(yte, s, 0.30)}
    res["R5"] = cap
    print("R5  share of five-year cancer deaths found in the highest-scored 10% / 20% / 30% of people")
    for k, v in cap.items():
        print(f"      {k:<8} {v['top10']:.1%} / {v['top20']:.1%} / {v['top30']:.1%}")

    # which variables matter
    print("\n    what the whole-picture model uses (permutation importance, held-out era)")
    samp = te.sample(min(4000, len(te)), random_state=0)
    cols = sets["S3 whole picture"]
    pi = permutation_importance(chosen["S3 whole picture"], samp[cols], samp["y"], scoring="roc_auc", n_repeats=2, random_state=0, n_jobs=1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda kv: -kv[1])[:20]
    res["top_variables"] = [{"variable": c, "auc_drop": round(float(v), 5)} for c, v in imp]
    for c, v in imp[:15]:
        print(f"      {c:<14} {v:+.4f}")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
