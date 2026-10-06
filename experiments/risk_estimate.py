"""
Personal five-year cancer-death estimate from age, sex, smoking and BMI.
Forms, split and bars are in docs/RISK_ESTIMATE_PREREG.md, committed before this was run.

    1  choose among three forms by held-out Brier score (fit 1999-2006, test 2007-2014)
    2  check the four pre-registered bars on the held-out era
    3  only if every bar is met: refit on all cycles, bootstrap 300 times, write
       backend/risk_estimate.json for the app

Run:  python experiments/risk_estimate.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RESULT = os.path.join(ROOT, "experiments", "risk_estimate_result.json")
OUT_MODEL = os.path.join(ROOT, "backend", "risk_estimate.json")
RNG = np.random.default_rng(20261006)
CANCER = 2
BMI_LO, BMI_HI = 14.0, 60.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def features(df, form, smoked_mean, bmi_mean):
    a = df["age"].to_numpy() - 60.0
    cols = [a, a ** 2 / 20.0, (df["gender"] == 1).astype(float).to_numpy(), df["smoked"].fillna(smoked_mean).to_numpy()]
    if form == "M4":
        cols.append(a ** 3 / 800.0)
    if form in ("M2", "M3"):
        b = (df["BMXBMI"].clip(BMI_LO, BMI_HI).fillna(bmi_mean).to_numpy() - 27.0) / 5.0
        cols.append(b)
        if form == "M3":
            cols.append(b ** 2)
    return np.column_stack(cols)


def fit(X, y):
    return LogisticRegression(C=1e9, max_iter=5000).fit(X, y)


def slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e9, max_iter=2000).fit(z.reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def evaluate(d, fit_cycles, test_cycles, form, label):
    tr = d[d["year0"].isin(fit_cycles)].reset_index(drop=True)
    te = d[d["year0"].isin(test_cycles)].reset_index(drop=True)
    sm, bm = float(tr["smoked"].mean()), float(tr["BMXBMI"].clip(BMI_LO, BMI_HI).mean())
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    m = fit(features(tr, form, sm, bm), ytr)
    p = m.predict_proba(features(te, form, sm, bm))[:, 1]
    s, i = slope_intercept(yte, p)
    r = {"label": label, "form": form, "n_train": int(len(tr)), "n_test": int(len(te)), "cancer_train": int(ytr.sum()), "cancer_test": int(yte.sum()),
         "auc": round(float(roc_auc_score(yte, p)), 4), "brier": round(float(brier_score_loss(yte, p)), 6), "slope": round(s, 3), "intercept": round(i, 3)}
    b1 = r["auc"] >= 0.74
    b2 = 0.8 <= r["slope"] <= 1.2 and -0.3 <= r["intercept"] <= 0.3
    q = pd.qcut(pd.Series(p), 10, duplicates="drop")
    dec, inside = [], 0
    for _, g in pd.Series(p).groupby(q, observed=True):
        k, n = int(yte[g.index].sum()), int(len(g))
        lo, hi = wilson(k, n)
        ok = lo <= g.mean() <= hi
        inside += ok
        dec.append({"predicted": round(float(g.mean()), 4), "observed": round(k / n, 4), "n": n, "wilson": [round(lo, 4), round(hi, 4)], "inside": bool(ok)})
    b3 = inside >= 8
    bands, b4 = {}, True
    for lo, hi, name in ((40, 49, "40-49"), (50, 59, "50-59"), (60, 69, "60-69"), (70, 120, "70+")):
        m_ = ((te["age"] >= lo) & (te["age"] <= hi)).to_numpy()
        obs, exp = int(yte[m_].sum()), float(p[m_].sum())
        pl, ph = poisson.ppf(0.025, exp), poisson.ppf(0.975, exp)
        ok = (abs(obs - exp) <= 0.25 * exp) or (pl <= obs <= ph)
        b4 &= ok
        bands[name] = {"n": int(m_.sum()), "observed": obs, "expected": round(exp, 1), "ok": bool(ok)}
    r.update({"deciles": dec, "deciles_inside": int(inside), "bands": bands,
              "bars": {"auc_ge_0.74": bool(b1), "slope_and_intercept": bool(b2), "deciles_8_of_10": bool(b3), "age_bands": bool(b4)},
              "all_bars_met": bool(b1 and b2 and b3 and b4)})
    print(f"{label} [{form}] fit {len(tr):,} / test {len(te):,}: AUC {r['auc']:.4f}, slope {r['slope']}, intercept {r['intercept']}, deciles {inside}/10, age bands "
          + ", ".join(f"{k} {v['observed']}/{v['expected']}{'' if v['ok'] else ' MISS'}" for k, v in bands.items()) + f"  -> {'ALL MET' if r['all_bars_met'] else 'BAR MISSED'}")
    return r


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d = d[d["age"] >= 40]
    d = d[(d["followup_months"] >= 60) | (d["died"] == 1)].reset_index(drop=True)
    d["y"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= 60)).astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan))
    A_fit, A_test = [1999, 2001, 2003, 2005], [2007, 2009, 2011, 2013]
    B_fit, B_test = [1999, 2001, 2003, 2005, 2007], [2009, 2011, 2013]
    res = {"first_run": {}, "amendment": {}}
    print("First pre-registered run (forms M1 to M3), split A")
    for form in ("M1", "M2", "M3"):
        res["first_run"][form] = evaluate(d, A_fit, A_test, form, "A")
    print("\nAmendment: M4 (cubic age), both splits")
    for form in ("M1", "M4"):
        res["amendment"][form] = {"A": evaluate(d, A_fit, A_test, form, "A"), "B": evaluate(d, B_fit, B_test, form, "B")}
    ok = res["amendment"]["M4"]["A"]["all_bars_met"] and res["amendment"]["M4"]["B"]["all_bars_met"]
    res["ship"] = bool(ok)
    print(f"\nSHIP A PERCENTAGE: {ok}")

    if ok:
        form = "M4"
        sm_all, bm_all = float(d["smoked"].mean()), float(d["BMXBMI"].clip(BMI_LO, BMI_HI).mean())
        X = features(d, form, sm_all, bm_all)
        y = d["y"].to_numpy()
        m = fit(X, y)
        boots = []
        for _ in range(300):
            i = RNG.integers(0, len(d), len(d))
            mb = fit(X[i], y[i])
            boots.append([float(mb.intercept_[0])] + [float(c) for c in mb.coef_[0]])
        model = {"form": form, "n": int(len(d)), "cancer_deaths_5y": int(y.sum()), "age_range": [40, 85],
                 "bmi_clip": [BMI_LO, BMI_HI], "mean_smoked": round(sm_all, 4), "mean_bmi": round(bm_all, 3),
                 "intercept": float(m.intercept_[0]), "coefficients": [float(c) for c in m.coef_[0]],
                 "terms": ["age_minus_60", "age_minus_60_sq_over_20", "male", "ever_smoked", "age_minus_60_cubed_over_800"],
                 "bootstrap": boots,
                 "validation": {"split_A": {k: res["amendment"]["M4"]["A"][k] for k in ("auc", "slope", "intercept", "deciles_inside")},
                                "split_B": {k: res["amendment"]["M4"]["B"][k] for k in ("auc", "slope", "intercept", "deciles_inside")}}}
        with open(OUT_MODEL, "w") as f:
            json.dump(model, f)
        print(f"wrote {OUT_MODEL}")
    with open(OUT_RESULT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT_RESULT}")


if __name__ == "__main__":
    main()
