"""
More data for the red-flag-labs study: tests T1-T3 of Addendum 5 in docs/RED_FLAG_LABS_PREREG.md (committed first).

T1  how well a routine blood count alone (anaemia, or anaemia with MCV < 80) finds guideline iron-deficiency anaemia, and
    how many ferritin tests a lab would add per case found (NHANES 2001-2002 and 2017-2020, survey-weighted);
T2  five-year cancer death versus other-cause death for microcytic anaemia, non-microcytic anaemia, unintentional weight
    loss and high platelets (NHANES 1999-2014 plus NHANES III, multinomial competing-risks model);
T3  timing: is the microcytic-anaemia cancer signal stronger in the first two years?

Run:  python experiments/red_flag_more.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from red_flag_labs import flags, ratio_est

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "red_flag_more_result.json")
RNG = np.random.default_rng(20261127)
B = 300


# ------------------------------------------------------------------ T1
def t1_one(path, hgb, fer, mcv, wt, women_periods, label):
    d = pd.read_csv(os.path.join(ROOT, "data", path))
    d = d[(d["RIDAGEYR"] >= 50) & d[hgb].notna() & d[fer].notna() & d[mcv].notna() & (d[wt] > 0)].copy()
    d["male"] = (d["RIAGENDR"] == 1).astype(int)
    d = d[~((d["male"] == 0) & (d[women_periods] == 1))]
    anaemia, ida, _ = flags(d[hgb].to_numpy(), d[fer].to_numpy(), d["male"].to_numpy())
    micro = anaemia & (d[mcv].to_numpy() < 80)
    w, h, c = d[wt].to_numpy(), d["SDMVSTRA"].to_numpy(), d["SDMVPSU"].to_numpy()
    out = {"dataset": label, "n": int(len(d)), "ida_n": int(ida.sum()), "anaemia_n": int(anaemia.sum()), "microcytic_anaemia_n": int(micro.sum())}
    for name, rule in (("any_anaemia", anaemia), ("microcytic_anaemia", micro)):
        sens, sci = ratio_est((rule & ida).astype(float), ida.astype(float), w, h, c)
        ppv, pci = ratio_est((rule & ida).astype(float), rule.astype(float), w, h, c)
        out[name] = {"sensitivity": sens, "sensitivity_ci": sci, "ppv": ppv, "ppv_ci": pci, "ferritin_tests_per_case": round(1 / ppv, 1) if ppv > 0 else None}
    return out


def t1():
    a = t1_one("red_flag_current.csv.gz", "LBXHGB", "LBXFER", "LBXMCVSI", "WTMECPRP", "RHQ031", "NHANES 2017-2020")
    b = t1_one("red_flag_2001.csv.gz", "LBXHGB", "LBXFER", "LBXMCVSI", "WTMEC2YR", "RHQ030", "NHANES 2001-2002")
    bar = all(x["any_anaemia"]["sensitivity_ci"][0] > 0.8 and x["any_anaemia"]["sensitivity"] >= 0.9 and x["any_anaemia"]["ferritin_tests_per_case"] <= 10 for x in (a, b))
    return {"2017-2020": a, "2001-2002": b, "bar_T1_met": bool(bar)}


# ------------------------------------------------------------------ T2, T3
def cohort():
    nav = pd.read_csv(os.path.join(ROOT, "data", "nhanes_navigator.csv.gz"))
    nav = nav[nav["cycle"].isin(["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008", "2009-2010", "2011-2012", "2013-2014"])]
    a = pd.DataFrame({"cohort": "NHANES 1999-2014", "age": nav["age"], "male": (nav["sex"] == "male").astype(float), "smoked": nav["ever_smoked"],
                      "hemoglobin": nav["hemoglobin"], "mcv": nav["mcv"], "platelets": nav["platelets"], "weight_loss": nav["weight_loss"],
                      "died": nav["died"], "ucod": nav["ucod"], "months": pd.to_numeric(nav["months"], errors="coerce"), "prior_cancer": 0})
    f = pd.read_csv(os.path.join(ROOT, "data", "red_flag_followup.csv.gz"), low_memory=False)
    f = f[f["dataset"] == "NHANES III 1988-1994"]
    b = pd.DataFrame({"cohort": "NHANES III", "age": f["age"], "male": f["male"], "smoked": f["smoked"], "hemoglobin": f["hemoglobin"], "mcv": f["mcv"],
                      "platelets": f["platelets"], "weight_loss": np.nan, "died": f["died"], "ucod": f["ucod"], "months": pd.to_numeric(f["months"], errors="coerce"),
                      "prior_cancer": f["prior_cancer"]})
    d = pd.concat([a, b], ignore_index=True)
    d = d[(d["age"] >= 50) & (d["age"] <= 84) & (d["prior_cancer"] == 0) & d["hemoglobin"].notna() & d["mcv"].notna() & d["platelets"].notna()
          & d["smoked"].notna() & d["died"].notna() & d["months"].notna()]
    d = d[(d["months"] >= 60) | (d["died"] == 1)].reset_index(drop=True)
    anaemia = np.where(d["male"] == 1, d["hemoglobin"] < 13, d["hemoglobin"] < 12)
    d["micro_anaemia"] = (anaemia & (d["mcv"] < 80)).astype(float)
    d["other_anaemia"] = (anaemia & (d["mcv"] >= 80)).astype(float)
    d["high_platelets"] = (d["platelets"] > 400).astype(float)
    d["y"] = np.where((d["died"] == 1) & (d["months"] <= 60) & (d["ucod"] == 2), 1, np.where((d["died"] == 1) & (d["months"] <= 60), 2, 0))
    return d


def mfit(s, cols, y):
    st = SplineTransformer(degree=3, n_knots=4, knots="quantile", extrapolation="constant", include_bias=False).fit(s[["age"]])
    X = np.column_stack([st.transform(s[["age"]].to_numpy()), s[["male", "smoked"]].to_numpy(), (s["cohort"] == "NHANES III").to_numpy().astype(float)[:, None], s[cols].to_numpy()])
    m = LogisticRegression(C=1e6, max_iter=4000).fit(X, y)
    cls = list(m.classes_)
    k = X.shape[1] - len(cols)
    return {c: (float(m.coef_[cls.index(1)][k + j] - m.coef_[cls.index(0)][k + j]), float(m.coef_[cls.index(2)][k + j] - m.coef_[cls.index(0)][k + j])) for j, c in enumerate(cols)}


def summarise(point, boots, n_exposed, deaths_exposed):
    e = lambda x: round(float(np.exp(x)), 3)  # noqa: E731
    ci = lambda a: [round(float(v), 3) for v in np.exp(np.percentile(a, [2.5, 97.5]))]  # noqa: E731
    bc, bo = np.array([b[0] for b in boots]), np.array([b[1] for b in boots])
    return {"n_exposed": n_exposed, "cancer_deaths_exposed": deaths_exposed, "rrr_cancer": e(point[0]), "rrr_cancer_ci": ci(bc), "rrr_other": e(point[1]), "rrr_other_ci": ci(bo),
            "ratio": e(point[0] - point[1]), "ratio_ci": ci(bc - bo)}


def t2(d):
    res = {"n": int(len(d)), "cancer_deaths": int((d["y"] == 1).sum()), "other_deaths": int((d["y"] == 2).sum()), "by_cohort": d.groupby("cohort").size().to_dict()}
    main_cols = ["micro_anaemia", "other_anaemia", "high_platelets"]
    pt = mfit(d, main_cols, d["y"].to_numpy())
    bs = []
    for _ in range(B):
        i = RNG.integers(0, len(d), len(d))
        s = d.iloc[i].reset_index(drop=True)
        bs.append(mfit(s, main_cols, s["y"].to_numpy()))
    for c in main_cols:
        m = d[c] == 1
        res[c] = summarise(pt[c], [b[c] for b in bs], int(m.sum()), int((d.loc[m, "y"] == 1).sum()))
    # weight loss: 1999-2014 only
    w = d[(d["cohort"] == "NHANES 1999-2014") & d["weight_loss"].notna()].reset_index(drop=True)
    cols = ["micro_anaemia", "other_anaemia", "high_platelets", "weight_loss"]
    ptw = mfit(w, cols, w["y"].to_numpy())
    bw = []
    for _ in range(B):
        i = RNG.integers(0, len(w), len(w))
        s = w.iloc[i].reset_index(drop=True)
        bw.append(mfit(s, cols, s["y"].to_numpy()))
    m = w["weight_loss"] == 1
    res["weight_loss_1999_2014"] = summarise(ptw["weight_loss"], [b["weight_loss"] for b in bw], int(m.sum()), int((w.loc[m, "y"] == 1).sum()))
    mic = res["micro_anaemia"]
    res["bar_T2_met"] = bool(mic["rrr_cancer_ci"][0] > 1 and mic["ratio_ci"][0] > 1)
    return res


def t3(d):
    def lor(s, early):
        if early:
            y = ((s["died"] == 1) & (s["ucod"] == 2) & (s["months"] <= 24)).astype(int).to_numpy()
            ss = s
        else:
            ss = s[(s["months"] > 24)].reset_index(drop=True)
            y = ((ss["died"] == 1) & (ss["ucod"] == 2) & (ss["months"] <= 60)).astype(int).to_numpy()
        st = SplineTransformer(degree=3, n_knots=4, knots="quantile", extrapolation="constant", include_bias=False).fit(ss[["age"]])
        X = np.column_stack([st.transform(ss[["age"]].to_numpy()), ss[["male", "smoked", "micro_anaemia", "other_anaemia"]].to_numpy(),
                             (ss["cohort"] == "NHANES III").to_numpy().astype(float)[:, None]])
        m = LogisticRegression(C=1e6, solver="newton-cholesky", max_iter=300).fit(X, y)
        return float(m.coef_[0][X.shape[1] - 3]), int(y[ss["micro_anaemia"].to_numpy() == 1].sum())

    e0, n0 = lor(d, True)
    l0, n1 = lor(d, False)
    bs = []
    for _ in range(B):
        i = RNG.integers(0, len(d), len(d))
        s = d.iloc[i].reset_index(drop=True)
        bs.append((lor(s, True)[0], lor(s, False)[0]))
    bt = np.array(bs)
    ci = lambda a: [round(float(v), 3) for v in np.exp(np.percentile(a, [2.5, 97.5]))]  # noqa: E731
    r = {"or_months_0_24": round(float(np.exp(e0)), 3), "or_0_24_ci": ci(bt[:, 0]), "cancer_deaths_exposed_0_24": n0,
         "or_months_25_60": round(float(np.exp(l0)), 3), "or_25_60_ci": ci(bt[:, 1]), "cancer_deaths_exposed_25_60": n1,
         "ratio_early_to_late": round(float(np.exp(e0 - l0)), 3), "ratio_ci": ci(bt[:, 0] - bt[:, 1])}
    r["bar_T3_met"] = bool(r["ratio_ci"][0] > 1)
    return r


def main():
    out = {"T1": t1()}
    print("T1", json.dumps(out["T1"], indent=1)[:2500], flush=True)
    d = cohort()
    out["T2"] = t2(d)
    print("T2", json.dumps(out["T2"], indent=1), flush=True)
    out["T3"] = t3(d)
    print("T3", json.dumps(out["T3"], indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
