"""
Descriptive statistics and sensitivity analyses for the navigator evidence study.

Run after experiments/navigator_evidence.py (it reads the alert cache that run writes).

    1  who is in the cohort, by era
    2  the alert treated as a diagnostic test for five-year cancer death
         (sensitivity, specificity, predictive values, likelihood ratio)
    3  absolute five-year cancer death by age band, alerted versus not
    4  does the cancer-specificity result survive other choices?
         by sex, by age group, with the first year of follow-up removed,
         a five-year window instead of ten, and only people with measured ferritin

Nothing here has a pass/fail bar. It exists so a reader can see whether the headline
findings depend on a particular choice.

Run:  python experiments/navigator_sensitivity.py
"""

import json
import os

import numpy as np
import pandas as pd

from navigator_evidence import ALERT, CANCER, design, irls, or_ci, wilson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "navigator_sensitivity_result.json")


def load():
    c = pd.read_csv(os.path.join(ROOT, "data", "nhanes_navigator.csv.gz"))
    c = c[c["age"] >= 40].dropna(subset=["hemoglobin", "platelets"]).reset_index(drop=True)
    a = pd.read_csv(os.path.join(ROOT, "data", "navigator_alerts.csv.gz"))
    assert len(a) == len(c), "run experiments/navigator_evidence.py first"
    c["lab_state"], c["ctx_state"] = a["lab_state"], a["ctx_state"]
    c["year0"] = c["cycle"].str[:4].astype(int)
    c["any_lab"] = c["lab_state"].isin(ALERT)
    c["any_ctx"] = c["ctx_state"].isin(ALERT)
    c["soon_lab"] = c["lab_state"].eq("talk_soon")
    return c


def five_year(d):
    d = d[(d["months"] >= 60) | (d["died"] == 1)].copy()
    d["y"] = ((d["died"] == 1) & (d["ucod"] == CANCER) & (d["months"] <= 60)).astype(int)
    return d


def accuracy(y, flag):
    tp, fp = int(((y == 1) & flag).sum()), int(((y == 0) & flag).sum())
    fn, tn = int(((y == 1) & ~flag).sum()), int(((y == 0) & ~flag).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv, npv = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(sens, 4), "sensitivity_ci": wilson(tp, tp + fn),
            "specificity": round(spec, 4), "specificity_ci": wilson(tn, tn + fp),
            "ppv": round(ppv, 4), "ppv_ci": wilson(tp, tp + fp), "npv": round(npv, 5),
            "lr_positive": round(sens / (1 - spec), 2) if spec < 1 else None,
            "flagged_share": round((tp + fp) / len(y), 4)}


def main():
    c = load()
    n_all = len(c)
    res = {}

    # 1 who is in the cohort
    rows = {}
    for era, m in (("1999-2006", c["year0"] <= 2005), ("2007-2014", (c["year0"] >= 2007) & (c["year0"] <= 2013)),
                   ("2015-2018", c["year0"] >= 2015)):
        d = c[m]
        died = d["died"] == 1
        rows[era] = {
            "n": int(len(d)), "mean_age": round(float(d["age"].mean()), 1), "female_pct": round(float((d["sex"] == "female").mean()), 3),
            "ever_smoked_pct": round(float(d["ever_smoked"].mean()), 3), "hemoglobin_mean": round(float(d["hemoglobin"].mean()), 2),
            "mcv_mean": round(float(d["mcv"].mean()), 1), "platelets_mean": round(float(d["platelets"].mean()), 0),
            "wbc_mean": round(float(d["wbc"].mean()), 2), "weight_loss_proxy_pct": round(float((d["weight_loss"] == 1).mean()), 3),
            "died_pct": round(float(died.mean()), 3), "cancer_deaths": int((died & (d["ucod"] == CANCER)).sum()),
            "any_lab_alert_pct": round(float(d["any_lab"].mean()), 3)}
    res["cohort_by_era"] = rows
    print("1  cohort by era")
    for k, v in rows.items():
        print(f"   {k}: n={v['n']:,} mean age {v['mean_age']} female {v['female_pct']:.0%} smoked {v['ever_smoked_pct']:.0%} cancer deaths {v['cancer_deaths']}")

    # 2 the alert as a diagnostic test for five-year cancer death
    f = five_year(c[c["year0"] <= 2013])
    acc = {}
    for label, col in (("any alert, labs only", "any_lab"), ("see a doctor soon, labs only", "soon_lab"),
                       ("any alert, labs + proxy symptoms", "any_ctx")):
        acc[label] = accuracy(f["y"].to_numpy(), f[col].to_numpy())
    # same flagged share, chosen by age alone, for a fair comparison
    for label, col in (("any alert, labs only", "any_lab"), ("any alert, labs + proxy symptoms", "any_ctx")):
        k = int(f[col].sum())
        oldest = f.sort_values("age", ascending=False).head(k)
        flag = f.index.isin(oldest.index)
        acc[f"{label} (versus oldest {k:,} by age alone)"] = accuracy(f["y"].to_numpy(), flag)
    res["alert_as_test"] = acc
    print("\n2  the alert as a test for five-year cancer death (1999-2013 examinations)")
    for k, v in acc.items():
        print(f"   {k:<62} flagged {v['flagged_share']:.1%} sens {v['sensitivity']:.1%} spec {v['specificity']:.1%} PPV {v['ppv']:.1%} LR+ {v['lr_positive']}")

    # 3 absolute risk by age band
    bands = [(40, 49), (50, 59), (60, 69), (70, 79), (80, 120)]
    ab = {}
    for lo, hi in bands:
        d = f[(f["age"] >= lo) & (f["age"] <= hi)]
        row = {"n": int(len(d))}
        for lab, col in (("alerted", "any_lab"), ("not_alerted", None)):
            m = d[col] if col else ~d["any_lab"]
            k, n = int(d.loc[m, "y"].sum()), int(m.sum())
            row[lab] = {"n": n, "cancer_deaths": k, "rate": round(k / max(n, 1), 4), "ci": wilson(k, n)}
        a, b = row["alerted"], row["not_alerted"]
        row["rate_ratio"] = round(a["rate"] / b["rate"], 2) if b["rate"] > 0 and a["n"] else None
        ab[f"{lo}-{hi if hi < 120 else '+'}"] = row
    res["absolute_risk_by_age"] = ab
    print("\n3  five-year cancer death, labs-only alert versus none, by age band")
    for k, v in ab.items():
        print(f"   {k:<6} alerted {v['alerted']['rate']:.2%} (n={v['alerted']['n']}) vs not {v['not_alerted']['rate']:.2%} (n={v['not_alerted']['n']})  RR {v['rate_ratio']}")

    # 4 cancer specificity sensitivity analyses
    dec = c[(c["year0"] <= 2013) & (c["died"] == 1)].copy()
    dec["cancer"] = (dec["ucod"] == CANCER).astype(int)
    dec["flag"] = dec["any_lab"].astype(float)
    dec["yr"] = dec["year0"] - 2006

    def fit(d, window=120, lag=0, label=""):
        d = d[(d["months"] <= window) & (d["months"] > lag)]
        if d["flag"].sum() < 20 or d["cancer"].sum() < 20:
            return None
        r = or_ci(design(d, ["flag", "yr"]), d["cancer"].to_numpy(), 4)
        r.update({"decedents": int(len(d)), "alerted": int(d["flag"].sum())})
        print(f"   {label:<42} decedents {r['decedents']:>5}  alerted {r['alerted']:>4}  OR {r['or']} {r['ci']}")
        return r

    sens = {}
    print("\n4  cancer specificity, labs-only alert, decedents: does the answer depend on the choices?")
    sens["all, 10-year window"] = fit(dec, label="all, 10-year window")
    sens["5-year window"] = fit(dec, window=60, label="5-year window")
    sens["first 12 months removed"] = fit(dec, lag=12, label="first 12 months removed")
    sens["first 24 months removed"] = fit(dec, lag=24, label="first 24 months removed")
    for sex in ("female", "male"):
        sens[f"{sex}s only"] = fit(dec[dec["sex"] == sex], label=f"{sex}s only")
    for lo, hi, nm in ((40, 59, "aged 40-59"), (60, 74, "aged 60-74"), (75, 120, "aged 75 and over")):
        sens[nm] = fit(dec[(dec["age"] >= lo) & (dec["age"] <= hi)], label=nm)
    sens["ferritin measured"] = fit(dec[dec["ferritin"].notna()], label="ferritin measured")
    sens["never smoked"] = fit(dec[dec["ever_smoked"] == 0], label="never smoked")
    sens["ever smoked"] = fit(dec[dec["ever_smoked"] == 1], label="ever smoked")
    res["specificity_sensitivity"] = sens

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
