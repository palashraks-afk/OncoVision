"""
Hidden red flags on ordinary lab reports. Rules: docs/RED_FLAG_LABS_PREREG.md (committed before any result).

Part A, societal (NHANES 2017-March 2020, survey-weighted, design-based intervals): how many US adults 50+ have
iron-deficiency anaemia (AGA 2020: anaemia and ferritin < 45 ng/mL), how many are not being treated for anaemia, and
whether the untreated have a usual place of care.

Part B, scientific (NHANES 2001-2002 and NHANES III, deaths through 2019): does IDA predict cancer death, more than it
predicts other-cause death, and more than non-iron anaemia does?

Run:  python experiments/red_flag_labs.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "red_flag_labs_result.json")
RNG = np.random.default_rng(20261126)
B = 500


# ------------------------------------------------------------------ design-based estimation
def ratio_est(y, x, w, strata, psu):
    """Weighted ratio sum(w*y)/sum(w*x) with Taylor-linearised standard error (with-replacement PSUs within strata)."""
    sx = (w * x).sum()
    r = (w * y).sum() / sx
    z = w * (y - r * x) / sx
    df = pd.DataFrame({"z": z, "h": strata, "c": psu}).groupby(["h", "c"])["z"].sum().reset_index()
    var = 0.0
    for _, g in df.groupby("h"):
        n = len(g)
        if n > 1:
            var += n / (n - 1) * ((g["z"] - g["z"].mean()) ** 2).sum()
    se = float(np.sqrt(var))
    return float(r), [float(max(r - 1.96 * se, 0)), float(min(r + 1.96 * se, 1))]


def flags(hgb, fer, male):
    anaemia = np.where(male == 1, hgb < 13, hgb < 12)
    ida = anaemia & (fer < 45)
    nia = anaemia & (fer >= 45)
    return anaemia, ida, nia


def part_a():
    d = pd.read_csv(os.path.join(ROOT, "data", "red_flag_current.csv.gz"))
    d = d[(d["RIDAGEYR"] >= 50) & d["LBXHGB"].notna() & d["LBXFER"].notna() & (d["WTMECPRP"] > 0)].copy()
    d["male"] = (d["RIAGENDR"] == 1).astype(int)
    d = d[~((d["male"] == 0) & (d["RHQ031"] == 1))]
    _, ida, nia = flags(d["LBXHGB"].to_numpy(), d["LBXFER"].to_numpy(), d["male"].to_numpy())
    d["ida"], d["nia"] = ida.astype(float), nia.astype(float)
    w, h, c = d["WTMECPRP"].to_numpy(), d["SDMVSTRA"].to_numpy(), d["SDMVPSU"].to_numpy()
    one = np.ones(len(d))
    pop = float(w.sum())
    res = {"n_50plus": int(len(d)), "weighted_population_50plus_millions": round(pop / 1e6, 1), "ida_n": int(d["ida"].sum())}
    p, ci = ratio_est(d["ida"].to_numpy(), one, w, h, c)
    res["ida_prevalence"] = {"est": p, "ci": ci, "millions": round(p * pop / 1e6, 2), "millions_ci": [round(ci[0] * pop / 1e6, 2), round(ci[1] * pop / 1e6, 2)]}
    for sex, m in (("men", d["male"] == 1), ("women", d["male"] == 0)):
        mm = m.to_numpy().astype(float)
        p2, ci2 = ratio_est(d["ida"].to_numpy() * mm, mm, w, h, c)
        res[f"ida_prevalence_{sex}"] = {"est": p2, "ci": ci2}
    p3, ci3 = ratio_est(d["nia"].to_numpy(), one, w, h, c)
    res["nonIron_anaemia_prevalence"] = {"est": p3, "ci": ci3}
    # sensitivity: without earlier cancer
    q = d[d["MCQ220"] != 1]
    p4, ci4 = ratio_est(q["ida"].to_numpy(), np.ones(len(q)), q["WTMECPRP"].to_numpy(), q["SDMVSTRA"].to_numpy(), q["SDMVPSU"].to_numpy())
    res["ida_prevalence_no_prior_cancer"] = {"est": p4, "ci": ci4}

    # among IDA: untreated, and the care they have
    known = d["MCQ053"].isin([1, 2]).to_numpy()
    idx = (d["ida"] == 1).to_numpy() & known
    untreated = (d["MCQ053"] == 2).to_numpy().astype(float)
    p5, ci5 = ratio_est(untreated * idx, idx.astype(float), w, h, c)
    res["ida_untreated_share"] = {"est": p5, "ci": ci5, "n": int(idx.sum()), "millions": round(p5 * res["ida_prevalence"]["millions"], 2)}
    u = idx & (untreated == 1)
    res["untreated_ida_n"] = int(u.sum())
    for name, val, valid in (("usual_place_of_care", d["HUQ030"].isin([1, 3]), d["HUQ030"].isin([1, 2, 3])),
                             ("two_plus_visits_past_year", d["HUQ051"] >= 2, d["HUQ051"].between(0, 8)),
                             ("insured", d["HIQ011"] == 1, d["HIQ011"].isin([1, 2])),
                             ("below_poverty", d["INDFMPIR"] < 1, d["INDFMPIR"].notna())):
        den = u & valid.to_numpy()
        p6, ci6 = ratio_est(val.to_numpy().astype(float) * den, den.astype(float), w, h, c)
        res[f"untreated_ida_{name}"] = {"est": p6, "ci": ci6, "n": int(den.sum())}
    race = {1: "Mexican American", 2: "Other Hispanic", 3: "non-Hispanic White", 4: "non-Hispanic Black", 6: "non-Hispanic Asian", 7: "other or multiracial"}
    res["ida_prevalence_by_group"] = {}
    for code, name in race.items():
        m = (d["RIDRETH3"] == code).to_numpy().astype(float)
        if m.sum() >= 100:
            p7, ci7 = ratio_est(d["ida"].to_numpy() * m, m, w, h, c)
            res["ida_prevalence_by_group"][name] = {"est": p7, "ci": ci7, "n": int(m.sum()), "ida_n": int((d["ida"].to_numpy() * m).sum())}
    res["bar_societal_met"] = bool(res["untreated_ida_usual_place_of_care"]["ci"][0] > 0.5)
    return res


# ------------------------------------------------------------------ follow-up
def load_followup():
    d = pd.read_csv(os.path.join(ROOT, "data", "red_flag_followup.csv.gz"), low_memory=False)
    d["months"] = pd.to_numeric(d["months"], errors="coerce")
    d = d[(d["age"] >= 50) & (d["age"] <= 84) & (d["prior_cancer"] == 0) & d["hemoglobin"].notna() & d["ferritin"].notna()
          & d["smoked"].notna() & d["died"].notna() & d["months"].notna()].copy()
    _, ida, nia = flags(d["hemoglobin"].to_numpy(), d["ferritin"].to_numpy(), d["male"].to_numpy())
    d["ida"], d["nia"] = ida.astype(float), nia.astype(float)
    for yrs in (5, 10):
        keep = (d["months"] >= 12 * yrs) | (d["died"] == 1)
        d[f"ok{yrs}"] = keep
        d[f"ca{yrs}"] = ((d["died"] == 1) & (d["ucod"] == 2) & (d["months"] <= 12 * yrs)).astype(int)
        d[f"oth{yrs}"] = ((d["died"] == 1) & (d["ucod"] != 2) & (d["months"] <= 12 * yrs)).astype(int)
    return d.reset_index(drop=True)


def design(d, st, pooled):
    cols = [st.transform(d[["age"]].to_numpy()), d[["male", "smoked", "ida", "nia"]].to_numpy()]
    if pooled:
        cols.append((d["dataset"] == "NHANES III 1988-1994").to_numpy().astype(float)[:, None])
    return np.column_stack(cols)


def logors(d, outcome, pooled):
    st = SplineTransformer(degree=3, n_knots=4, knots="quantile", extrapolation="constant", include_bias=False).fit(d[["age"]])
    m = LogisticRegression(C=1e6, solver="newton-cholesky", max_iter=200).fit(design(d, st, pooled), d[outcome].to_numpy())
    k = st.transform(d[["age"]].iloc[:1].to_numpy()).shape[1]
    return float(m.coef_[0][k + 2]), float(m.coef_[0][k + 3])      # IDA, non-iron anaemia


def part_b(yrs):
    d0 = load_followup()
    d = d0[d0[f"ok{yrs}"]].reset_index(drop=True)
    res = {}
    for name, sub, pooled in (("NHANES 2001-2002", d[d["dataset"] == "NHANES 2001-2002"], False),
                              ("NHANES III 1988-1994", d[d["dataset"] == "NHANES III 1988-1994"], False), ("pooled", d, True)):
        sub = sub.reset_index(drop=True)
        ca, oth = f"ca{yrs}", f"oth{yrs}"
        r = {"n": int(len(sub)), "ida_n": int(sub["ida"].sum()), "nia_n": int(sub["nia"].sum()),
             "cancer_deaths": int(sub[ca].sum()), "cancer_deaths_ida": int(sub.loc[sub["ida"] == 1, ca].sum()),
             "cancer_deaths_nia": int(sub.loc[sub["nia"] == 1, ca].sum()), "cancer_deaths_none": int(sub.loc[(sub["ida"] == 0) & (sub["nia"] == 0), ca].sum()),
             "other_deaths_ida": int(sub.loc[sub["ida"] == 1, oth].sum())}
        for g, lab in (("ida", "ida"), ("nia", "nia")):
            m = sub[g] == 1
            r[f"crude_cancer_rate_{lab}"] = round(float(sub.loc[m, ca].mean()), 4) if m.any() else None
        r["crude_cancer_rate_none"] = round(float(sub.loc[(sub["ida"] == 0) & (sub["nia"] == 0), ca].mean()), 4)
        r["people_with_ida_per_cancer_death"] = round(r["ida_n"] / r["cancer_deaths_ida"], 1) if r["cancer_deaths_ida"] else None
        r["people_without_anaemia_per_cancer_death"] = round(int(((sub["ida"] == 0) & (sub["nia"] == 0)).sum()) / max(r["cancer_deaths_none"], 1), 1)
        if r["cancer_deaths_ida"] < 3:
            r["note"] = "too few cancer deaths in people with IDA for a model"
            res[name] = r
            continue
        c_ida, c_nia = logors(sub, ca, pooled)
        o_ida, o_nia = logors(sub, oth, pooled)
        boots = []
        for _ in range(B):
            i = RNG.integers(0, len(sub), len(sub))
            s = sub.iloc[i].reset_index(drop=True)
            if s.loc[s["ida"] == 1, ca].sum() == 0 or s.loc[s["ida"] == 1, oth].sum() == 0:
                continue
            a, b = logors(s, ca, pooled)
            o, _ = logors(s, oth, pooled)
            boots.append((a, b, o))
        bt = np.array(boots)
        pc = lambda x: [round(float(v), 3) for v in np.exp(np.percentile(x, [2.5, 97.5]))]  # noqa: E731
        r.update({"or_cancer_ida": round(np.exp(c_ida), 3), "or_cancer_ida_ci": pc(bt[:, 0]),
                  "or_cancer_nia": round(np.exp(c_nia), 3), "or_cancer_nia_ci": pc(bt[:, 1]),
                  "or_other_ida": round(np.exp(o_ida), 3), "or_other_ida_ci": pc(bt[:, 2]),
                  "ratio_of_ors_cancer_vs_other_ida": round(np.exp(c_ida - o_ida), 3), "ratio_of_ors_ci": pc(bt[:, 0] - bt[:, 2]),
                  "ida_minus_nia_cancer_ratio_of_ors": round(np.exp(c_ida - c_nia), 3), "ida_minus_nia_ci": pc(bt[:, 0] - bt[:, 1]),
                  "bootstrap_used": int(len(bt))})
        res[name] = r
    p = res["pooled"]
    if "or_cancer_ida_ci" in p:
        each = [bool(res[k].get("or_cancer_ida", 0) > 1) for k in ("NHANES 2001-2002", "NHANES III 1988-1994")]
        res["bars"] = {"S1_signal": bool(p["or_cancer_ida_ci"][0] > 1 and all(each)), "S1_each_dataset_above_1": each,
                       "S2_cancer_specific": bool(p["ratio_of_ors_ci"][0] > 1), "S3_ida_beats_non_iron": bool(p["ida_minus_nia_ci"][0] > 1)}
    return res


def main():
    out = {"A_societal_2017_2020": part_a()}
    print(json.dumps(out["A_societal_2017_2020"], indent=1)[:3000])
    for yrs in (5, 10):
        out[f"B_scientific_{yrs}y"] = part_b(yrs)
        print(f"\n--- {yrs}-year"); print(json.dumps(out[f"B_scientific_{yrs}y"], indent=1))
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
