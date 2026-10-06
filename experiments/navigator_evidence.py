"""
Navigator evidence study. Bars are in docs/NAVIGATOR_EVIDENCE_PREREG.md and were
committed before this was run.

Runs the REAL rule engine (backend/navigator.py) on 39,692 NHANES adults (1999-2014, linked
to the National Death Index) and on NHANES III (1988-1994), then answers six questions.

Run:  python experiments/navigator_evidence.py
Needs data/nhanes_navigator.csv.gz (python fetch_nhanes_navigator.py).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import navigator as nav  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "navigator_evidence_result.json")
CACHE = os.path.join(ROOT, "data", "navigator_alerts.csv.gz")
RNG = np.random.default_rng(20261005)
ALERT = ("talk_soon", "worth_raising", "mention")
BANDS = [(40, 49), (50, 59), (60, 69), (70, 79), (80, 120)]
CANCER = 2


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0, c - h)), 4), round(float(min(1, c + h)), 4)]


def labs_of(r):
    labs = {}
    for k in ("hemoglobin", "mcv", "platelets", "wbc", "ferritin"):
        v = r.get(k)
        if v is not None and not pd.isna(v):
            labs[k] = float(v)
    return labs


def run_engine(req):
    try:
        out = nav.evaluate(req)
    except ValueError:
        return None, []
    return out["state"], [m["id"] for m in out["matches"]]


def score(df, with_context):
    """Run the engine on each row. with_context adds smoking and proxy symptoms."""
    memo, states, rules = {}, [], []
    for r in df.to_dict("records"):
        req = {"age": float(r["age"]), "sex": r["sex"], "labs": labs_of(r)}
        if with_context:
            if not pd.isna(r.get("ever_smoked", np.nan)):
                req["ever_smoked"] = bool(r["ever_smoked"])
            syms = [{"key": k} for k in ("weight_loss", "cough", "shortness_of_breath") if r.get(k) == 1]
            req["symptoms"] = syms
        key = json.dumps(req, sort_keys=True)
        if key not in memo:
            memo[key] = run_engine(req)
        s, ids = memo[key]
        states.append(s)
        rules.append("|".join(ids))
    return states, rules


def irls(X, y, iters=50):
    """Logistic regression with Wald standard errors. X includes the intercept column."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    W = p * (1 - p) + 1e-9
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def or_ci(X, y, j):
    b, se = irls(X, y)
    return {"or": round(float(np.exp(b[j])), 3),
            "ci": [round(float(np.exp(b[j] - 1.96 * se[j])), 3), round(float(np.exp(b[j] + 1.96 * se[j])), 3)]}


def auc_gain(y, p_base, p_full, n_boot=1000):
    a0, a1 = roc_auc_score(y, p_base), roc_auc_score(y, p_full)
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    gains = []
    for _ in range(n_boot):
        i = np.concatenate([RNG.choice(idx_pos, len(idx_pos)), RNG.choice(idx_neg, len(idx_neg))])
        gains.append(roc_auc_score(y[i], p_full[i]) - roc_auc_score(y[i], p_base[i]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"auc_base": round(float(a0), 4), "auc_with_alert": round(float(a1), 4),
            "gain": round(float(a1 - a0), 4), "gain_ci": [round(float(lo), 4), round(float(hi), 4)]}


def design(df, extras=()):
    cols = [np.ones(len(df)), df["age"].to_numpy() - 60, (df["age"].to_numpy() - 60) ** 2 / 20,
            (df["sex"] == "male").astype(float).to_numpy()]
    for e in extras:
        cols.append(df[e].fillna(0).astype(float).to_numpy())
    return np.column_stack(cols)


def main():
    cohort = pd.read_csv(os.path.join(ROOT, "data", "nhanes_navigator.csv.gz"))
    cohort = cohort[(cohort["age"] >= 40)].dropna(subset=["hemoglobin", "platelets"]).reset_index(drop=True)
    cohort["year0"] = cohort["cycle"].str[:4].astype(int)
    cohort["era"] = np.where(cohort["year0"] <= 2005, "1999-2006", np.where(cohort["year0"] <= 2013, "2007-2014", "2015-2018"))

    n3 = pd.read_csv(os.path.join(ROOT, "data", "nhanes3_mortality_full.csv"))
    n3 = n3[(n3["age"] >= 40)].dropna(subset=["hemoglobin", "platelets"]).reset_index(drop=True)
    n3 = n3.rename(columns={"followup_months": "months", "ucod_leading": "ucod"})
    n3["sex"] = np.where(n3["gender"] == 1, "male", "female")
    n3["era"] = "1988-1994 (NHANES III)"
    for c in ("ferritin", "ever_smoked", "weight_loss", "cough", "shortness_of_breath"):
        n3[c] = np.nan
    n3["died"] = n3["died"].astype(float)

    print(f"NHANES 1999-2018: {len(cohort):,} adults 40+;  NHANES III: {len(n3):,}\n")

    if os.path.exists(CACHE):
        cache = pd.read_csv(CACHE)
        for c in ("lab_state", "lab_rules", "ctx_state", "ctx_rules"):
            cohort[c] = cache[c].where(cache[c].notna(), None).to_numpy() if len(cache) == len(cohort) else None
    if "lab_state" not in cohort or cohort["lab_state"].isna().all():
        cohort["lab_state"], cohort["lab_rules"] = score(cohort, False)
        cohort["ctx_state"], cohort["ctx_rules"] = score(cohort, True)
        cohort[["SEQN", "cycle", "lab_state", "lab_rules", "ctx_state", "ctx_rules"]].to_csv(CACHE, index=False, compression="gzip")
    n3["lab_state"], n3["lab_rules"] = score(n3, False)

    result = {"n_cohort": int(len(cohort)), "n_nhanes3": int(len(n3))}

    # Q1 burden by era
    print("Q1  Labs-only alert burden by era and age band")
    q1 = {}
    allsets = {"1988-1994 (NHANES III)": n3, "1999-2006": cohort[cohort["era"] == "1999-2006"],
               "2007-2014": cohort[cohort["era"] == "2007-2014"], "2015-2018": cohort[cohort["era"] == "2015-2018"]}
    worst_talk, worst_any = 0.0, 0.0
    for era, d in allsets.items():
        q1[era] = {}
        for lo, hi in BANDS:
            m = d[(d["age"] >= lo) & (d["age"] <= hi)]
            n = len(m)
            row = {"n": int(n)}
            for name, states in (("talk_soon", ["talk_soon"]), ("any_alert", list(ALERT))):
                k = int(m["lab_state"].isin(states).sum())
                row[name] = {"n": k, "share": round(k / max(n, 1), 4), "ci": wilson(k, n)}
            q1[era][f"{lo}-{hi if hi < 120 else '+'}"] = row
            worst_talk = max(worst_talk, row["talk_soon"]["share"])
            worst_any = max(worst_any, row["any_alert"]["share"])
            print(f"   {era:<24}{lo}-{hi if hi < 120 else '+':<4}{n:>6}  talk soon {row['talk_soon']['share']:>6.1%}  any {row['any_alert']['share']:>6.1%}")
    q1_pass = worst_talk <= 0.03 and worst_any <= 0.20
    result["q1"] = {"bands": q1, "worst_talk_soon": round(worst_talk, 4), "worst_any_alert": round(worst_any, 4),
                    "bar": "talk soon <= 3% and any alert <= 20% in every band and era", "passes": bool(q1_pass)}
    print(f"   worst talk-soon {worst_talk:.1%}, worst any-alert {worst_any:.1%}  ->  {'PASSES' if q1_pass else 'FAILS'} the bar\n")

    # Everything below needs an outcome, so it uses only cycles with enough follow-up.
    cohort_all = cohort
    cohort = cohort[cohort["year0"] <= 2013].reset_index(drop=True)
    result["n_outcome_cohort"] = int(len(cohort))

    # Q2 cancer-specific among decedents
    print("Q2  Among people who died within 10 years: are alerted people more likely to die of cancer?")
    q2 = {}
    for label, d, col in (("1999-2014 labs only", cohort, "lab_state"), ("1999-2014 labs and proxies", cohort, "ctx_state"),
                          ("NHANES III labs only", n3, "lab_state")):
        dd = d[(d["died"] == 1) & (d["months"] <= 120)].copy()
        dd["alert"] = dd[col].isin(ALERT).astype(float)
        dd["cancer"] = (dd["ucod"] == CANCER).astype(int)
        extras = ["alert"] + (["year0"] if "year0" in dd else [])
        if "year0" in dd:
            dd["year0"] = dd["year0"] - 2006
        X = design(dd, extras)
        r = or_ci(X, dd["cancer"].to_numpy(), 4)
        r.update({"decedents": int(len(dd)), "cancer_deaths": int(dd["cancer"].sum()), "alerted": int(dd["alert"].sum()),
                  "cancer_share_alerted": round(float(dd.loc[dd["alert"] == 1, "cancer"].mean()), 4),
                  "cancer_share_not": round(float(dd.loc[dd["alert"] == 0, "cancer"].mean()), 4)})
        q2[label] = r
        print(f"   {label:<28} deaths {r['decedents']:>5} (cancer {r['cancer_deaths']:>4}), alerted {r['alerted']:>4}: "
              f"cancer share {r['cancer_share_alerted']:.1%} vs {r['cancer_share_not']:.1%}, adjusted OR {r['or']} {r['ci']}")
    q2_pass = q2["1999-2014 labs and proxies"]["ci"][0] > 1 and q2["NHANES III labs only"]["or"] > 1
    result["q2"] = {"results": q2, "bar": "adjusted OR 95% CI above 1 in pooled 1999-2014 (labs and proxies) and OR above 1 in NHANES III",
                    "passes": bool(q2_pass)}
    print(f"   -> {'PASSES' if q2_pass else 'FAILS'} the bar\n")

    # Q3 added value over age and sex
    print("Q3  Five-year cancer death: does an alert add to age and sex?")
    def five_year(d):
        d = d.copy()
        known = (d["months"] >= 60) | (d["died"] == 1)
        d = d[known]
        d["y"] = ((d["died"] == 1) & (d["ucod"] == CANCER) & (d["months"] <= 60)).astype(int)
        return d
    c5 = five_year(cohort)
    n35 = five_year(n3)
    for d in (c5, n35):
        d["alert_ctx"] = d["ctx_state"].isin(ALERT).astype(float) if "ctx_state" in d and d["ctx_state"].notna().any() else d["lab_state"].isin(ALERT).astype(float)
        d["alert_lab"] = d["lab_state"].isin(ALERT).astype(float)
        d["talk"] = d["lab_state"].eq("talk_soon").astype(float)
    tr, te = c5[c5["era"] == "1999-2006"], c5[c5["era"] == "2007-2014"]
    q3 = {}
    for label, extras in (("age+sex vs +labs alert", ("alert_lab",)), ("age+sex vs +labs and proxies alert", ("alert_ctx",))):
        base = LogisticRegression(penalty=None, max_iter=1000).fit(design(tr)[:, 1:], tr["y"])
        full = LogisticRegression(penalty=None, max_iter=1000).fit(design(tr, extras)[:, 1:], tr["y"])
        g = auc_gain(te["y"].to_numpy(), base.predict_proba(design(te)[:, 1:])[:, 1],
                     full.predict_proba(design(te, extras)[:, 1:])[:, 1])
        g.update({"train_n": int(len(tr)), "test_n": int(len(te)), "test_cancer_deaths": int(te["y"].sum())})
        q3[label] = g
        print(f"   {label:<38} AUC {g['auc_base']:.3f} -> {g['auc_with_alert']:.3f}, gain {g['gain']:+.4f} {g['gain_ci']}")
    base = LogisticRegression(penalty=None, max_iter=1000).fit(design(c5)[:, 1:], c5["y"])
    full = LogisticRegression(penalty=None, max_iter=1000).fit(design(c5, ("alert_lab",))[:, 1:], c5["y"])
    g3 = auc_gain(n35["y"].to_numpy(), base.predict_proba(design(n35)[:, 1:])[:, 1],
                  full.predict_proba(design(n35, ("alert_lab",))[:, 1:])[:, 1])
    g3.update({"test_n": int(len(n35)), "test_cancer_deaths": int(n35["y"].sum())})
    q3["trained 1999-2014, tested NHANES III (labs alert)"] = g3
    print(f"   trained 1999-2014, tested NHANES III     AUC {g3['auc_base']:.3f} -> {g3['auc_with_alert']:.3f}, gain {g3['gain']:+.4f} {g3['gain_ci']}")
    q3_pass = q3["age+sex vs +labs and proxies alert"]["gain_ci"][0] > 0
    result["q3"] = {"results": q3, "bar": "AUC gain 95% bootstrap interval excludes 0 in 2007-2014 (labs and proxies)", "passes": bool(q3_pass)}
    print(f"   -> {'PASSES' if q3_pass else 'FAILS'} the bar\n")

    # Q4 ferritin vs the MCV proxy
    print("Q4  Does small red cells (MCV < 80) stand in for low ferritin?")
    f = cohort_all.dropna(subset=["ferritin", "mcv", "hemoglobin"]).copy()
    anaemic = np.where(f["sex"] == "male", f["hemoglobin"] < 13, f["hemoglobin"] < 12)
    ida_true = anaemic & (f["ferritin"] < 15)
    ida_proxy = anaemic & (f["mcv"] < 80)
    tp = int((ida_true & ida_proxy).sum())
    q4 = {"n_with_ferritin": int(len(f)), "anaemic": int(anaemic.sum()), "true_ida_by_ferritin": int(ida_true.sum()),
          "proxy_ida_by_mcv": int(ida_proxy.sum()), "proxy_sensitivity": round(tp / max(int(ida_true.sum()), 1), 3),
          "proxy_ppv": round(tp / max(int(ida_proxy.sum()), 1), 3)}
    f60 = f[f["age"] >= 60]
    # How often the colorectal "urgent" rule would fire with ferritin versus the proxy.
    with_f, without_f = score(f60, False)[0], score(f60.assign(ferritin=np.nan), False)[0]
    q4["age60_talk_soon_with_ferritin"] = int(pd.Series(with_f).eq("talk_soon").sum())
    q4["age60_talk_soon_proxy_only"] = int(pd.Series(without_f).eq("talk_soon").sum())
    q4["age60_n"] = int(len(f60))
    result["q4"] = q4
    print(f"   {q4['n_with_ferritin']:,} adults with ferritin; {q4['anaemic']} anaemic; {q4['true_ida_by_ferritin']} iron-deficient by ferritin")
    print(f"   proxy sensitivity {q4['proxy_sensitivity']:.0%}, proxy PPV {q4['proxy_ppv']:.0%}")
    print(f"   age 60+ ({q4['age60_n']}): 'talk soon' {q4['age60_talk_soon_proxy_only']} with the proxy, {q4['age60_talk_soon_with_ferritin']} with measured ferritin\n")

    # Q5 per rule, descriptive
    print("Q5  Rules that fired, labs and proxies, and five-year cancer death (descriptive)")
    q5 = {}
    rc = c5.copy()
    rc["rule_list"] = rc["ctx_rules"].fillna("").str.split("|")
    base_rate = float(rc["y"].mean())
    for rid in sorted({r for rs in rc["rule_list"] for r in rs if r}):
        m = rc["rule_list"].apply(lambda rs: rid in rs)
        n, k = int(m.sum()), int(rc.loc[m, "y"].sum())
        q5[rid] = {"n": n, "cancer_deaths_5y": k, "rate": round(k / n, 4), "ci": wilson(k, n),
                   "mean_age": round(float(rc.loc[m, "age"].mean()), 1)}
        print(f"   {rid:<30} n={n:>6}  5-yr cancer death {k / n:>6.1%}  mean age {q5[rid]['mean_age']}")
    q5["_everyone"] = {"n": int(len(rc)), "rate": round(base_rate, 4)}
    result["q5"] = q5
    print()

    # Q6 context layer, calibrated out of era
    print("Q6  Age, sex and smoking context layer, trained 1999-2006 and tested 2007-2014")
    cols = ["age", "male", "smoked"]
    def ctxX(d):
        return np.column_stack([d["age"] - 60, (d["age"] - 60) ** 2 / 20, (d["sex"] == "male").astype(float),
                                d["ever_smoked"].fillna(d["ever_smoked"].mean() if d["ever_smoked"].notna().any() else 0)])
    m6 = LogisticRegression(penalty=None, max_iter=1000).fit(ctxX(tr), tr["y"])
    p = m6.predict_proba(ctxX(te))[:, 1]
    logit = np.log(p / (1 - p))
    Xc = np.column_stack([np.ones(len(te)), logit])
    b, _ = irls(Xc, te["y"].to_numpy().astype(float))
    brier = float(np.mean((p - te["y"]) ** 2))
    brier_null = float(np.mean((tr["y"].mean() - te["y"]) ** 2))
    bins = pd.qcut(pd.Series(p), 10, duplicates="drop")
    calib = [{"predicted": round(float(g.mean()), 4), "observed": round(float(te["y"].to_numpy()[g.index].mean()), 4),
              "n": int(len(g))} for _, g in pd.Series(p).groupby(bins, observed=True)]
    q6 = {"calibration_slope": round(float(b[1]), 3), "calibration_intercept": round(float(b[0]), 3),
          "auc": round(float(roc_auc_score(te["y"], p)), 4), "brier": round(brier, 5), "brier_if_no_model": round(brier_null, 5),
          "deciles": calib, "coefficients": {"age_minus_60": float(m6.coef_[0][0]), "age_sq": float(m6.coef_[0][1]),
                                              "male": float(m6.coef_[0][2]), "ever_smoked": float(m6.coef_[0][3]),
                                              "intercept": float(m6.intercept_[0])}}
    q6["passes"] = bool(0.8 <= q6["calibration_slope"] <= 1.2)
    q6["bar"] = "calibration slope between 0.8 and 1.2 in 2007-2014"
    result["q6"] = q6
    print(f"   AUC {q6['auc']:.3f}, calibration slope {q6['calibration_slope']}, intercept {q6['calibration_intercept']}, "
          f"Brier {q6['brier']:.5f} (no model {q6['brier_if_no_model']:.5f})  ->  {'PASSES' if q6['passes'] else 'FAILS'} the bar")

    # Q7 exploratory: which individual flags are cancer-specific among decedents
    print("\nQ7  Exploratory: each single flag, decedents within 10 years, cancer versus other death (age, sex, year adjusted)")
    dd = cohort[(cohort["died"] == 1) & (cohort["months"] <= 120)].copy()
    dd["cancer"] = (dd["ucod"] == CANCER).astype(int)
    dd["year0"] = dd["year0"] - 2006
    male = dd["sex"] == "male"
    flags = {
        "anaemia": np.where(male, dd["hemoglobin"] < 13, dd["hemoglobin"] < 12),
        "small red cells (MCV < 80)": dd["mcv"] < 80,
        "high platelets (> 400)": dd["platelets"] > 400,
        "raised white count (> 11)": dd["wbc"] > 11,
        "weight loss (proxy)": dd["weight_loss"] == 1,
        "chronic cough (proxy)": dd["cough"] == 1,
        "breathless (proxy)": dd["shortness_of_breath"] == 1,
        "ever smoked": dd["ever_smoked"] == 1,
        "any alert, labs only": dd["lab_state"].isin(ALERT),
    }
    q7 = {}
    for name, flag in flags.items():
        dd["flag"] = pd.Series(np.asarray(flag, dtype=float), index=dd.index)
        use = dd[dd["flag"].notna()]
        if use["flag"].sum() < 20:
            continue
        r = or_ci(design(use, ["flag", "year0"]), use["cancer"].to_numpy(), 4)
        r.update({"decedents_with_flag": int(use["flag"].sum()), "decedents": int(len(use))})
        q7[name] = r
        print(f"   {name:<30} flagged {r['decedents_with_flag']:>5} of {r['decedents']:>5}   OR {r['or']:>5}  {r['ci']}")
    result["q7"] = {"results": q7, "note": "Exploratory. Nine comparisons, so only intervals far from 1 should be believed. Not a basis for changing a rule without a clinician."}

    result["summary"] = {"q1": q1_pass, "q2": q2_pass, "q3": q3_pass, "q6": q6["passes"]}
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
