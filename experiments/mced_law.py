"""
Who Medicare's multi-cancer early detection (MCED) test law reaches, and what free questions add.
Questions, assumptions and bars: docs/MCED_LAW_PREREG.md (committed before this was run).

Everything descriptive is NHIS interview years 2005-2009, survey-weighted (sample-adult weight).
Outcome: death from cancer within five years of interview. Yield per test is anchored to PATHFINDER 2
(detection 0.54% in adults 50+, specificity 99.64%) through the OBSERVED cancer-death rate of each group,
so ranking people by the model and then scoring them by the model never happens.

Run:  python experiments/mced_law.py       needs data/nhis_cancer_risk.csv.gz with the medicare column
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from cancer_age import CANCER, Fit

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "mced_law_result.json")
RNG = np.random.default_rng(20261012)
DETECT, SPEC, PRICE = 0.0054, 0.9964, 508.87
B = 400


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    d = d[(d["age"] >= 35) & (d["age"] <= 84) & (d["prior_cancer"] == 0) & (d["year"] <= 2009)].copy()
    d = d[(d["months"] >= 60) | (d["mortstat"] == 1)]
    d["y"] = ((d["mortstat"] == 1) & (d["ucod"] == CANCER) & (d["months"] <= 60)).astype(int)
    d = d.dropna(subset=["smoke_status", "bmi"]).reset_index(drop=True)
    d["bmi"] = d["bmi"].clip(15, 55)
    d["quit_years"] = d["quit_years"].fillna(0).clip(0, 50)
    d["cigs_per_day"] = d["cigs_per_day"].fillna(0).clip(0, 60)
    return d


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def boot(fn, n, b=B):
    out = []
    for _ in range(b):
        i = RNG.integers(0, n, n)
        v = fn(i)
        if v is not None and np.isfinite(v):
            out.append(v)
    return [round(float(x), 4) for x in np.percentile(out, [2.5, 97.5])]


def yield_from_rate(rate, k):
    det = k * rate
    fp = (1 - SPEC) * (1 - det)
    return {"cancers_found_per_1000_tests": round(1000 * det, 2), "false_alarms_per_1000_tests": round(1000 * fp, 2),
            "ppv": round(det / (det + fp), 3), "medicare_test_cost_per_cancer_found": round(PRICE / det, 0)}


def main():
    d = load()
    tr = d[d["year"] <= 2002].reset_index(drop=True)
    te = d[(d["year"] >= 2005) & d["weight"].notna()].reset_index(drop=True)
    model = Fit(tr, 1)                     # age (spline), sex, smoking: the cheapest model that suffices
    te["p"] = model.predict(te)
    y, w, age, p = te["y"].to_numpy(float), te["weight"].to_numpy(float), te["age"].to_numpy(), te["p"].to_numpy()
    mc = te["medicare"].to_numpy()
    n = len(te)
    res = {"n_test": n, "cancer_deaths_5y": int(y.sum()), "medicare_known": round(float(np.isfinite(mc).mean()), 3),
           "medicare_share_under65": round(wmean(np.nan_to_num(mc) == 1, w * (age < 65)), 4) if True else None}
    print(f"NHIS 2005-2009: {n:,} adults, {int(y.sum())} five-year cancer deaths; Medicare recode known for {res['medicare_known']:.1%}")

    # anchor: adults 50+ reproduce PATHFINDER 2's detection rate
    a50 = age >= 50
    k = DETECT / wmean(y[a50], w[a50])
    res["anchor"] = {"observed_5y_cancer_death_rate_50plus": round(wmean(y[a50], w[a50]), 5), "k": round(k, 4)}

    under65_mc = (age < 65) & (mc == 1)
    elig_pop = (age >= 65) | under65_mc
    cov28 = ((age >= 65) & (age <= 67)) | under65_mc
    cov35 = ((age >= 65) & (age <= 74)) | under65_mc

    # Q1: share of five-year cancer deaths among Medicare-eligible adults that the law pays for
    print("\nQ1. Share of Medicare-eligible adults' five-year cancer deaths in people the law covers")
    res["Q1"] = {}
    for name, cov in (("2028", cov28), ("2035 (stylised)", cov35)):
        def share(i, cov=cov):
            e = elig_pop[i]
            return (y[i] * w[i] * cov[i] * e).sum() / (y[i] * w[i] * e).sum()
        s = share(np.arange(n))
        pop = (w * cov * elig_pop).sum() / (w * elig_pop).sum()
        ci = boot(share, n)
        res["Q1"][name] = {"share_of_cancer_deaths": round(float(s), 4), "ci": ci, "share_of_population": round(float(pop), 4)}
        print(f"  {name}: covers {pop:.1%} of Medicare-eligible adults and {s:.1%} of their cancer deaths [{ci[0]:.1%}, {ci[1]:.1%}]")
    res["Q1"]["bar_met"] = bool(res["Q1"]["2028"]["ci"][1] < 0.5)

    # Q2: yield per test by group, from observed rates
    print("\nQ2. Expected yield per 1,000 tests by group (observed five-year cancer-death rate x anchor)")
    groups = {"50-64 without Medicare": (age >= 50) & (age < 65) & (mc != 1), "under 65 with Medicare": under65_mc & (age >= 50),
              "65-67 (covered 2028)": (age >= 65) & (age <= 67), "68-74 (never covered)": (age >= 68) & (age <= 74), "75-84 (never covered)": age >= 75}
    res["Q2"] = {}
    for g, m in groups.items():
        rate = wmean(y[m], w[m])
        r = {"n": int(m.sum()), "cancer_deaths": int(y[m].sum()), "rate_5y": round(rate, 5), **yield_from_rate(rate, k)}
        # sensitivity: detection proportional to predicted risk^0.5, anchored the same way
        k2 = DETECT / wmean(np.sqrt(p[a50]), w[a50])
        det2 = wmean(k2 * np.sqrt(p[m]), w[m])
        r["sens_sqrt_cancers_per_1000"] = round(1000 * det2, 2)
        r["sens_sqrt_ppv"] = round(det2 / (det2 + (1 - SPEC) * (1 - det2)), 3)
        res["Q2"][g] = r
        print(f"  {g:<26} n={r['n']:>6,} deaths={r['cancer_deaths']:>4}  {r['cancers_found_per_1000_tests']:>5.2f}/1000  PPV {r['ppv']:.0%}  ${r['medicare_test_cost_per_cancer_found']:,.0f} per cancer   (sqrt: {r['sens_sqrt_cancers_per_1000']:.2f}, PPV {r['sens_sqrt_ppv']:.0%})")

    # Q3: free-information triage inside the 2028-covered group
    print("\nQ3. Inside the 2028-covered group: test the highest-risk 20%, 30%, 50% vs the same number at random")
    idx = np.where(cov28)[0]
    res["Q3"] = {}
    for share_tested in (0.2, 0.3, 0.5):
        def ratio(ii, share_tested=share_tested):
            ii = idx[ii] if ii is not None else idx
            ww, yy, pp = w[ii], y[ii], p[ii] + RNG.random(len(ii)) * 1e-9
            o = np.argsort(-pp)
            cw = np.cumsum(ww[o])
            top = o[: np.searchsorted(cw, share_tested * cw[-1]) + 1]
            r_top = (yy[top] * ww[top]).sum() / ww[top].sum()
            r_all = (yy * ww).sum() / ww.sum()          # random testing = the group's average
            return r_top / r_all if r_all > 0 else np.nan
        rt = ratio(None)
        ci = [round(float(x), 3) for x in np.percentile([ratio(RNG.integers(0, len(idx), len(idx))) for _ in range(B)], [2.5, 97.5])]
        r_all = wmean(y[idx], w[idx])
        res["Q3"][str(share_tested)] = {"yield_ratio": round(float(rt), 3), "ci": ci, "random": yield_from_rate(r_all, k), "triaged": yield_from_rate(r_all * rt, k)}
        t = res["Q3"][str(share_tested)]
        print(f"  test {share_tested:.0%}: yield x{rt:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  cancers/1000 {t['random']['cancers_found_per_1000_tests']:.2f} -> {t['triaged']['cancers_found_per_1000_tests']:.2f}, "
              f"PPV {t['random']['ppv']:.0%} -> {t['triaged']['ppv']:.0%}, cost/cancer ${t['random']['medicare_test_cost_per_cancer_found']:,.0f} -> ${t['triaged']['medicare_test_cost_per_cancer_found']:,.0f}")
    res["Q3"]["bar_met"] = bool(res["Q3"]["0.2"]["ci"][0] >= 1.25)

    # Q4: under-65 Medicare beneficiaries against same-age adults without Medicare (ages 35-64), age-standardised
    print("\nQ4. Under-65 Medicare beneficiaries vs same-age adults without Medicare (age-standardised to the latter)")
    u = (age < 65) & np.isfinite(mc)
    band = (age // 5).astype(int)

    def std_rate(ii, val, grp):
        ii = ii[u[ii]]
        ref = ii[mc[ii] != 1]
        tgt = ii[mc[ii] == grp]
        tot = 0.0
        refw = w[ref].sum()
        for b_ in np.unique(band[ref]):
            rb, tb = ref[band[ref] == b_], tgt[band[tgt] == b_]
            if len(tb) == 0 or w[tb].sum() == 0:
                continue
            tot += w[rb].sum() / refw * (val[tb] * w[tb]).sum() / w[tb].sum()
        return tot

    allidx = np.arange(n)
    smk = (te["smoke_status"].to_numpy() == 2).astype(float)
    res["Q4"] = {"n_medicare_under65": int(((mc == 1) & (age < 65)).sum()), "cancer_deaths_medicare_under65": int(y[(mc == 1) & (age < 65)].sum())}
    for nm, val in (("observed_5y_cancer_death", y), ("predicted_5y_risk", p), ("current_smoker", smk)):
        a, b_ = std_rate(allidx, val, 1), std_rate(allidx, val, 0)
        ci = boot(lambda i, val=val: std_rate(i, val, 1) - std_rate(i, val, 0), n, 200)
        res["Q4"][nm] = {"medicare": round(a, 5), "no_medicare": round(b_, 5), "ratio": round(a / b_, 2), "diff_ci": ci}
        print(f"  {nm:<26} Medicare {a:.4f}  no Medicare {b_:.4f}  ratio {a / b_:.2f}  diff CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    res["Q4"]["bar_met"] = bool(res["Q4"]["observed_5y_cancer_death"]["diff_ci"][0] > 0 and res["Q4"]["predicted_5y_risk"]["diff_ci"][0] > 0)
    print("\nbars: Q1", res["Q1"]["bar_met"], " Q3", res["Q3"]["bar_met"], " Q4", res["Q4"]["bar_met"])
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
