"""
Survey-weighted version of the cost-matched comparison, and what it means for the whole US adult population.
Exploratory; added after the main results (the main analysis is unweighted). Uses the NHIS sample-adult weight
(WTFA_SA); bootstrap resamples respondents and ignores the survey's clustering, so intervals are a little narrow.

Run:  python experiments/cancer_age_national.py
"""

import json
import os
import warnings

import numpy as np

from cancer_age import Fit, load_nhis

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "cancer_age_national_result.json")
RNG = np.random.default_rng(20261012)


def wcov(y, w, key, share):
    """Weighted share of cancer deaths among the top `share` of the weighted population by key."""
    o = np.argsort(-key, kind="stable")
    cw = np.cumsum(w[o])
    k = np.searchsorted(cw, share * cw[-1])
    return float((y[o][: k + 1] * w[o][: k + 1]).sum() / (y * w).sum())


def main():
    d = load_nhis()
    tr, te = d[d["year"] <= 2002].reset_index(drop=True), d[d["year"] >= 2005].reset_index(drop=True)
    te = te[te["age"].between(40, 74) & te["weight"].notna()].reset_index(drop=True)
    p = Fit(tr, 2).predict(te)
    y, w = te["y"].to_numpy().astype(float), te["weight"].to_numpy().astype(float)
    age = te["age"].to_numpy() + RNG.random(len(te)) * 0.01
    pop = w.sum() / 5.0      # five pooled interview years -> average yearly population
    res = {"n": int(len(te)), "population_40_74_millions": round(pop / 1e6, 1), "weighted_cancer_deaths_10y_millions": round(float((y * w).sum() / 5.0 / 1e6), 3), "shares": {}, "fixed_coverage": {}}
    print(f"US adults 40-74 (civilian, non-institutionalised, 2005-09 average): {pop / 1e6:.1f} million")
    for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        a, r = wcov(y, w, age, s), wcov(y, w, p, s)
        dd = []
        for _ in range(300):
            i = RNG.integers(0, len(y), len(y))
            dd.append(wcov(y[i], w[i], p[i], s) - wcov(y[i], w[i], age[i], s))
        ci = np.percentile(dd, [2.5, 97.5])
        res["shares"][str(s)] = {"age_only": round(a, 4), "risk_based": round(r, 4), "diff": round(r - a, 4), "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}
        print(f"  invite {s:.0%}: age {a:.1%}  risk {r:.1%}  diff {r - a:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    grid = np.linspace(0.01, 1.0, 100)
    for tgt in (0.4, 0.5, 0.6):
        need = {}
        for nm, key in (("age", age), ("risk", p)):
            cov = np.array([wcov(y, w, key, s) for s in grid])
            need[nm] = float(grid[np.argmax(cov >= tgt)])
        saved = need["age"] - need["risk"]
        res["fixed_coverage"][str(tgt)] = {"share_invited_age": round(need["age"], 2), "share_invited_risk": round(need["risk"], 2),
                                           "invitations_saved_millions": round(saved * pop / 1e6, 1), "fewer_invitations": round(saved / need["age"], 3)}
        print(f"  to cover {tgt:.0%} of cancer deaths: age {need['age']:.0%}, risk {need['risk']:.0%} -> {saved * pop / 1e6:.1f} million fewer invitations ({saved / need['age']:.0%})")
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
