"""Independent re-implementation of the MCED-law analysis (Q1, Q3, Q4) plus robustness checks.

Written from the definitions in docs/MCED_LAW_PREREG.md; does not import project code.
Run from the repo root:  python docs/independent_check/mced/recheck.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "recheck_result.json")
rng = np.random.default_rng(7)
NB = 300


def rcs(x, knots):
    """Restricted cubic spline basis (Harrell), linear term + k-2 nonlinear terms."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    t = k[-1] - k[-2]
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    for j in range(len(k) - 2):
        c = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / t
             + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / t)
        cols.append(c / norm)
    return np.column_stack(cols)


def load(years_out=5, keep_prior=False):
    d = pd.read_csv(os.path.join(ROOT, "data", "nhis_cancer_risk.csv.gz"))
    d = d[(d.age.between(35, 84)) & (d.year <= 2009)].copy()
    if not keep_prior:
        d = d[d.prior_cancer == 0]
    m = 12 * years_out
    # alive people all have >=121 months of NDI follow-up, so no censoring inside 5 or 10 years
    assert (d.loc[d.mortstat != 1, "months"] >= m).all()
    d["y"] = ((d.mortstat == 1) & (d.ucod == 2) & (d.months <= m)).astype(int)
    d = d.dropna(subset=["smoke_status", "bmi"]).copy()  # same analytic sample definition as the prereg ladder
    d["cpd"] = d.cigs_per_day.fillna(0).clip(0, 60)
    d["qy"] = d.quit_years.fillna(0).clip(0, 50)
    return d.reset_index(drop=True)


class Model:
    def __init__(self, tr, smoking=True):
        self.smoking = smoking
        self.knots = np.percentile(tr.age, [5, 27.5, 50, 72.5, 95])
        self.m = {}
        for s in (0.0, 1.0):
            g = tr[tr.male == s]
            self.m[s] = LogisticRegression(C=1e6, max_iter=10000).fit(self.X(g), g.y)

    def X(self, g):
        cols = [rcs(g.age, self.knots)]
        if self.smoking:
            ss = g.smoke_status.to_numpy()
            cols.append(np.column_stack([ss == 1, ss == 2, g.cpd / 20, (ss == 1) * g.qy / 10, (ss == 1) * (g.qy / 10) ** 2]).astype(float))
        return np.hstack(cols)

    def predict(self, g):
        p = np.zeros(len(g))
        for s in (0.0, 1.0):
            k = (g.male == s).to_numpy()
            p[k] = self.m[s].predict_proba(self.X(g[k]))[:, 1]
        return p


def wr(y, w):
    return float(np.sum(y * w) / np.sum(w))


def top_ratio(y, w, p, frac=0.2):
    o = np.argsort(-(p + rng.random(len(p)) * 1e-12))
    cw = np.cumsum(w[o]) / w.sum()
    top = o[cw <= frac + 1e-12]
    if len(top) == 0:
        top = o[:1]
    return wr(y[top], w[top]) / wr(y, w)


def ci(fn, n, nb=NB):
    v = [fn(rng.integers(0, n, n)) for _ in range(nb)]
    v = [x for x in v if np.isfinite(x)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def analysis(years_out=5, keep_prior=False, weighted=True, label=""):
    d = load(years_out, keep_prior)
    tr = d[d.year <= 2002]
    te = d[d.year >= 2005].reset_index(drop=True)
    te["p"] = Model(tr).predict(te)
    te["p0"] = Model(tr, smoking=False).predict(te)
    y = te.y.to_numpy(float)
    w = te.weight.to_numpy(float) if weighted else np.ones(len(te))
    age = te.age.to_numpy()
    mc = te.medicare.to_numpy()
    u65mc = (age < 65) & (mc == 1)
    elig = (age >= 65) | u65mc
    elig_mc = ((age >= 65) & (mc == 1)) | u65mc
    cov28 = (age <= 67) & elig
    enacted29 = u65mc & (age >= 50)  # enacted law, 2029: under 65 on 1 Jan and 50+, i.e. under-65 beneficiaries 50-64
    r = {"label": label, "n": len(te), "deaths": int(y.sum())}

    def share(cov, base):
        f = lambda i: np.sum((y * w * cov * base)[i]) / np.sum((y * w * base)[i])
        return {"pop_share": round(float(np.sum(w * cov * base) / np.sum(w * base)), 4),
                "death_share": round(float(f(np.arange(len(te)))), 4), "ci": ci(f, len(te))}

    r["Q1_2028_prereg"] = share(cov28, elig)
    r["Q1_2028_only_medicare_65plus"] = share(cov28 & elig_mc, elig_mc)
    r["Q1_enacted_2029_u65mc_50to64"] = share(enacted29, elig)

    a50 = age >= 50
    r["anchor_rate50"] = round(wr(y[a50], w[a50]), 5)
    r["anchor_k"] = round(0.0054 / wr(y[a50], w[a50]), 4)

    def q3(mask, col="p", frac=0.2):
        idx = np.where(mask)[0]
        pp = te[col].to_numpy()
        f = lambda i: top_ratio(y[idx[i]], w[idx[i]], pp[idx[i]], frac)
        return {"n": len(idx), "deaths": int(y[idx].sum()), "ratio": round(f(np.arange(len(idx))), 3), "ci": ci(f, len(idx))}

    r["Q3_cov28_full"] = q3(cov28)
    r["Q3_cov28_agesex_only"] = q3(cov28, "p0")
    r["Q3_65to67_only"] = q3(cov28 & (age >= 65))
    r["Q3_65to67_agesex_only"] = q3(cov28 & (age >= 65), "p0")
    r["Q3_u65mc_only"] = q3(u65mc)
    r["Q3_enacted29_full"] = q3(enacted29)
    r["Q3_enacted29_agesex_only"] = q3(enacted29, "p0")

    # Q4: age-standardised (5-yr bands, standard = no-Medicare 35-64) Medicare vs no-Medicare, under 65
    band = age // 5

    def std(i, val, g):
        i = i[(age[i] < 65) & np.isfinite(mc[i])]
        ref = i[mc[i] == 0]
        tg = i[mc[i] == g]
        tot = 0.0
        for b in np.unique(band[ref]):
            rb, tb = ref[band[ref] == b], tg[band[tg] == b]
            if w[tb].sum() > 0:
                tot += w[rb].sum() / w[ref].sum() * wr(val[tb], w[tb])
        return tot

    alli = np.arange(len(te))
    for nm, v in (("observed", y), ("predicted", te.p.to_numpy())):
        a, b = std(alli, v, 1), std(alli, v, 0)
        r[f"Q4_{nm}"] = {"medicare": round(a, 5), "none": round(b, 5), "ratio": round(a / b, 3),
                         "ratio_ci": ci(lambda i, v=v: std(i, v, 1) / std(i, v, 0), len(te), 200)}
    r["Q4_n_u65mc"] = int(u65mc.sum())
    r["Q4_deaths_u65mc"] = int(y[u65mc].sum())
    r["Q4_n_u65mc_50to64"] = int(enacted29.sum())
    r["Q4_deaths_u65mc_50to64"] = int(y[enacted29].sum())
    return r


if __name__ == "__main__":
    runs = [analysis(label="main: 5y, weighted, prior cancer excluded"),
            analysis(years_out=10, label="10y outcome"),
            analysis(weighted=False, label="unweighted"),
            analysis(keep_prior=True, label="prior cancer kept")]
    json.dump(runs, open(OUT, "w"), indent=1)
    json.dump(runs, sys.stdout, indent=1)
