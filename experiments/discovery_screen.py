"""
Hypothesis-free discovery screen: does ANY free NHIS questionnaire item add to cancer-death risk beyond age, sex,
smoking and BMI, replicably and specifically for cancer?  Rules: docs/DISCOVERY_SCREEN_PREREG.md (committed first).

Stages: (1) discovery 1997-2002, likelihood-ratio test of each item on top of the baseline, false discovery rate
q < 0.05; (2) the hit list is written to disk, then and only then (3) replication 2005-2009 with Bonferroni over
the hits and the same direction; (4) specificity (cancer gain at least half the other-cause gain); (5) usefulness
(AUC gain >= 0.005 and coverage gain at 20% invited >= 1 point, intervals above zero).

One recorded departure from the pre-registration, decided before any result: interview-year indicators are added to
the baseline (a variable asked only in some years would otherwise be confounded with period trends in cancer death).

Run:  python experiments/discovery_screen.py [equity]
"""

import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, "data", "audit", "nhis_full")
RES = os.path.join(ROOT, "experiments")
RNG = np.random.default_rng(20261018)
DISC, REPL = [1997, 1998, 1999, 2000, 2001, 2002], [2005, 2006, 2007, 2008, 2009]
C_REG = 1e4

# Items that are the baseline itself, survey machinery, or cancer-related (leakage / reverse causation).
EXCL_NAME = re.compile(r"^(SRVY_YR|WTFA_SA|STRATUM|PSU|INTV_QRT|ASTATFLG|CSTATFLG|AGE_P|SEX|BMI|AHEIGHT|AWEIGHTP|SMK|CIGSDAY|CIGDAMO|CIGQTYR|SMKQT|SMKEV|SMKNOW|SMKSTAT|"
                       r"CAN|CN|PAP|MAM|PSA|CRE|CBE|SKNX|HFO|RHFO|MDHFO|SIG|COL|LUN|BREAST|HYST|HPV|MD(REC|HFOB)|RMAM|RPAP|RPSA|RCRE|DOD|elig|died|ucod|dodq|dody|pid|k$|year|REGION$|FMX|HHX|FPX|PX$|"
                       r"P_)")
EXCL_LABEL = re.compile(r"cancer|tumou?r|chemo|radiation|screen|mammogra|pap smear|psa|colonosc|sigmoid|stool|biops|malignan|mortality|age first|year of death|weight|height|body mass|smok|cigar", re.I)


def read_labels(year):
    import requests
    txt = requests.get(f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Program_Code/NHIS/{year}/samadult.sas", verify=False, timeout=240, headers={"User-Agent": "Mozilla/5.0"}).content.decode("latin-1")
    out = {}
    for m in re.finditer(r'^\s+([A-Z][A-Z0-9_]*)\s+="([^"]*)"', txt, re.M):
        out.setdefault(m.group(1), m.group(2).strip().lower())
    return out


def load():
    frames = []
    for f in sorted(os.listdir(FULL)):
        if f.endswith(".csv.gz"):
            frames.append(pd.read_csv(os.path.join(FULL, f), low_memory=False))
    d = pd.concat(frames, ignore_index=True, sort=False)
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    o = pd.DataFrame({"year": d["year"], "age": n("AGE_P"), "male": (n("SEX") == 1).astype(float), "bmi": n("BMI")})
    o.loc[o["bmi"] > 90, "bmi"] = np.nan
    ever, now = n("SMKEV"), n("SMKNOW")
    st = np.where(ever == 2, 0, np.where((ever == 1) & now.isin([1, 2]), 2, np.where((ever == 1) & (now == 3), 1, np.nan)))
    o["smoke_status"] = st
    cig = n("CIGSDAY").where(n("CIGSDAY").between(1, 96))
    o["cigs_per_day"] = np.where(st == 2, cig, 0.0)
    unit = {1: 1 / 365.25, 2: 1 / 52.18, 3: 1 / 12.0, 4: 1.0}
    yrs = n("SMKQTNO").where(n("SMKQTNO").between(0, 80)) * n("SMKQTTP").map(unit)
    o["quit_years"] = np.where(st == 1, yrs, np.nan)
    o["prior_cancer"] = (n("CANEV") == 1).astype(int)
    elig, died, ucod = n("elig"), n("died"), n("ucod")
    q = n("INTV_QRT").where(n("INTV_QRT").between(1, 4), 2.0)
    t0 = o["year"] + (q - 0.5) / 4.0
    td = n("dody") + (n("dodq") - 0.5) / 4.0
    o["months"] = np.where(died == 1, (td - t0) * 12, (2020.0 - t0) * 12)
    o["died"], o["ucod"], o["elig"] = died, ucod, elig
    h = n("P_HISPAN_I").where(n("P_HISPAN_I").notna(), n("P_HISPAN_P"))
    r = n("P_RACERPI2") if "P_RACERPI2" in d else n("P_RACEREC")
    nh = h == 12
    o["race4"] = np.select([h.between(1, 11), nh & (r == 1), nh & (r == 2), nh], [2, 0, 1, 3], default=-1)
    keep = (o["elig"] == 1) & o["age"].between(35, 84) & (o["prior_cancer"] == 0) & o["smoke_status"].notna() & o["bmi"].notna() & (o["year"] <= 2009)
    keep &= (o["months"] >= 120) | (o["died"] == 1)
    o["y"] = ((o["died"] == 1) & (o["ucod"] == 2) & (o["months"] <= 120)).astype(int)
    o["y_other"] = ((o["died"] == 1) & (o["ucod"] != 2) & (o["months"] <= 120)).astype(int)
    o["bmi"] = o["bmi"].clip(15, 55)
    o["quit_years"] = o["quit_years"].fillna(0).clip(0, 50)
    o["cigs_per_day"] = o["cigs_per_day"].fillna(0).clip(0, 60)
    return d.loc[keep].reset_index(drop=True), o.loc[keep].reset_index(drop=True)


def candidates(raw, labels_by_year):
    """Items with a stable label, present in >= 2 discovery and >= 2 replication years, passing the exclusions."""
    out = {}
    cols = [c for c in raw.columns if not EXCL_NAME.match(c)]
    for c in cols:
        labs = {y: labels_by_year.get(y, {}).get(c) for y in DISC + REPL if y in labels_by_year}
        labs = {y: l for y, l in labs.items() if l}
        if not labs:
            continue
        modal = pd.Series(list(labs.values())).mode()[0]
        if EXCL_LABEL.search(modal):
            continue
        years = [y for y, l in labs.items() if l == modal and raw.loc[raw["year"] == y, c].notna().any()]
        if sum(y in DISC for y in years) >= 2 and sum(y in REPL for y in years) >= 2:
            out[c] = (modal, years)
    return out


class Base:
    """Baseline design: age spline, sex, smoking, BMI, interview-year indicators."""

    def __init__(self, o_train):
        self.st = SplineTransformer(degree=3, n_knots=5, knots="quantile", extrapolation="constant", include_bias=False).fit(o_train[["age"]])
        self.years = sorted(o_train["year"].unique())

    def X(self, o):
        s = o["smoke_status"].to_numpy()
        cols = [self.st.transform(o[["age"]].to_numpy()), o[["male"]].to_numpy(),
                np.column_stack([(s == 1), (s == 2), o["cigs_per_day"] / 20.0, (s == 1) * o["quit_years"] / 10.0, o["bmi"] / 10.0]).astype(float)]
        yr = np.column_stack([(o["year"].to_numpy() == y).astype(float) for y in self.years[1:]]) if len(self.years) > 1 else np.zeros((len(o), 0))
        return np.column_stack(cols + [yr])


def blocks(v, ref_series=None, ref=None):
    """Dummy block for one item. Returns (matrix, names, reference); `ref` fixes the layout for replication."""
    x = v.copy()
    if ref is None:
        nun = x.nunique(dropna=True)
        if nun > 12:
            q = pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            edges = x.groupby(q).max().to_numpy()
            cat = pd.Series(np.searchsorted(edges[:-1], x.to_numpy(), side="left").astype(float), index=x.index).where(x.notna(), -1)
            kind = ("bin", edges[:-1].tolist())
        else:
            cat = x.fillna(-1)
            kind = ("cat", None)
        cnt = cat.value_counts()
        keep = [c for c, k in cnt.items() if k >= 50]
        if len(keep) < 2:
            return None
        refc = cnt.index[0]
        ref = {"kind": kind, "keep": keep, "ref": refc}
    else:
        kind = ref["kind"]
        if kind[0] == "bin":
            cat = pd.Series(np.searchsorted(np.array(kind[1]), x.to_numpy(), side="left").astype(float), index=x.index).where(x.notna(), -1)
        else:
            cat = x.fillna(-1)
    cat = cat.where(cat.isin(ref["keep"]), -99)
    levels = [c for c in ref["keep"] if c != ref["ref"]] + ([-99] if (cat == -99).any() else [])
    M = np.column_stack([(cat == c).to_numpy().astype(float) for c in levels]) if levels else np.zeros((len(x), 0))
    return M, levels, ref


def fit(X, y):
    m = LogisticRegression(C=C_REG, max_iter=500)
    m.fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return m, ll


def screen_one(name, label, years, raw, o, train_years):
    msk = o["year"].isin([y for y in years if y in train_years]).to_numpy()
    if msk.sum() < 5000:
        return None
    oo, xx = o.loc[msk].reset_index(drop=True), raw.loc[msk, name].reset_index(drop=True)
    b = blocks(xx)
    if b is None:
        return None
    M, levels, ref = b
    if M.shape[1] == 0:
        return None
    y = oo["y"].to_numpy()
    if y.sum() < 100:
        return None
    base = Base(oo)
    Xb = base.X(oo)
    _, ll0 = fit(Xb, y)
    m1, ll1 = fit(np.column_stack([Xb, M]), y)
    lr = max(2 * (ll1 - ll0), 0.0)
    df = M.shape[1]
    coefs = m1.coef_[0][Xb.shape[1]:]
    j = int(np.argmax(np.abs(coefs)))
    return {"item": name, "label": label, "n": int(msk.sum()), "events": int(y.sum()), "df": df, "lr": round(lr, 3), "p": float(chi2.sf(lr, df)),
            "main_level": float(levels[j]), "main_coef": round(float(coefs[j]), 4), "ref_level": float(ref["ref"])}


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def main(group=None):
    raw, o = load()
    tag = "all adults" if group is None else {0: "non-Hispanic White", 1: "non-Hispanic Black", 2: "Hispanic", 3: "other"}[group]
    if group is not None:
        m = (o["race4"] == group).to_numpy()
        raw, o = raw.loc[m].reset_index(drop=True), o.loc[m].reset_index(drop=True)
    labs = {y: read_labels(y) for y in DISC + REPL}
    cand = candidates(raw, labs)
    print(f"{tag}: {len(o):,} adults, {int(o['y'].sum())} cancer deaths, {len(cand)} candidate items", flush=True)
    rows = Parallel(n_jobs=4)(delayed(screen_one)(c, lab, yrs, raw, o, DISC) for c, (lab, yrs) in cand.items())
    d = pd.DataFrame([r for r in rows if r])
    d["q"] = bh(d["p"].to_numpy())
    hits = d[d["q"] < 0.05].sort_values("p")
    sfx = "" if group is None else f"_race{group}"
    d.to_csv(os.path.join(RES, f"discovery_screen_all{sfx}.csv"), index=False)
    hits.to_csv(os.path.join(RES, f"discovery_screen_hits{sfx}.csv"), index=False)       # frozen before replication is touched
    print(f"discovery: {len(d)} items screened, {len(hits)} with q < 0.05 (hit list frozen)", flush=True)
    res = {"group": tag, "n": int(len(o)), "candidates": len(cand), "screened": int(len(d)), "discovery_hits": int(len(hits)), "replicated": [], "items": {}}

    # ---- replication, specificity, usefulness
    alpha = 0.05 / max(len(hits), 1)
    trn = o["year"].isin(DISC).to_numpy()
    for h in hits.itertuples():
        lab, yrs = cand[h.item]
        mt = o["year"].isin([y for y in yrs if y in DISC]).to_numpy()
        me = o["year"].isin([y for y in yrs if y in REPL]).to_numpy()
        otr, xtr = o.loc[mt].reset_index(drop=True), raw.loc[mt, h.item].reset_index(drop=True)
        ote, xte = o.loc[me].reset_index(drop=True), raw.loc[me, h.item].reset_index(drop=True)
        M0, levels, ref = blocks(xtr)
        Mt, _, _ = blocks(xte, ref=ref)
        if Mt.shape[1] != M0.shape[1]:
            continue
        # replication test: refit baseline and item in the replication set itself
        yte = ote["y"].to_numpy()
        base_e = Base(ote)
        Xe = base_e.X(ote)
        _, l0 = fit(Xe, yte)
        me1, l1 = fit(np.column_stack([Xe, Mt]), yte)
        lr = max(2 * (l1 - l0), 0)
        p_rep = float(chi2.sf(lr, Mt.shape[1]))
        coefs_e = me1.coef_[0][Xe.shape[1]:]
        lev_idx = levels.index(h.main_level) if h.main_level in levels else None
        same_dir = bool(lev_idx is not None and np.sign(coefs_e[lev_idx]) == np.sign(h.main_coef))
        rep = bool(p_rep < alpha and same_dir)
        info = {"label": lab, "years": yrs, "p_discovery": h.p, "p_replication": p_rep, "same_direction": same_dir, "replicated": rep}
        if rep:
            # AUC gains out of sample: train on discovery years, score on replication years
            base_t = Base(otr)
            Xt, Xs = base_t.X(otr), base_t.X(ote)
            auc = {}
            preds = {}
            for outcome in ("y", "y_other"):
                yt = otr[outcome].to_numpy()
                mb, _ = fit(Xt, yt)
                mv, _ = fit(np.column_stack([Xt, M0]), yt)
                pb, pv = mb.predict_proba(Xs)[:, 1], mv.predict_proba(np.column_stack([Xs, Mt]))[:, 1]
                ys = ote[outcome].to_numpy()
                preds[outcome] = (ys, pb, pv)
                auc[outcome] = float(roc_auc_score(ys, pv) - roc_auc_score(ys, pb))
            ys, pb, pv = preds["y"]
            g, ci = [], []
            for _ in range(300):
                i = RNG.integers(0, len(ys), len(ys))
                if ys[i].min() == ys[i].max():
                    continue
                g.append(roc_auc_score(ys[i], pv[i]) - roc_auc_score(ys[i], pb[i]))
            ci = [float(x) for x in np.percentile(g, [2.5, 97.5])]
            inv = ote["age"].between(40, 74).to_numpy()
            yi, bi, vi = ys[inv], pb[inv], pv[inv]
            k = int(0.2 * len(yi))

            def cov(yy, sc):
                return float(yy[np.argsort(-sc, kind="stable")[:k]].sum() / max(yy.sum(), 1))
            dd = []
            for _ in range(300):
                i = RNG.integers(0, len(yi), len(yi))
                dd.append(cov(yi[i], vi[i]) - cov(yi[i], bi[i]))
            cci = [float(x) for x in np.percentile(dd, [2.5, 97.5])]
            specific = bool(auc["y"] >= 0.5 * auc["y_other"]) if auc["y_other"] > 0 else True
            useful = bool(auc["y"] >= 0.005 and ci[0] > 0 and cov(yi, vi) - cov(yi, bi) >= 0.01 and cci[0] > 0)
            info.update({"auc_gain_cancer": auc["y"], "auc_gain_cancer_ci": ci, "auc_gain_other": auc["y_other"], "specific": specific,
                         "coverage_gain_20pct": cov(yi, vi) - cov(yi, bi), "coverage_gain_ci": cci, "discovery": bool(specific and useful)})
            res["replicated"].append(h.item)
        res["items"][h.item] = info
        print(f"  {h.item:<12} {lab[:60]:<60} disc p={h.p:.1e}  repl p={p_rep:.1e} dir={same_dir} rep={rep}" + (f"  AUC cancer {info.get('auc_gain_cancer', 0):+.4f} other {info.get('auc_gain_other', 0):+.4f} disc={info.get('discovery')}" if rep else ""), flush=True)
    res["discoveries"] = [k for k, v in res["items"].items() if v.get("discovery")]
    print(f"\nreplicated: {len(res['replicated'])}   discoveries (replicated, cancer-specific, useful): {len(res['discoveries'])}")
    json.dump(res, open(os.path.join(RES, f"discovery_screen_result{sfx}.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main(None if len(sys.argv) < 2 or sys.argv[1] != "equity" else int(sys.argv[2]))
