"""
Hypothesis-free discovery screen: does ANY free NHIS questionnaire item add to cancer-death risk beyond age, sex,
smoking and BMI, replicably and specifically for cancer?  Rules: docs/DISCOVERY_SCREEN_PREREG.md (committed first).

Stages: (1) discovery 1997-2002, likelihood-ratio test of each item on top of the baseline, false discovery rate
q < 0.05; (2) the hit list is written to disk, then and only then (3) replication 2005-2009 with Bonferroni over
the hits and the same direction; (4) specificity (cancer gain at least half the other-cause gain); (5) usefulness
(AUC gain >= 0.005 and coverage gain at 20% invited >= 1 point, intervals above zero).

One recorded departure from the pre-registration, decided before any result: interview-year indicators are added to
the baseline (a variable asked only in some years would otherwise be confounded with period trends in cancer death).

Run:  python experiments/discovery_screen.py [equity 1|2|3] discover|replicate
      'discover' writes the hit list and its SHA-256 and stops. Commit them. 'replicate' refuses to run if the hit list
      on disk no longer matches the recorded hash.
"""

import hashlib
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
MAX_ITER = 100   # Newton iterations; the exact solver below converges in a handful
MIN_PEOPLE, MIN_EVENTS = 5000, 100   # stability floors written into the code before any result; see addendum 2
N_JOBS = 8
GROUPS = {0: "non-Hispanic White", 1: "non-Hispanic Black", 2: "Hispanic", 3: "other"}

# Items that are the baseline itself, survey machinery, or cancer-related (leakage / reverse causation).
EXCL_NAME = re.compile(r"^(SRVY_YR|WTFA_SA|STRATUM|PSU|INTV_QRT|ASTATFLG|CSTATFLG|AGE_P|SEX|BMI|AHEIGHT|AWEIGHTP|SMK|CIGSDAY|CIGDAMO|CIGQTYR|SMKQT|SMKEV|SMKNOW|SMKSTAT|"
                       r"CAN|CN|PAP|MAM|PSA|CRE|CBE|SKNX|HFO|RHFO|MDHFO|SIG|COL|LUN|BREAST|HYST|HPV|MD(REC|HFOB)|RMAM|RPAP|RPSA|RCRE|DOD|elig|died|ucod|dodq|dody|pid|k$|year|REGION$|FMX|HHX|FPX|PX$|"
                       r"P_|RACE|RACER|MRAC|RCDT|RC_S|RCIMP|HISPAN|ORIGIN|ORIGIMPT|HISPIMPT|RACRE)")
EXCL_LABEL = re.compile(r"\brace\b|racial|hispanic|latino|ethnicity|cancer|tumou?r|chemo|radiation|screen|mammogra|pap smear|psa|colonosc|sigmoid|stool|biops|malignan|mortality|age first|year of death|weight|height|body mass|smok|cigar", re.I)


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
    # Race and ethnicity sit under a different name in different years, and the old test ("is the column in the pooled
    # frame") picked one name for every year, which left 1997-2002 with no race at all. Take the first variable that has
    # a value for each person, in an order that leaves 2005-2009 exactly as before. Codes agree across all of them:
    # Hispanic subgroups are 0-11 and 12 is not Hispanic; for race 1 is White and 2 is Black.
    def first(*names):
        out = pd.Series(np.nan, index=d.index)
        for nm in names:
            if nm in d:
                out = out.where(out.notna(), pd.to_numeric(d[nm], errors="coerce"))
        return out
    h = first("P_HISPAN_I", "P_HISPAN_P", "HISPANCR", "HISPAN_I", "HISPAN_P")
    r = first("P_RACERPI2", "P_RACEREC", "RACERP_I", "RACER_P", "RACE")
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


def load_labels():
    """Item labels from the NCHS SAS programs, cached next to the data so the run is reproducible offline."""
    out = {}
    for y in DISC + REPL:
        f = os.path.join(FULL, f"labels_{y}.json")
        if os.path.exists(f):
            out[y] = json.load(open(f))
        else:
            out[y] = read_labels(y)
            json.dump(out[y], open(f, "w"))
    return out


def candidates(raw, labels_by_year):
    """Items with a stable label, present in >= 2 discovery and >= 2 replication years, passing the exclusions.
    Returns (items, counts, excluded) so every stage can be reported, not only the survivors."""
    out, excluded = {}, []
    counts = {"columns": 0, "excluded_by_name": 0, "excluded_by_label": 0, "no_label": 0, "label_wording_differs": 0, "too_few_years": 0}
    avail = raw.notna().groupby(raw["year"]).any()
    for c in raw.columns:
        counts["columns"] += 1
        if EXCL_NAME.match(c):
            counts["excluded_by_name"] += 1
            excluded.append((c, "name", ""))
            continue
        labs = {y: labels_by_year.get(y, {}).get(c) for y in DISC + REPL if y in labels_by_year}
        labs = {y: l for y, l in labs.items() if l}
        if not labs:
            counts["no_label"] += 1
            excluded.append((c, "no label in the SAS programs", ""))
            continue
        modal = pd.Series(list(labs.values())).mode()[0]
        if EXCL_LABEL.search(modal):
            counts["excluded_by_label"] += 1
            excluded.append((c, "label", modal))
            continue
        by_name = [y for y in labs if y in avail.index and bool(avail.at[y, c])]
        years = [y for y in by_name if labs[y] == modal]
        enough = lambda ys: sum(y in DISC for y in ys) >= 2 and sum(y in REPL for y in ys) >= 2  # noqa: E731
        if enough(years):
            out[c] = (modal, years)
        elif enough(by_name):
            # Asked in enough years under this name, but the label text changed (for example "had" became "have").
            # The registered rule is the same label, so it is not screened here; it is listed, not silently lost.
            counts["label_wording_differs"] += 1
            excluded.append((c, "label wording differs between years (asked in enough years by name)", modal))
        else:
            counts["too_few_years"] += 1
            excluded.append((c, "asked in fewer than 2 discovery or 2 replication years", modal))
    counts["candidates"] = len(out)
    return out, counts, excluded


class Base:
    """Baseline design: age spline, sex, smoking, BMI, interview-year indicators."""

    def __init__(self, o_train):
        self.st = SplineTransformer(degree=3, n_knots=5, knots="quantile", extrapolation="constant", include_bias=False).fit(o_train[["age"]])
        self.years = sorted(o_train["year"].unique())

    def X(self, o):
        s = o["smoke_status"].to_numpy()
        cols = [self.st.transform(o[["age"]]), o[["male"]].to_numpy(),
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
        ref = {"kind": kind, "keep": keep, "ref": refc, "has_other": bool((~cat.isin(keep)).any())}
    else:
        kind = ref["kind"]
        if kind[0] == "bin":
            cat = pd.Series(np.searchsorted(np.array(kind[1]), x.to_numpy(), side="left").astype(float), index=x.index).where(x.notna(), -1)
        else:
            cat = x.fillna(-1)
    cat = cat.where(cat.isin(ref["keep"]), -99)
    levels = [c for c in ref["keep"] if c != ref["ref"]] + ([-99] if ref["has_other"] else [])
    M = np.column_stack([(cat == c).to_numpy().astype(float) for c in levels]) if levels else np.zeros((len(x), 0))
    return M, levels, ref


def fit(X, y):
    # Exact second-order solver, tight tolerance. The default lbfgs stopped before converging, so some fits with an
    # extra item scored below the baseline, the likelihood-ratio statistic was clipped to zero (14% of null tests)
    # and null p-values averaged 0.61 instead of 0.50. Checked on shuffled items: see addendum 2.
    m = LogisticRegression(C=C_REG, solver="newton-cholesky", tol=1e-10, max_iter=MAX_ITER)
    m.fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return m, ll


def screen_one(name, label, years, col, o, train_years):
    """One item added to the baseline in the discovery years. Returns a result row, or a row with a 'skipped' reason."""
    def skip(why):
        return {"item": name, "label": label, "skipped": why}
    msk = o["year"].isin([y for y in years if y in train_years]).to_numpy()
    if msk.sum() < MIN_PEOPLE:
        return skip(f"fewer than {MIN_PEOPLE} people in the discovery years")
    oo, xx = o.loc[msk].reset_index(drop=True), col[msk].reset_index(drop=True)
    b = blocks(xx)
    if b is None:
        return skip("fewer than 2 categories with 50 or more people")
    M, levels, ref = b
    if M.shape[1] == 0:
        return skip("no usable category")
    y = oo["y"].to_numpy()
    if y.sum() < MIN_EVENTS:
        return skip(f"fewer than {MIN_EVENTS} cancer deaths in the discovery years")
    base = Base(oo)
    Xb = base.X(oo)
    m0, ll0 = fit(Xb, y)
    m1, ll1 = fit(np.column_stack([Xb, M]), y)
    lr = max(2 * (ll1 - ll0), 0.0)
    df = M.shape[1]
    coefs = m1.coef_[0][Xb.shape[1]:]
    j = int(np.argmax(np.abs(coefs)))
    return {"item": name, "label": label, "n": int(msk.sum()), "events": int(y.sum()), "df": df, "lr": round(lr, 3), "p": float(chi2.sf(lr, df)),
            "main_level": float(levels[j]), "main_coef": round(float(coefs[j]), 4), "ref_level": float(ref["ref"]),
            "converged": bool(max(int(m0.n_iter_[0]), int(m1.n_iter_[0])) < MAX_ITER)}


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def setup(group):
    raw, o = load()
    tag = "all adults" if group is None else GROUPS[group]
    if group is not None:
        m = (o["race4"] == group).to_numpy()
        raw, o = raw.loc[m].reset_index(drop=True), o.loc[m].reset_index(drop=True)
    cand, counts, excluded = candidates(raw, load_labels())
    return raw, o, tag, cand, counts, excluded


def paths(group):
    sfx = "" if group is None else f"_race{group}"
    j = lambda n: os.path.join(RES, n.format(sfx=sfx))  # noqa: E731
    return {"all": j("discovery_screen_all{sfx}.csv"), "hits": j("discovery_screen_hits{sfx}.csv"), "skipped": j("discovery_screen_skipped{sfx}.csv"),
            "excluded": j("discovery_screen_excluded{sfx}.csv"), "stage1": j("discovery_screen_stage1{sfx}.json"), "result": j("discovery_screen_result{sfx}.json")}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def discover(group=None):
    """Stages 1-2: likelihood-ratio screen in the discovery years, FDR, and the hit list written to disk and hashed. Stops there."""
    raw, o, tag, cand, counts, excluded = setup(group)
    P = paths(group)
    print(f"{tag}: {len(o):,} adults, {int(o['y'].sum())} cancer deaths; columns {counts['columns']}, candidates {counts['candidates']}", flush=True)
    rows = Parallel(n_jobs=N_JOBS)(delayed(screen_one)(c, lab, yrs, raw[c], o, DISC) for c, (lab, yrs) in cand.items())
    skipped = [r for r in rows if r and "skipped" in r]
    done = [r for r in rows if r and "skipped" not in r]
    pd.DataFrame(excluded, columns=["item", "reason", "label"]).to_csv(P["excluded"], index=False)
    pd.DataFrame(skipped, columns=["item", "label", "skipped"]).to_csv(P["skipped"], index=False)
    stage = {"group": tag, "adults": int(len(o)), "cancer_deaths": int(o["y"].sum()), **counts, "skipped_by_size_floors": len(skipped),
             "skipped_reasons": pd.Series([s["skipped"] for s in skipped], dtype=str).value_counts().to_dict(), "screened": len(done)}
    if not done:
        stage.update(discovery_hits=0, note="no item could be screened in this group: every candidate fell below the size floors")
        json.dump(stage, open(P["stage1"], "w"), indent=2)
        print("nothing could be screened:", stage["skipped_reasons"], flush=True)
        return
    d = pd.DataFrame(done)
    d["q"] = bh(d["p"].to_numpy())
    hits = d[d["q"] < 0.05].sort_values("p")
    d.to_csv(P["all"], index=False)
    hits.to_csv(P["hits"], index=False)        # frozen: commit this file and the hash below before replicating
    stage.update(discovery_hits=int(len(hits)), not_converged=int((~d["converged"]).sum()), hits_file=os.path.basename(P["hits"]), hits_sha256=sha256(P["hits"]))
    json.dump(stage, open(P["stage1"], "w"), indent=2)
    print(f"discovery: {len(d)} items screened, {len(hits)} with q < 0.05, {stage['not_converged']} fits hit the iteration limit, {len(skipped)} skipped by size floors", flush=True)
    print(f"hit list sha256 {stage['hits_sha256']}", flush=True)


def replicate(group=None):
    """Stages 3-5 on the frozen hit list: replication, specificity, usefulness. Refuses to run if the hit list changed."""
    raw, o, tag, cand, counts, excluded = setup(group)
    P = paths(group)
    stage = json.load(open(P["stage1"]))
    if stage.get("discovery_hits", 0) == 0:
        json.dump({**stage, "replicated": [], "discoveries": [], "items": {}}, open(P["result"], "w"), indent=2, default=float)
        print("no discovery hits, nothing to replicate")
        return
    if sha256(P["hits"]) != stage["hits_sha256"]:
        raise SystemExit("the hit list on disk does not match the hash recorded at discovery; refusing to replicate")
    hits = pd.read_csv(P["hits"])
    equity = group is not None
    auc_bar = 0.01 if equity else 0.005          # pre-registered: 0.005 for all adults, 0.01 for the equity screen
    res = {**stage, "equity_screen": equity, "auc_bar": auc_bar, "replicated": [], "items": {}}

    alpha = 0.05 / max(len(hits), 1)
    for h in hits.itertuples():
        assert h.item in cand, f"{h.item} in the frozen hit list is not a candidate now; the run is not deterministic"
        lab, yrs = cand[h.item]
        info = {"label": lab, "years": yrs, "p_discovery": float(h.p), "q_discovery": float(h.q), "replicated": False}
        mt = o["year"].isin([y for y in yrs if y in DISC]).to_numpy()
        me = o["year"].isin([y for y in yrs if y in REPL]).to_numpy()
        otr, xtr = o.loc[mt].reset_index(drop=True), raw.loc[mt, h.item].reset_index(drop=True)
        ote, xte = o.loc[me].reset_index(drop=True), raw.loc[me, h.item].reset_index(drop=True)
        M0, levels, ref = blocks(xtr)
        Mt, _, _ = blocks(xte, ref=ref)
        # Same columns as in discovery. A category nobody in the replication years falls into cannot be tested, so the
        # test counts only columns that vary there; a hit is never dropped without a recorded reason.
        live = np.flatnonzero(Mt.std(axis=0) > 0)
        if len(live) == 0:
            info["reason"] = "no variation in the item in the replication years"
            res["items"][h.item] = info
            print(f"  {h.item:<12} {lab[:60]:<60} NOT REPLICATED: {info['reason']}", flush=True)
            continue
        yte = ote["y"].to_numpy()
        Xe = Base(ote).X(ote)
        _, l0 = fit(Xe, yte)
        me1, l1 = fit(np.column_stack([Xe, Mt[:, live]]), yte)
        lr = max(2 * (l1 - l0), 0)
        p_rep = float(chi2.sf(lr, len(live)))
        coefs_e = np.zeros(Mt.shape[1])
        coefs_e[live] = me1.coef_[0][Xe.shape[1]:]
        lev_idx = levels.index(h.main_level) if h.main_level in levels else None
        same_dir = bool(lev_idx is not None and coefs_e[lev_idx] != 0 and np.sign(coefs_e[lev_idx]) == np.sign(h.main_coef))
        rep = bool(p_rep < alpha and same_dir)
        info.update({"p_replication": p_rep, "same_direction": same_dir, "replicated": rep, "replication_df": int(len(live)), "discovery_df": int(M0.shape[1])})
        if rep:
            # AUC gains out of sample: train on discovery years, score on replication years
            base_t = Base(otr)
            Xt, Xs = base_t.X(otr), base_t.X(ote)
            auc, preds = {}, {}
            for outcome in ("y", "y_other"):
                yt = otr[outcome].to_numpy()
                mb, _ = fit(Xt, yt)
                mv, _ = fit(np.column_stack([Xt, M0]), yt)
                pb, pv = mb.predict_proba(Xs)[:, 1], mv.predict_proba(np.column_stack([Xs, Mt]))[:, 1]
                ys = ote[outcome].to_numpy()
                preds[outcome] = (ys, pb, pv)
                auc[outcome] = float(roc_auc_score(ys, pv) - roc_auc_score(ys, pb))
            ys, pb, pv = preds["y"]
            g = []
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
            auc_ok = bool(auc["y"] >= auc_bar and ci[0] > 0)
            cov_ok = bool(cov(yi, vi) - cov(yi, bi) >= 0.01 and cci[0] > 0)
            useful = auc_ok if equity else (auc_ok and cov_ok)   # equity bar as registered: AUC gain only; coverage is reported
            info.update({"auc_gain_cancer": auc["y"], "auc_gain_cancer_ci": ci, "auc_gain_other": auc["y_other"], "specific": specific,
                         "coverage_gain_20pct": cov(yi, vi) - cov(yi, bi), "coverage_gain_ci": cci, "auc_bar_met": auc_ok, "coverage_bar_met": cov_ok,
                         "discovery": bool(specific and useful)})
            res["replicated"].append(h.item)
        res["items"][h.item] = info
        print(f"  {h.item:<12} {lab[:60]:<60} disc p={h.p:.1e}  repl p={p_rep:.1e} dir={same_dir} rep={rep}" + (f"  AUC cancer {info.get('auc_gain_cancer', 0):+.4f} other {info.get('auc_gain_other', 0):+.4f} disc={info.get('discovery')}" if rep else ""), flush=True)
    res["discoveries"] = [k for k, v in res["items"].items() if v.get("discovery")]
    res["stages"] = {"candidates": stage["candidates"], "screened": stage["screened"], "discovery_hits": int(len(hits)), "replicated": len(res["replicated"]),
                     "cancer_specific": sum(bool(v.get("specific")) for v in res["items"].values()), "discoveries": len(res["discoveries"])}
    print(f"\n{tag} stages: {res['stages']}")
    json.dump(res, open(P["result"], "w"), indent=2, default=float)


if __name__ == "__main__":
    a = sys.argv[1:]
    grp = None
    if a and a[0] == "equity":
        grp, a = int(a[1]), a[2:]
        assert grp in (1, 2, 3), "equity groups are 1 (Black), 2 (Hispanic), 3 (other)"
    if a == ["discover"]:
        discover(grp)
    elif a == ["replicate"]:
        replicate(grp)
    else:
        raise SystemExit("usage: discovery_screen.py [equity 1|2|3] discover|replicate")
