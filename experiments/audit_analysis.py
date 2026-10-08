"""
Step 3: the audit statistics (A1, A2), cue-versus-adjudicated agreement, and the controlled
re-analysis (A3, A4) on the cross-sectional NHANES cancer-status task.
Bars are in docs/AUDIT_PREREG.md.

Run:  python experiments/audit_analysis.py
Needs data/audit/*.csv and data/nhanes_xs_8c.csv.gz
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_adjudicated as adj  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "audit_result.json")
RNG = np.random.default_rng(20261007)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0, c - h)), 3), round(float(min(1, c + h)), 3)]


# ------------------------------------------------------------------ Part 1
def part1():
    df = adj.table()
    auto = pd.read_csv(os.path.join(ROOT, "data", "audit", "auto_codes.csv"))
    df = df.merge(auto, on="pmid", suffixes=("", "_auto"))
    n = len(df)
    res = {"records_found": 79, "included": n, "excluded": 79 - n, "open_access_full_text_in_included": int((~df["abstract_only"]).sum()),
           "years": [int(df["year"].min()), int(df["year"].max())]}
    res["tasks"] = df["task"].value_counts().to_dict()
    res["sites"] = df["site"].value_counts().to_dict()
    ext = int(df["ext"].sum())
    res["A1"] = {"external_or_heldout": ext, "heldout_cycle": int(df["heldout"].sum()), "no_external": n - ext,
                 "share_no_external": round((n - ext) / n, 3), "ci": wilson(n - ext, n),
                 "validation_types": df["validation"].value_counts().to_dict(),
                 "full_text_only": {"n": int((~df["abstract_only"]).sum()),
                                    "share_no_external": round(float(((~df["ext"]) & (~df["abstract_only"])).sum() / (~df["abstract_only"]).sum()), 3)}}
    res["A1"]["supported"] = bool((n - ext) / n >= 0.5)
    age = int(df["agesex"].sum())
    comp = int((df["agesex"] | df["comparator"]).sum())
    res["A2"] = {"age_or_agesex_baseline": age, "share": round(age / n, 3), "ci": wilson(age, n),
                 "any_comparator_logistic_cox_or_clinical": comp, "comparator_share": round(comp / n, 3), "supported": bool(age / n <= 0.20)}
    wk = df["weights"].dropna()
    res["weights_in_fitting"] = {"assessable": int(len(wk)), "used": int(wk.sum())}
    res["calibration_reported"] = {"yes": int(df["calib"].fillna(False).sum()), "assessable": int(df["calib"].notna().sum())}
    rs = df["resample"].dropna()
    res["resampling_used"] = {"yes": int(rs.sum()), "assessable": int(len(rs))}
    gd = df[df["resample"] == True]  # noqa: E712
    res["leakage_guard_among_resampling_papers"] = {"papers": int(len(gd)), "guard_reported": int((gd["guard"] == True).sum())}  # noqa: E712
    aucs = df["auc"].dropna()
    res["auc_all"] = {"n": int(len(aucs)), "median": round(float(aucs.median()), 3), "min": float(aucs.min()), "max": float(aucs.max()),
                      "share_ge_0.80": round(float((aucs >= 0.80).mean()), 3), "share_ge_0.90": round(float((aucs >= 0.90).mean()), 3)}
    xs = df[(df["task"].isin(["xs", "comorb"])) & df["auc"].notna()]
    res["auc_xs"] = {"n": int(len(xs)), "median": round(float(xs["auc"].median()), 3), "iqr": [round(float(xs["auc"].quantile(.25)), 3), round(float(xs["auc"].quantile(.75)), 3)]}
    # agreement between the cue rules and the adjudicated codes
    kappa = {}
    for name, cue, code in (("external or held-out cycle", df["external_validation"] > 0, df["ext"]),
                            ("random split or CV described", df["random_split"] > 0, df["validation"].isin(["random", "cv", "split_unstated"])),
                            ("age-only baseline", df["baseline_agesex"] > 0, df["agesex"]),
                            ("survey weights in text", df["survey_weights"] > 0, df["weights"].fillna(False).astype(bool)),
                            ("calibration", df["calibration"] > 0, df["calib"].fillna(False).astype(bool)),
                            ("resampling", df["resampling"] > 0, df["resample"].fillna(False).astype(bool))):
        k = cohen_kappa_score(code.astype(int), cue.astype(int))
        agree = float((code.astype(int) == cue.astype(int)).mean())
        kappa[name] = {"kappa": round(float(k), 2), "raw_agreement": round(agree, 2)}
    res["cue_vs_adjudicated"] = kappa
    res["table"] = df[adj.COLS + ["pmid", "pmcid", "year"]].astype(object).where(pd.notna(df[adj.COLS + ["pmid", "pmcid", "year"]]), None).to_dict("records")
    return res, df


# ------------------------------------------------------------------ Part 2
SITE_CODES = {"breast": {14}, "prostate": {30}, "colorectal": {16, 31}, "uterine": {38}, "bladder": {10}, "digestive": {16, 31, 17, 35, 22, 29, 18}}


def ridge(C=0.05):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0)


def boot_gain(y, a, b, n=500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    g = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        g.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return [round(float(x), 4) for x in np.percentile(g, [2.5, 97.5])]


def part2():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_xs_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d["male"] = d["gender"].astype(float)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan)) if "SMQ020" in d else np.nan
    meta = {"SEQN", "gender", "cycle", "cancer", "MCQ230A", "year0", "male", "smoked"}
    allc = [c for c in d.columns if c not in meta and d[c].dtype != object]
    nof = [c for c in allc if not c.endswith("__missing")]
    flags = [c for c in allc if c.endswith("__missing")]
    sets = {"age + sex": ["age", "male"], "age + sex + smoking + BMI": ["age", "male", "smoked", "BMXBMI"],
            "whole picture, no missing flags": ["male"] + nof + ["smoked"], "whole picture + missing flags": ["male"] + allc + ["smoked"]}
    sets = {k: list(dict.fromkeys(v)) for k, v in sets.items()}
    out = {"n": int(len(d)), "cancer_history": int(d["cancer"].sum()), "variables": {k: len(v) for k, v in sets.items()}, "tasks": {}}

    def task_frame(site):
        if site == "any":
            return d
        codes = SITE_CODES[site]
        keep = (d["cancer"] == 0) | d["MCQ230A"].isin(codes)
        t = d[keep].copy()
        t["cancer"] = t["MCQ230A"].isin(codes).astype(int)
        if site in ("breast", "uterine"):
            t = t[t["male"] == 0]
        if site == "prostate":
            t = t[t["male"] == 1]
        return t

    def run(t, label):
        res = {"n": int(len(t)), "cases": int(t["cancer"].sum())}
        # random split, as most papers do
        tr, te = train_test_split(t, test_size=0.3, random_state=1, stratify=t["cancer"])
        # cycle-held-out split
        tre, tee = t[t["year0"] <= 2005], t[t["year0"] >= 2007]
        for sp, (a, b) in {"random": (tr, te), "later_cycles": (tre, tee)}.items():
            ya, yb = a["cancer"].to_numpy(), b["cancer"].to_numpy()
            if yb.sum() < 15 or ya.sum() < 15:
                continue
            row = {"test_n": int(len(b)), "test_cases": int(yb.sum())}
            preds = {}
            for name, cols in sets.items():
                if name == "age + sex":
                    model = LogisticRegression(max_iter=2000)
                    X = lambda z: np.column_stack([z["age"], (z["age"] / 10) ** 2, z["male"]])  # noqa: E731
                    model.fit(X(a), ya)
                    p = model.predict_proba(X(b))[:, 1]
                elif len(cols) <= 4:
                    model = ridge(1.0).fit(a[cols], ya)
                    p = model.predict_proba(b[cols])[:, 1]
                else:
                    model = hgb().fit(a[cols], ya)
                    p = model.predict_proba(b[cols])[:, 1]
                preds[name] = p
                row[name] = round(float(roc_auc_score(yb, p)), 4)
            base = preds["age + sex"]
            row["gain_whole_picture_flags"] = round(row["whole picture + missing flags"] - row["age + sex"], 4)
            row["gain_ci_whole_picture_flags"] = boot_gain(yb, base, preds["whole picture + missing flags"])
            res[sp] = row
        out["tasks"][label] = res
        r_ = res
        print(f"{label:<28} n={r_['n']:>6} cases={r_['cases']:>5} | " + " | ".join(
            f"{sp}: age+sex {r_[sp]['age + sex']:.3f} -> whole(+flags) {r_[sp]['whole picture + missing flags']:.3f} (no flags {r_[sp]['whole picture, no missing flags']:.3f})"
            for sp in ("random", "later_cycles") if sp in r_), flush=True)

    d20 = d[d["age"] >= 20]
    d40 = d[d["age"] >= 40]
    out["tasks"] = {}
    for lab, frame in (("any cancer, 20+", d20), ("any cancer, 40+", d40)):
        d_backup = d
        d = frame
        run(task_frame("any"), lab)
        d = d_backup
    for site in ("breast", "prostate", "colorectal", "digestive", "uterine", "bladder"):
        t = task_frame_site(d40, site) if False else None
    # site tasks on adults 40+, as most published work restricts to older adults
    d_full = d
    d = d40
    for site in ("breast", "prostate", "colorectal", "digestive", "uterine", "bladder"):
        t = task_frame(site)
        if t["cancer"].sum() >= 60:
            run(t, f"{site}, 40+")
    d = d_full

    # age-only AUC by site for the per-paper comparison: 5-fold cross-validated, adults 40+ and 20+
    ageonly = {}
    for lab, frame in (("40+", d40), ("20+", d20)):
        d = frame
        for site in ["any"] + list(SITE_CODES):
            t = task_frame(site)
            if t["cancer"].sum() < 60:
                continue
            X = np.column_stack([t["age"], (t["age"] / 10) ** 2, t["male"] if site not in ("breast", "uterine", "prostate") else np.zeros(len(t))])
            p = cross_val_predict(LogisticRegression(max_iter=2000), X, t["cancer"], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            ageonly[f"{site}, {lab}"] = {"auc": round(float(roc_auc_score(t["cancer"], p)), 4), "cases": int(t["cancer"].sum()), "n": int(len(t))}
    d = d_full
    out["age_only_by_site"] = ageonly

    # leakage mechanism: survey cycle from missing flags alone
    tt = d[d["year0"].isin([1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013])]
    era = (tt["year0"] >= 2007).astype(int)
    pe = cross_val_predict(LogisticRegression(C=0.5, max_iter=3000), tt[flags].fillna(0), era, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    out["era_from_flags_auc"] = round(float(roc_auc_score(era, pe)), 4)
    out["n_flags"] = len(flags)

    # the classic pitfall: oversample before the split
    t = d40
    y = t["cancer"].to_numpy()
    cols = sets["whole picture, no missing flags"]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rep = RNG.choice(pos, len(neg), replace=True)
    idx = np.concatenate([neg, rep])
    big = t.iloc[idx].reset_index(drop=True)
    tr_, te_ = train_test_split(big, test_size=0.3, random_state=1, stratify=big["cancer"])
    p_leak = hgb().fit(tr_[cols], tr_["cancer"]).predict_proba(te_[cols])[:, 1]
    tr2, te2 = train_test_split(t, test_size=0.3, random_state=1, stratify=t["cancer"])
    pos2, neg2 = np.where(tr2["cancer"].to_numpy() == 1)[0], np.where(tr2["cancer"].to_numpy() == 0)[0]
    idx2 = np.concatenate([neg2, RNG.choice(pos2, len(neg2), replace=True)])
    trb = tr2.iloc[idx2]
    p_ok = hgb().fit(trb[cols], trb["cancer"]).predict_proba(te2[cols])[:, 1]
    out["oversample_before_split"] = {"auc_oversample_then_split": round(float(roc_auc_score(te_["cancer"], p_leak)), 4),
                                      "auc_split_then_oversample_train_only": round(float(roc_auc_score(te2["cancer"], p_ok)), 4),
                                      "test_duplicates_of_training_cases": "cases duplicated 1:1 before the split put copies of the same person in both sets"}
    print(f"\noversample-then-split AUC {out['oversample_before_split']['auc_oversample_then_split']:.3f} vs correct {out['oversample_before_split']['auc_split_then_oversample_train_only']:.3f}")
    print(f"survey cycle predicted from {len(flags)} missing flags alone: AUC {out['era_from_flags_auc']:.3f}")
    return out


def main():
    p1, df = part1()
    print(f"Part 1: {p1['included']} included of {p1['records_found']}")
    print(f"  A1: {p1['A1']['no_external']}/{p1['included']} with no external cohort and no held-out cycle ({p1['A1']['share_no_external']:.0%}, CI {p1['A1']['ci']}) -> {'supported' if p1['A1']['supported'] else 'NOT supported'}")
    print(f"  A2: {p1['A2']['age_or_agesex_baseline']}/{p1['included']} report an age-only baseline ({p1['A2']['share']:.0%}); any comparator {p1['A2']['any_comparator_logistic_cox_or_clinical']} -> {'supported' if p1['A2']['supported'] else 'NOT supported'}")
    print(f"  AUC reported: median {p1['auc_all']['median']} (min {p1['auc_all']['min']}, max {p1['auc_all']['max']}); cross-sectional median {p1['auc_xs']['median']}")
    for k, v in p1["cue_vs_adjudicated"].items():
        print(f"  agreement {k:<30} kappa {v['kappa']:>5}  raw {v['raw_agreement']}")
    print("\nPart 2: controlled re-analysis on cross-sectional NHANES cancer status")
    p2 = part2()

    # A3: per-paper gap against the age-only AUC for its site
    xs = df[(df["task"].isin(["xs"])) & df["auc"].notna()]
    gaps = {"40+": [], "20+": []}
    rows = []
    for _, r in xs.iterrows():
        site = r["site"]
        g = {}
        for age in ("40+", "20+"):
            ao = p2["age_only_by_site"].get(f"{site}, {age}")
            if ao:
                g[age] = round(float(r["auc"]) - ao["auc"], 3)
                gaps[age].append(g[age])
        rows.append({"pmid": r["pmid"], "site": site, "paper_auc": r["auc"], "gap": g})
    a3 = {"papers": len(rows), "median_gap_vs_age_only_40plus": round(float(np.median(gaps["40+"])), 3), "median_gap_vs_age_only_20plus": round(float(np.median(gaps["20+"])), 3),
          "share_gap_le_0.05_40plus": round(float(np.mean([g <= 0.05 for g in gaps["40+"]])), 3), "rows": rows,
          "verdict_basis": "the lower, more conservative age-only baseline (adults 40+), which makes the papers look better"}
    a3["supported"] = bool(a3["median_gap_vs_age_only_40plus"] <= 0.05)
    a4 = {"random_vs_later_gap_any_40plus": None}
    t = p2["tasks"].get("any cancer, 40+")
    if t and "random" in t and "later_cycles" in t:
        a4["random_vs_later_gap_any_40plus"] = round(t["random"]["whole picture + missing flags"] - t["later_cycles"]["whole picture + missing flags"], 4)
    a4["era_from_flags_auc"] = p2["era_from_flags_auc"]
    a4["supported"] = bool(a4["random_vs_later_gap_any_40plus"] is not None and a4["random_vs_later_gap_any_40plus"] >= 0.03 and p2["era_from_flags_auc"] > 0.9)
    print(f"\nA3: median gap between a paper's best AUC and age-only AUC for its site: {a3['median_gap_vs_age_only_40plus']:+.3f} (adults 40+), {a3['median_gap_vs_age_only_20plus']:+.3f} (20+)  -> {'supported' if a3['supported'] else 'NOT supported'}")
    print(f"A4: random minus later-cycle AUC for the whole picture with flags: {a4['random_vs_later_gap_any_40plus']}; cycle from flags {a4['era_from_flags_auc']}  -> {'supported' if a4['supported'] else 'NOT supported'}")
    with open(OUT, "w") as f:
        json.dump({"part1": p1, "part2": p2, "A3": a3, "A4": a4}, f, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
