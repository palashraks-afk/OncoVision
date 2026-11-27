"""
Do abnormal flags on other routine tests point to cancer specifically? Rules: docs/ROUTINE_TESTS_PREREG.md
(committed first). Discovery NHANES 1999-2014; replication NHANES III; five-year cancer death versus other-cause death.

Run:  python experiments/routine_test_flags.py discover      # writes the hit list and stops
      python experiments/routine_test_flags.py replicate     # downloads the NHANES III lab file (~40 MB)
"""

import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments")
LABS = ["calcium", "alp", "albumin", "ggt", "wbc", "platelets", "ast", "alt", "hba1c", "ualb", "ucre", "crp"]
FLAGS = ["high_calcium", "high_alp", "low_albumin", "high_ggt", "high_wbc", "high_platelets", "high_fib4", "new_diabetes_hba1c",
         "urine_albumin", "high_crp", "two_or_more"]
ALPHA = 0.05 / len(FLAGS)


def add_flags(d):
    """Flags are 1/0, or NaN when a test needed for them is missing."""
    f = pd.DataFrame(index=d.index)

    def flag(cond, *need):
        ok = np.ones(len(d), bool)
        for c in need:
            ok &= d[c].notna().to_numpy()
        return np.where(ok, cond, np.nan)

    corrected = d["calcium"] + 0.8 * (4 - d["albumin"])
    f["high_calcium"] = flag(corrected > 10.5, "calcium", "albumin")
    f["high_alp"] = flag(d["alp"] > 130, "alp")
    f["low_albumin"] = flag(d["albumin"] < 3.5, "albumin")
    f["high_ggt"] = flag(d["ggt"] > 60, "ggt")
    f["high_wbc"] = flag(d["wbc"] > 11.0, "wbc")
    f["high_platelets"] = flag(d["platelets"] > 400, "platelets")
    fib4 = d["age"] * d["ast"] / (d["platelets"] * np.sqrt(d["alt"].where(d["alt"] > 0)))
    f["high_fib4"] = np.where(fib4.notna(), fib4 > 2.67, np.nan)
    f["new_diabetes_hba1c"] = flag((d["hba1c"] >= 6.5) & (d["told_diabetes"] != 1), "hba1c")
    uacr = 100 * d["ualb"] / d["ucre"].where(d["ucre"] > 0)
    f["urine_albumin"] = np.where(uacr.notna(), uacr >= 30, np.nan)
    f["high_crp"] = flag(d["crp"] > 1.0, "crp")
    base = f[FLAGS[:-1]]
    f["two_or_more"] = np.where(base.notna().sum(axis=1) >= 6, (base.fillna(0).sum(axis=1) >= 2).astype(float), np.nan)
    return pd.concat([d, f], axis=1)


def finish(d):
    d = d[(d["age"] >= 50) & (d["age"] <= 84) & (d["prior_cancer"] == 0) & d["ever_smoked"].notna() & d["died"].notna() & d["months"].notna()]
    d = d[(d["months"] >= 60) | (d["died"] == 1)].reset_index(drop=True)
    d["y"] = np.where((d["died"] == 1) & (d["months"] <= 60) & (d["ucod"] == 2), 1, np.where((d["died"] == 1) & (d["months"] <= 60), 2, 0))
    return add_flags(d)


def discovery_data():
    w = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"), low_memory=False)
    cols = {"calcium": "LBXSCA", "alp": "LBXSAPSI", "albumin": "LBXSAL", "ggt": "LBXSGTSI", "wbc": "LBXWBCSI", "platelets": "LBXPLTSI",
            "ast": "LBXSASSI", "alt": "LBXSATSI", "hba1c": "LBXGH", "ualb": "URXUMA", "ucre": "URXUCR", "crp": "LBXCRP"}
    d = pd.DataFrame({k: pd.to_numeric(w[v], errors="coerce") for k, v in cols.items()})
    d["age"], d["male"], d["wave"] = w["age"], w["gender"].astype(float), w["cycle"]
    d["ever_smoked"] = w["SMQ020"].map({1.0: 1.0, 2.0: 0.0})
    d["told_diabetes"] = (w["DIQ010"] == 1).astype(int)
    d["died"], d["ucod"], d["months"], d["prior_cancer"] = w["died"], w["ucod_leading"], pd.to_numeric(w["followup_months"], errors="coerce"), 0
    return finish(d)


def replication_data():
    from fetch_red_flag_labs import NH3_LAB, get
    spec = {"SEQN": (1, 5), "wbc": (1273, 1277), "crp": (1667, 1671), "calcium": (1731, 1734), "alt": (1824, 1826), "ast": (1821, 1823),
            "ggt": (1827, 1830), "alp": (1835, 1838), "albumin": (1846, 1848), "hba1c": (1861, 1864), "ucre": (1956, 1960), "ualb": (1965, 1970)}
    lab = pd.read_fwf(io.BytesIO(get(NH3_LAB)), colspecs=[(a - 1, b) for a, b in spec.values()], names=list(spec))
    top = {"wbc": 200, "crp": 80, "calcium": 20, "alt": 888, "ast": 888, "ggt": 8888, "alp": 8888, "albumin": 8, "hba1c": 30, "ucre": 8888, "ualb": 88888}
    for c, hi in top.items():
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
        lab[c] = lab[c].mask(lab[c] >= hi)           # fill codes (8s) sit at or above these limits
    base = pd.read_csv(os.path.join(ROOT, "data", "red_flag_followup.csv.gz"), low_memory=False)
    base = base[base["dataset"] == "NHANES III 1988-1994"]
    # told-diabetes comes from the NHANES III adult file (HAD1: ever told you have diabetes)
    adult = pd.read_fwf(io.BytesIO(get("https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat")), colspecs=HAD1_SPEC(), names=["SEQN", "had1"])
    d = base[["SEQN", "age", "male", "smoked", "platelets", "died", "ucod", "months", "prior_cancer"]].merge(lab, on="SEQN", how="left").merge(adult, on="SEQN", how="left")
    d = d.rename(columns={"smoked": "ever_smoked"})
    d["months"] = pd.to_numeric(d["months"], errors="coerce")
    d["told_diabetes"] = (pd.to_numeric(d["had1"], errors="coerce") == 1).astype(int)
    d["wave"] = "NHANES III"
    med = d[LABS].median().round(2).to_dict()
    print("NHANES III lab medians (unit check):", med)
    return finish(d)


def HAD1_SPEC():
    import re
    from fetch_red_flag_labs import get
    txt = get("https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.sas").decode("latin-1")
    m = re.search(r"\n\s+HAD1\s+(\d+)(?:\s*-\s*(\d+))?\s", txt)
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return [(0, 5), (a - 1, b)]


def rcs(x, knots):
    """Restricted cubic spline basis (linear plus k-2 nonlinear terms)."""
    t = np.asarray(knots, float)
    k = len(t)
    cube = lambda u: np.clip(u, 0, None) ** 3  # noqa: E731
    out = [(x - t[0]) / (t[-1] - t[0])]
    for j in range(k - 2):
        out.append((cube(x - t[j]) - cube(x - t[k - 2]) * (t[-1] - t[j]) / (t[-1] - t[k - 2]) + cube(x - t[-1]) * (t[k - 2] - t[j]) / (t[-1] - t[k - 2])) / (t[-1] - t[0]) ** 3)
    return np.column_stack(out)


def multinomial(X, y, iters=60, ridge=1e-8):
    """Maximum-likelihood multinomial logistic regression (classes 0, 1, 2; 0 is reference) by Newton steps.
    Returns coefficients (2 x p) and their covariance (2p x 2p)."""
    n, p = X.shape
    Y = np.column_stack([(y == 1), (y == 2)]).astype(float)
    B = np.zeros((2, p))
    for _ in range(iters):
        eta = X @ B.T
        m = np.exp(np.column_stack([np.zeros(n), eta]))
        P = (m / m.sum(axis=1, keepdims=True))[:, 1:]
        grad = np.concatenate([X.T @ (Y[:, k] - P[:, k]) for k in range(2)]) - ridge * B.ravel()
        H = np.zeros((2 * p, 2 * p))
        for a in range(2):
            for b in range(2):
                wt = P[:, a] * ((a == b) - P[:, b])
                H[a * p:(a + 1) * p, b * p:(b + 1) * p] = X.T @ (X * wt[:, None])
        H += ridge * np.eye(2 * p)
        step = np.linalg.solve(H, grad)
        B += step.reshape(2, p)
        if np.max(np.abs(step)) < 1e-8:
            break
    return B, np.linalg.inv(H)


def one_flag(d, name):
    s = d[d[name].notna()].reset_index(drop=True)
    exp = s[name].to_numpy(float)
    n1, ca1 = int(exp.sum()), int(((s["y"] == 1) & (s[name] == 1)).sum())
    out = {"n": int(len(s)), "flagged": n1, "flagged_share": round(n1 / max(len(s), 1), 4), "cancer_deaths_flagged": ca1,
           "other_deaths_flagged": int(((s["y"] == 2) & (s[name] == 1)).sum())}
    if ca1 < 5:
        out["note"] = "fewer than 5 cancer deaths among the flagged; not modelled"
        return out
    knots = np.percentile(s["age"], [5, 35, 65, 95])
    cols = [np.ones((len(s), 1)), rcs(s["age"].to_numpy(float), knots), s[["male", "ever_smoked"]].to_numpy(float)]
    waves = sorted(s["wave"].unique())
    if len(waves) > 1:
        cols.append(np.column_stack([(s["wave"] == w).to_numpy(float) for w in waves[1:]]))
    X = np.column_stack(cols + [exp[:, None]])
    B, V = multinomial(X, s["y"].to_numpy())
    p = X.shape[1]
    i, j = p - 1, 2 * p - 1
    bc, bo = B[0, -1], B[1, -1]
    sc, so = np.sqrt(V[i, i]), np.sqrt(V[j, j])
    sr = np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
    ci = lambda b, se: [round(float(np.exp(b - 1.96 * se)), 3), round(float(np.exp(b + 1.96 * se)), 3)]  # noqa: E731
    out.update({"rrr_cancer": round(float(np.exp(bc)), 3), "rrr_cancer_ci": ci(bc, sc), "p_cancer": float(2 * norm.sf(abs(bc / sc))),
                "rrr_other": round(float(np.exp(bo)), 3), "rrr_other_ci": ci(bo, so),
                "ratio": round(float(np.exp(bc - bo)), 3), "ratio_ci": ci(bc - bo, sr), "p_ratio": float(2 * norm.sf(abs((bc - bo) / sr)))})
    return out


def run(d):
    return {f: one_flag(d, f) for f in FLAGS}


def show(res):
    for f, r in res.items():
        if "rrr_cancer" in r:
            print(f"  {f:<20} flagged {r['flagged']:>5} ({r['flagged_share']:.1%}) ca {r['cancer_deaths_flagged']:>3}  cancer {r['rrr_cancer']:>5.2f} {r['rrr_cancer_ci']}  "
                  f"other {r['rrr_other']:>5.2f}  ratio {r['ratio']:>5.2f} {r['ratio_ci']}  p_ratio {r['p_ratio']:.2g}")
        else:
            print(f"  {f:<20} flagged {r['flagged']:>5}  ca {r['cancer_deaths_flagged']}  {r['note']}")


def main(mode):
    disc_path = os.path.join(RES, "routine_test_flags_discovery.json")
    if mode == "discover":
        d = discovery_data()
        res = run(d)
        hits = [f for f, r in res.items() if "rrr_cancer" in r and r["rrr_cancer"] > 1 and r["ratio"] > 1 and r["p_cancer"] < ALPHA and r["p_ratio"] < ALPHA]
        print(f"discovery: {len(d):,} adults, {int((d['y'] == 1).sum())} cancer deaths, {int((d['y'] == 2).sum())} other deaths")
        show(res)
        print("hits (frozen):", hits)
        json.dump({"n": len(d), "cancer_deaths": int((d["y"] == 1).sum()), "other_deaths": int((d["y"] == 2).sum()), "alpha": ALPHA, "flags": res, "hits": hits},
                  open(disc_path, "w"), indent=2)
        return
    disc = json.load(open(disc_path))
    d = replication_data()
    res = run(d)
    print(f"replication: {len(d):,} adults, {int((d['y'] == 1).sum())} cancer deaths, {int((d['y'] == 2).sum())} other deaths")
    show(res)
    confirmed = [f for f in disc["hits"] if "rrr_cancer" in res[f] and res[f]["rrr_cancer_ci"][0] > 1 and res[f]["ratio_ci"][0] > 1]
    print("confirmed:", confirmed)
    json.dump({"n": len(d), "cancer_deaths": int((d["y"] == 1).sum()), "other_deaths": int((d["y"] == 2).sum()), "flags": res, "confirmed": confirmed},
              open(os.path.join(RES, "routine_test_flags_replication.json"), "w"), indent=2)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main(sys.argv[1])
