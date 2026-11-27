"""
Can the rest of the blood count decide which anaemic adults 50+ need a ferritin test?
Rules and bar: docs/CBC_RULE_PREREG.md (committed first).

Run:  python experiments/cbc_ferritin_rule.py derive      # NHANES III: freezes the rules
      python experiments/cbc_ferritin_rule.py validate    # NHANES 2001-2002 and 2017-March 2020
"""

import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fetch_red_flag_labs import NH3_LAB, get  # noqa: E402
from red_flag_labs import flags, ratio_est  # noqa: E402

FROZEN = os.path.join(ROOT, "experiments", "cbc_rule_frozen.json")
OUT = os.path.join(ROOT, "experiments", "cbc_rule_result.json")
FEATS = ["mcv", "mch", "mchc", "rdw", "log_plt", "age", "male"]
SIMPLE = {"MCV below 80": lambda d: d["mcv"] < 80, "MCH below 27": lambda d: d["mch"] < 27,
          "MCHC below 32": lambda d: d["mchc"] < 32, "RDW above 14.5": lambda d: d["rdw"] > 14.5}


def check_ranges(d, name):
    med = d[["hgb", "mcv", "mch", "mchc", "rdw", "plt"]].median().round(1).to_dict()
    print(f"{name} medians: {med}")
    for c, (lo, hi) in {"hgb": (12, 16), "mcv": (82, 96), "mch": (27, 33), "mchc": (31, 36), "rdw": (11.5, 15), "plt": (200, 320)}.items():
        if not lo <= med[c] <= hi:
            raise ValueError(f"{name}: median {c}={med[c]} outside {lo}-{hi}; wrong column or unit")


def anaemic(d):
    a, ida, _ = flags(d["hgb"].to_numpy(), d["ferritin"].to_numpy(), d["male"].to_numpy())
    d = d.assign(ida=ida.astype(int), log_plt=np.log(d["plt"].clip(lower=10)))
    return d[a].reset_index(drop=True)


def nhanes3():
    spec = {"SEQN": (1, 5), "mcv": (1340, 1344), "mch": (1345, 1349), "mchc": (1355, 1359), "rdw": (1360, 1364)}
    lab = pd.read_fwf(io.BytesIO(get(NH3_LAB)), colspecs=[(a - 1, b) for a, b in spec.values()], names=list(spec))
    for c in list(spec)[1:]:
        lab[c] = pd.to_numeric(lab[c], errors="coerce").mask(lambda s: s >= 888)
    if lab["mchc"].median() > 100:                      # SI units (g/L)
        lab["mchc"] = lab["mchc"] / 10
    b = pd.read_csv(os.path.join(ROOT, "data", "red_flag_followup.csv.gz"), low_memory=False)
    b = b[b["dataset"] == "NHANES III 1988-1994"].rename(columns={"hemoglobin": "hgb", "platelets": "plt"})
    d = b[["SEQN", "age", "male", "hgb", "plt", "ferritin"]].merge(lab.drop(columns=[]), on="SEQN", how="inner")
    d = d[(d["age"] >= 50)].dropna(subset=["hgb", "ferritin", "mcv", "mch", "mchc", "rdw", "plt"])
    check_ranges(d, "NHANES III")
    return anaemic(d)


def modern():
    out = []
    for base, cbc_url, wt, periods, label in (
            ("red_flag_2001.csv.gz", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/L25_B.xpt", "WTMEC2YR", "RHQ030", "2001-2002"),
            ("red_flag_current.csv.gz", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_CBC.xpt", "WTMECPRP", "RHQ031", "2017-2020")):
        d = pd.read_csv(os.path.join(ROOT, "data", base))
        cbc = pd.read_sas(io.BytesIO(get(cbc_url)), format="xport")[["SEQN", "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMCVSI"]]
        d = d.drop(columns=[c for c in ("LBXPLTSI", "LBXMCVSI") if c in d]).merge(cbc, on="SEQN", how="left")
        d["male"] = (d["RIAGENDR"] == 1).astype(int)
        d = d[(d["RIDAGEYR"] >= 50) & (d[wt] > 0) & ~((d["male"] == 0) & (d[periods] == 1))]
        d = d.rename(columns={"RIDAGEYR": "age", "LBXHGB": "hgb", "LBXFER": "ferritin", "LBXMCVSI": "mcv", "LBXMCHSI": "mch", "LBXMC": "mchc",
                              "LBXRDW": "rdw", "LBXPLTSI": "plt"})
        d = d.dropna(subset=["hgb", "ferritin", "mcv", "mch", "mchc", "rdw", "plt"])
        check_ranges(d, label)
        d = anaemic(d)
        d["w"], d["strata"], d["psu"], d["survey"] = d[wt], label + "-" + d["SDMVSTRA"].astype(int).astype(str), d["SDMVPSU"], label
        out.append(d)
    return pd.concat(out, ignore_index=True)


def rules(d, frozen):
    preds = {k: f(d).to_numpy() for k, f in SIMPLE.items()}
    preds["Any of the four"] = np.column_stack(list(preds.values())).any(axis=1)
    m = frozen["model"]
    z = m["intercept"] + sum(m["coef"][f] * d[f].to_numpy(float) for f in FEATS)
    preds["Model on all indices"] = 1 / (1 + np.exp(-z)) >= m["cutoff"]
    return preds


def derive():
    d = nhanes3()
    y = d["ida"].to_numpy()
    lr = LogisticRegression(C=1e6, solver="newton-cholesky", max_iter=500).fit(d[FEATS], y)
    p = lr.predict_proba(d[FEATS])[:, 1]
    cut = float(np.quantile(p[y == 1], 0.05, method="lower"))
    frozen = {"n_anaemic": int(len(d)), "ida": int(y.sum()),
              "model": {"intercept": float(lr.intercept_[0]), "coef": {f: float(c) for f, c in zip(FEATS, lr.coef_[0])}, "cutoff": cut}}
    frozen["derivation"] = {k: {"sensitivity": float(v[y == 1].mean()), "tests_avoided": float(1 - v.mean())} for k, v in rules(d, frozen).items()}
    json.dump(frozen, open(FROZEN, "w"), indent=2)
    print(f"NHANES III anaemic adults 50+: {len(d)}, with ferritin < 45: {y.sum()}")
    for k, v in frozen["derivation"].items():
        print(f"  {k:<22} sensitivity {v['sensitivity']:.1%}  tests avoided {v['tests_avoided']:.1%}")


def validate():
    frozen = json.load(open(FROZEN))
    d = modern()
    res = {}
    for label, sub in [("pooled", d)] + [(s, d[d["survey"] == s].reset_index(drop=True)) for s in ("2001-2002", "2017-2020")]:
        w, h, c, y = sub["w"].to_numpy(), sub["strata"].to_numpy(), sub["psu"].to_numpy(), sub["ida"].to_numpy(float)
        r = {"n_anaemic": int(len(sub)), "ida": int(y.sum())}
        for k, v in rules(sub, frozen).items():
            se, se_ci = ratio_est(v * y, y, w, h, c)
            av, av_ci = ratio_est((~v).astype(float), np.ones(len(sub)), w, h, c)
            r[k] = {"sensitivity": se, "sensitivity_ci": se_ci, "tests_avoided": av, "tests_avoided_ci": av_ci, "missed_cases": int(((~v) & (y == 1)).sum())}
        res[label] = r
        print(f"\n{label}: anaemic {r['n_anaemic']}, ferritin < 45: {r['ida']}")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:<22} sensitivity {v['sensitivity']:.1%} [{v['sensitivity_ci'][0]:.0%}, {v['sensitivity_ci'][1]:.0%}]  "
                      f"tests avoided {v['tests_avoided']:.1%}  missed {v['missed_cases']}")
    res["usable_rules"] = [k for k, v in res["pooled"].items() if isinstance(v, dict) and v["sensitivity"] >= 0.95 and v["sensitivity_ci"][0] >= 0.90 and v["tests_avoided"] >= 0.25]
    print("\nusable rules:", res["usable_rules"])
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    {"derive": derive, "validate": validate}[sys.argv[1]]()
