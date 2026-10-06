"""
Three follow-up analyses, fixed in docs/RISK_RANKING_PREREG.md (Addendum) before running.

F1  a curated, cycle-stable feature set (S4) against the strong baseline
F2  is the whole-picture failure a survey-cycle artefact?
F3  does the pooled cancer-specificity result differ from NHANES III?

Run:  python experiments/risk_followups.py     (needs data/nhanes_wide_8c.csv.gz and
      experiments/navigator_evidence_result.json)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from risk_ranking import CANCER, boot_gain, logreg, slope

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "risk_followups_result.json")

S4_COLS = ["LBXGH", "LBXSAL", "LBXSCH", "LBXSGTSI", "LBXSUA", "LBXHGB", "LBXMCVSI", "LBXPLTSI", "LBXWBCSI", "BMXWAIST", "BMXWT",
           "DIQ010", "BPQ020", "MCQ010", "MCQ053", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160K",
           "MCQ160L", "HUQ010"]


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_wide_8c.csv.gz"))
    d["year0"] = d["cycle"].str[:4].astype(int)
    d = d[d["age"] >= 40]
    d = d[(d["followup_months"] >= 60) | (d["died"] == 1)].reset_index(drop=True)
    d["y"] = ((d["died"] == 1) & (d["ucod_leading"] == CANCER) & (d["followup_months"] <= 60)).astype(int)
    d["smoked"] = np.where(d["SMQ020"] == 1, 1.0, np.where(d["SMQ020"] == 2, 0.0, np.nan))
    d["male"] = (d["gender"] == 1).astype(float)
    w0, w1 = d["WHD020"].where(d["WHD020"].between(50, 600)), d["WHD050"].where(d["WHD050"].between(50, 600))
    d["weight_change"] = (w1 - w0) / w1
    s1 = ["age", "male", "smoked", "BMXBMI"]
    s4 = list(dict.fromkeys(s1 + [c for c in S4_COLS if c in d.columns] + ["weight_change"]))
    # medical-condition codes: 1 = yes, 2 = no, 7/9 = unknown -> 1/0/missing
    for c in s4:
        if c.startswith(("DIQ", "BPQ", "MCQ")):
            d[c] = np.where(d[c] == 1, 1.0, np.where(d[c] == 2, 0.0, np.nan))
    tr, te = d[d["year0"] <= 2005].reset_index(drop=True), d[d["year0"] >= 2007].reset_index(drop=True)
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "s4_variables": len(s4), "s4_list": s4}

    # ---------------- F1
    m1 = logreg(1.0).fit(tr[s1], ytr)
    m4 = logreg(0.05).fit(tr[s4], ytr)
    p1, p4 = m1.predict_proba(te[s1])[:, 1], m4.predict_proba(te[s4])[:, 1]
    gain, gci = roc_auc_score(yte, p4) - roc_auc_score(yte, p1), boot_gain(yte, p1, p4)
    sl, ic = slope(yte, p4)
    dm = ((te["died"] == 1) & (te["followup_months"] <= 60)).to_numpy()
    yd = yte[dm]
    dgain, dci = roc_auc_score(yd, p4[dm]) - roc_auc_score(yd, p1[dm]), boot_gain(yd, p1[dm], p4[dm])
    cv_auc = roc_auc_score(ytr, cross_val_predict(logreg(0.05), tr[s4], ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1])
    res["F1"] = {"auc_S1": round(float(roc_auc_score(yte, p1)), 4), "auc_S4": round(float(roc_auc_score(yte, p4)), 4), "cv_auc_S4_train_era": round(float(cv_auc), 4),
                 "gain": round(float(gain), 4), "gain_ci": gci, "slope": sl, "decedent_gain": round(float(dgain), 4), "decedent_gain_ci": dci,
                 "decedent_auc_S4": round(float(roc_auc_score(yd, p4[dm])), 4), "decedent_auc_S1": round(float(roc_auc_score(yd, p1[dm])), 4)}
    res["F1"]["passes"] = bool(gain >= 0.01 and gci[0] > 0 and 0.8 <= sl <= 1.2 and dci[0] > 0)
    res["F1"]["parts"] = {"gain_bar": bool(gain >= 0.01 and gci[0] > 0), "calibration_bar": bool(0.8 <= sl <= 1.2), "decedent_bar": bool(dci[0] > 0)}
    print(f"F1  curated set ({len(s4)} variables): AUC {res['F1']['auc_S4']:.3f} vs baseline {res['F1']['auc_S1']:.3f}, gain {gain:+.4f} {gci}")
    print(f"    calibration slope {sl}; decedent AUC {res['F1']['decedent_auc_S4']:.3f} vs {res['F1']['decedent_auc_S1']:.3f}, gain {dgain:+.4f} {dci}  ->  "
          f"{'PASSES' if res['F1']['passes'] else 'FAILS'} ({res['F1']['parts']})")

    # ---------------- F2
    flags = [c for c in d.columns if c.endswith("__missing")]
    era = (d["year0"] >= 2007).astype(int)
    keep = d["year0"].isin(list(range(1999, 2006)) + list(range(2007, 2014)))
    Xf, ye = d.loc[keep, flags].fillna(0), era[keep]
    pe = cross_val_predict(LogisticRegression(C=0.5, max_iter=3000), Xf, ye, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    era_auc = roc_auc_score(ye, pe)
    allcols = [c for c in d.columns if c not in {"SEQN", "died", "ucod_leading", "followup_months", "cycle", "y", "year0", "smoked", "male", "gender", "weight_change"}
               and d[c].dtype != object]
    s3 = list(dict.fromkeys(["age", "male"] + allcols + ["smoked"]))
    a, b = train_test_split(d, test_size=0.5, random_state=1, stratify=d["y"])
    m3 = logreg(0.05).fit(a[s3], a["y"])
    rand_auc = roc_auc_score(b["y"], m3.predict_proba(b[s3])[:, 1])
    nf = [c for c in s3 if not c.endswith("__missing")]
    m3b = logreg(0.05).fit(a[nf], a["y"])
    rand_auc_nf = roc_auc_score(b["y"], m3b.predict_proba(b[nf])[:, 1])
    res["F2"] = {"era_from_flags_auc": round(float(era_auc), 4), "n_flags": len(flags), "whole_picture_random_split_auc": round(float(rand_auc), 4),
                 "whole_picture_era_split_auc": 0.6288, "no_flags_random_split_auc": round(float(rand_auc_nf), 4),
                 "reading": "artefact consistent" if era_auc > 0.9 and rand_auc > 0.70 else "artefact not supported"}
    print(f"\nF2  survey era predicted from missing-value flags alone: AUC {era_auc:.3f} ({len(flags)} flags)")
    print(f"    whole picture, random half of all cycles: AUC {rand_auc:.3f}  (era split gave 0.629;  no flags, random split {rand_auc_nf:.3f})  -> {res['F2']['reading']}")

    # ---------------- F3
    ev = json.load(open(os.path.join(ROOT, "experiments", "navigator_evidence_result.json")))["q2"]["results"]
    a_, b_ = ev["1999-2014 labs only"], ev["NHANES III labs only"]
    se = lambda r: (np.log(r["ci"][1]) - np.log(r["ci"][0])) / (2 * 1.96)
    z = (np.log(a_["or"]) - np.log(b_["or"])) / np.sqrt(se(a_) ** 2 + se(b_) ** 2)
    res["F3"] = {"or_1999_2014": a_["or"], "or_nhanes3": b_["or"], "z": round(float(z), 3), "p": round(float(2 * (1 - norm.cdf(abs(z)))), 5)}
    print(f"\nF3  labs-only OR 1999-2014 {a_['or']} vs NHANES III {b_['or']}: z = {z:.2f}, p = {res['F3']['p']}")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
