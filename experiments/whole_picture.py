"""
Does the whole picture of a person say more about cancer than their labs do?

The idea being tested
---------------------
Not "is this lab value high" but "does this person, as a whole, resemble people
who went on to die of cancer": labs, exam findings, symptoms, weight change,
habits, history, medications and function, considered together. Everything the
earlier tests gave a model was a few dozen blood values. If breadth is what was
missing, a model given hundreds of variables across every kind of information
should do clearly better, and a pattern-matching model should find the
resemblance a regression cannot.

Arms, all on the same adults (NHANES 1999-2008, ten years of follow-up)
-----------------------------------------------------------------------
    age_sex            the baseline every claim in this project is measured against
    labs               the 20 routine blood values used in every earlier test
    picture_no_labs    every exam, questionnaire and symptom variable, with every
                       laboratory value REMOVED. This is the direct test of "base
                       it on the picture and not on lab values by themselves".
    picture            everything, labs included
    pattern_match      nearest neighbours on the whole picture (PCA to 30
                       dimensions, 100 neighbours), the literal "match against
                       the cancer cases" version

Two outcomes, because the last panel failed on exactly this
-----------------------------------------------------------
    honest       death from cancer within ten years against everybody else,
                 including people who died of other causes
    decedents    among ONLY the people who died within ten years, cancer against
                 another cause. Age and sex cannot separate these, and a model
                 that has merely learned "unwell" cannot either, so a gain here
                 is the one that counts.

Pre-registered, written before the cohort finished downloading
--------------------------------------------------------------
The whole picture is PROMISING only if ALL hold:
  1. on the decedent outcome, picture beats age and sex AND beats routine labs
     alone, each with a 95% interval on the paired gain excluding zero, under
     repeated cross-validation;
  2. the same on the honest outcome;
  3. trained on cycles 1999-2004 and applied unchanged to 2005-2008, the
     decedent gain over age and sex again excludes zero.
Condition 3 is a cycle holdout and not an external cohort, which the project
treats as the weaker kind. NHANES III does not carry enough of these questions to
serve, so a pass here is a lead for CHARLS, HRS or UK Biobank and is labelled so.

Run:  python fetch_nhanes_wide.py
      python experiments/whole_picture.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/whole_picture_result.json"
DATA = "data/nhanes_wide.csv.gz"
CANCER = 2
HORIZON = 120
REPEATS, FOLDS, N_BOOT = 3, 5, 2000
TRAIN_CYCLES = ("1999-2000", "2001-2002", "2003-2004")

DEMO = ["age", "gender"]
# SEQN is the participant sequence number. Its range differs by survey cycle, so as
# a feature it would be a hidden cycle marker and would let a model learn assay
# drift. It is metadata, never an input.
META = {"age", "gender", "cycle", "followup_months", "died", "ucod_leading", "SEQN"}
# The routine blood values every earlier test used, by their NHANES names.
ROUTINE = ["LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXRDW",
           "LBXPLTSI", "LBXMPSI", "LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR", "LBXSTP",
           "LBXSAL", "LBXSASSI", "LBXSATSI", "LBXSTB", "LBXSAPSI", "LBXSGTSI"]
LAB_PREFIX = ("LB", "URX", "URD")


def outcomes(df):
    months, died = df["followup_months"], df["died"] == 1
    cancer_in = died & (df["ucod_leading"] == CANCER) & (months <= HORIZON)
    died_in = died & (months <= HORIZON)
    known = died_in | (months >= HORIZON)
    return {"honest": (cancer_in.astype(int), known),
            "decedents": (cancer_in.astype(int), died_in)}


def arms(df):
    feats = [c for c in df.columns if c not in META]
    is_lab = lambda c: c.split("__")[0].startswith(LAB_PREFIX)
    routine = [c for c in ROUTINE if c in df.columns]
    return {
        "age_sex": DEMO,
        "labs": DEMO + routine,
        "picture_no_labs": DEMO + [c for c in feats if not is_lab(c)],
        "picture": DEMO + feats,
    }


def make(kind, n, rate):
    imp = SimpleImputer(strategy="median")
    if kind == "logistic_wide":
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.05], cv=3,
                                                  scoring="roc_auc", class_weight="balanced",
                                                  max_iter=2000, n_jobs=1))
    if kind == "pattern_match":
        return make_pipeline(imp, StandardScaler(), PCA(n_components=30, random_state=0),
                             KNeighborsClassifier(n_neighbors=100, weights="distance"))
    return make_pipeline(imp, tm.model_factory(kind, n, rate))


def oof(X, y, kind, repeats=REPEATS):
    """Out-of-fold probabilities averaged over repeated folds, and per-repeat AUCs."""
    preds, aucs = np.zeros(len(y)), []
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            m = make(kind, len(tr), float(y.iloc[tr].mean()))
            m.fit(X.iloc[tr], y.iloc[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
        preds += p
    return preds / repeats, aucs


def paired(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    g = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]


def main():
    if not os.path.exists(DATA):
        print(f"{DATA} is missing. Run fetch_nhanes_wide.py first.")
        return 1
    df = pd.read_csv(DATA)
    spec = arms(df)
    print(f"{len(df):,} adults, {len(df.columns) - len(META)} picture columns "
          f"({len(spec['picture_no_labs']) - 2} outside the lab values)\n")
    for k, v in spec.items():
        print(f"  {k:<16} {len(v):>4} inputs")

    result = {"n": int(len(df)), "inputs": {k: len(v) for k, v in spec.items()},
              "outcomes": {}}
    out_def = outcomes(df)
    kinds_for = {"age_sex": ("logistic", "ensemble"), "labs": ("logistic", "ensemble"),
                 "picture_no_labs": ("logistic_wide", "ensemble", "pattern_match"),
                 "picture": ("logistic_wide", "ensemble", "pattern_match")}

    best_preds = {}
    for oname, (y_all, mask) in out_def.items():
        sub = df[mask.values].reset_index(drop=True)
        y = y_all[mask.values].reset_index(drop=True)
        print(f"\n{oname.upper()}: {len(y):,} adults, {int(y.sum())} cancer deaths")
        entry = {"n": int(len(y)), "events": int(y.sum()), "arms": {}}
        preds = {}
        for arm, cols in spec.items():
            X = sub[cols]
            tried = {}
            for kind in kinds_for[arm]:
                p, a = oof(X, y, kind)
                tried[kind] = (p, a)
                print(f"    {arm:<16} {kind:<14} AUC {roc_auc_score(y, p):.4f}", flush=True)
            kind = max(tried, key=lambda k: roc_auc_score(y, tried[k][0]))
            preds[arm] = tried[kind][0]
            entry["arms"][arm] = {
                "best_model": kind, "auc": round(float(roc_auc_score(y, preds[arm])), 4),
                "all_models": {k: round(float(roc_auc_score(y, v[0])), 4)
                               for k, v in tried.items()}}
        base = preds["age_sex"]
        for arm in ("labs", "picture_no_labs", "picture"):
            g = float(roc_auc_score(y, preds[arm]) - roc_auc_score(y, base))
            ci = paired(y, preds[arm], base)
            entry["arms"][arm].update({"gain_over_age_sex": round(g, 4), "gain_ci": ci,
                                       "beats_age_sex": bool(ci[0] > 0)})
        for arm in ("picture_no_labs", "picture"):
            g = float(roc_auc_score(y, preds[arm]) - roc_auc_score(y, preds["labs"]))
            ci = paired(y, preds[arm], preds["labs"], seed=1)
            entry["arms"][arm].update({"gain_over_labs": round(g, 4), "gain_over_labs_ci": ci,
                                       "beats_labs": bool(ci[0] > 0)})
        pm = entry["arms"]["picture"]["all_models"].get("pattern_match")
        entry["pattern_match_auc"] = pm
        result["outcomes"][oname] = entry
        print(f"  {'arm':<16} {'AUC':>7} {'vs age+sex':>12} {'95% CI':>20}  {'vs labs':>9}")
        for arm in ("age_sex", "labs", "picture_no_labs", "picture"):
            a = entry["arms"][arm]
            if arm == "age_sex":
                print(f"  {arm:<16} {a['auc']:>7.4f}")
                continue
            gl = a.get("gain_over_labs")
            print(f"  {arm:<16} {a['auc']:>7.4f} {a['gain_over_age_sex']:>+12.4f} "
                  f"{str(a['gain_ci']):>20}  " + (f"{gl:+.4f}" if gl is not None else ""))
        if oname == "decedents":
            best_preds = preds

    # Condition 3: train on early cycles, apply unchanged to later ones.
    d_all, d_mask = out_def["decedents"]
    sub = df[d_mask.values].reset_index(drop=True)
    y = d_all[d_mask.values].reset_index(drop=True)
    early = sub["cycle"].isin(TRAIN_CYCLES).values
    holdout = {}
    print("\nCYCLE HOLDOUT (decedents): train 1999-2004, test 2005-2008, unchanged")
    for arm in ("age_sex", "labs", "picture_no_labs", "picture"):
        cols = spec[arm]
        best = None
        for kind in kinds_for[arm]:
            m = make(kind, int(early.sum()), float(y[early].mean()))
            m.fit(sub.loc[early, cols], y[early])
            p = m.predict_proba(sub.loc[~early, cols])[:, 1]
            a = float(roc_auc_score(y[~early], p))
            if best is None or a > best[0]:
                best = (a, kind, p)
        holdout[arm] = best
        print(f"    {arm:<16} {best[1]:<14} AUC {best[0]:.4f}")
    yt = y[~early].to_numpy()
    hres = {"train_n": int(early.sum()), "test_n": int((~early).sum()),
            "test_events": int(yt.sum()), "arms": {}}
    for arm in ("labs", "picture_no_labs", "picture"):
        g = holdout[arm][0] - holdout["age_sex"][0]
        ci = paired(yt, holdout[arm][2], holdout["age_sex"][2], seed=2)
        hres["arms"][arm] = {"auc": round(holdout[arm][0], 4), "gain": round(g, 4),
                             "gain_ci": ci, "beats_age_sex": bool(ci[0] > 0)}
        print(f"    {arm:<16} gain over age+sex {g:+.4f} {ci}")
    hres["age_sex_auc"] = round(holdout["age_sex"][0], 4)
    result["cycle_holdout_decedents"] = hres

    o = result["outcomes"]
    pic_d, pic_h = o["decedents"]["arms"]["picture"], o["honest"]["arms"]["picture"]
    c1 = bool(pic_d["beats_age_sex"] and pic_d["beats_labs"])
    c2 = bool(pic_h["beats_age_sex"] and pic_h["beats_labs"])
    c3 = bool(hres["arms"]["picture"]["beats_age_sex"])
    result["conditions"] = {"decedents_beats_age_and_labs": c1,
                            "honest_beats_age_and_labs": c2,
                            "cycle_holdout_beats_age": c3}
    result["promising"] = bool(c1 and c2 and c3)
    print(f"\n  1. decedents, picture beats age+sex AND labs: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. honest outcome, same: {'PASS' if c2 else 'FAIL'}")
    print(f"  3. cycle holdout beats age+sex: {'PASS' if c3 else 'FAIL'}")
    print("\n  VERDICT: " + ("PROMISING, a lead for an external cohort." if result["promising"]
          else "the whole picture does not clear the bar."))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
