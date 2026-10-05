"""
Do labs add anything to what the existing life-expectancy tools already use?

The question that decides whether this is new
---------------------------------------------
screening_benefit_reclassification.py showed that labs improve ten-year survival
prediction over age and sex. That is not the interesting comparison. The standard
tools for deciding whether an older adult is likely to live ten years (the
Lee-Schonberg index is the usual one) do not use age and sex alone. They use age,
sex, BMI, smoking, diabetes, emphysema, heart failure, past cancer, hospital stays
and difficulty walking and managing daily tasks, and no laboratory values at all.

So the real test is whether laboratory values add information on top of that kind
of history and function, in the age range where the screening decision is made.
If they add nothing, a lab-based tool offers nothing the existing ones lack.

Arms, adults 65 to 84 in NHANES 1999-2008, ten-year death from any cause
-----------------------------------------------------------------------
    age_sex           the baseline
    history_function  every NHANES exam, questionnaire and symptom variable with
                      every laboratory value removed. A broader stand-in for the
                      inputs the Lee-Schonberg index uses, so it is if anything
                      the harder comparator to beat.
    labs              age, sex and the 20 routine blood values
    history_and_labs  all of it

Repeated five-fold cross-validation averaged over 3 repeats, with a paired
bootstrap on the averaged out-of-fold predictions. This is internal, and says so:
the history and function questions do not exist in NHANES III, so there is no
external cohort to confirm it.

Pre-registered, written before running: labs are WORTH ADDING only if
history_and_labs beats history_function with a 95% interval excluding zero. Anything
less means the labs duplicate what history and function already say.

Run:  python experiments/screening_benefit_vs_history.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/screening_benefit_vs_history_result.json"
DATA = "data/nhanes_wide.csv.gz"
HORIZON, ZONE = 120, (65, 84)
REPEATS, FOLDS, N_BOOT = 3, 5, 2000
DEMO = ["age", "gender"]
META = {"age", "gender", "cycle", "followup_months", "died", "ucod_leading", "SEQN"}
ROUTINE = ["LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXRDW",
           "LBXPLTSI", "LBXMPSI", "LBXSGL", "LBXSCA", "LBXSBU", "LBXSCR", "LBXSTP",
           "LBXSAL", "LBXSASSI", "LBXSATSI", "LBXSTB", "LBXSAPSI", "LBXSGTSI"]
LAB_PREFIX = ("LB", "URX", "URD")


def make(kind, n, rate):
    imp = SimpleImputer(strategy="median")
    if kind == "logistic_wide":
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.05], cv=3,
                                                  scoring="roc_auc", max_iter=2000))
    return make_pipeline(imp, tm.model_factory(kind, n, rate))


def oof(X, y, kind):
    pred = np.zeros(len(y))
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            m = make(kind, len(tr), float(y.iloc[tr].mean()))
            m.fit(X.iloc[tr], y.iloc[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        pred += p
    return pred / REPEATS


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
    df = pd.read_csv(DATA)
    months, died = df["followup_months"], df["died"] == 1
    died_in = died & (months <= HORIZON)
    known = died_in | (months >= HORIZON)
    zone = df["age"].between(*ZONE)
    d = df[(known & zone).values].reset_index(drop=True)
    y = died_in[(known & zone).values].astype(int).reset_index(drop=True)
    feats = [c for c in d.columns if c not in META]
    is_lab = lambda c: c.split("__")[0].startswith(LAB_PREFIX)
    spec = {"age_sex": DEMO,
            "history_function": DEMO + [c for c in feats if not is_lab(c)],
            "labs": DEMO + [c for c in ROUTINE if c in d.columns],
            "history_and_labs": DEMO + feats}
    print(f"Adults {ZONE[0]}-{ZONE[1]}: {len(y):,}, {int(y.sum()):,} died within 10 years "
          f"({y.mean():.1%})\n")
    kinds = {"age_sex": ("logistic", "ensemble"), "labs": ("logistic", "ensemble"),
             "history_function": ("logistic_wide", "ensemble"),
             "history_and_labs": ("logistic_wide", "ensemble")}
    preds, result = {}, {"n": int(len(y)), "events": int(y.sum()), "ages": list(ZONE), "arms": {}}
    for arm, cols in spec.items():
        best = None
        for kind in kinds[arm]:
            p = oof(d[cols], y, kind)
            a = float(roc_auc_score(y, p))
            print(f"    {arm:<17} {kind:<14} AUC {a:.4f} ({len(cols)} inputs)", flush=True)
            if best is None or a > best[0]:
                best = (a, kind, p)
        preds[arm] = best[2]
        result["arms"][arm] = {"inputs": len(cols), "model": best[1], "auc": round(best[0], 4)}

    print()
    rows = [("history_function", "age_sex", "history and function over age and sex"),
            ("labs", "age_sex", "labs over age and sex"),
            ("history_and_labs", "history_function", "LABS ON TOP OF history and function"),
            ("history_and_labs", "labs", "history and function on top of labs")]
    for a, b, text in rows:
        g = float(roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b]))
        ci = paired(y, preds[a], preds[b])
        result[f"{a}_over_{b}"] = {"gain": round(g, 4), "ci": ci, "excludes_zero": bool(ci[0] > 0)}
        print(f"  {text:<42} {g:+.4f} {ci}  {'*' if ci[0] > 0 else ''}")

    key = result["history_and_labs_over_history_function"]
    result["labs_worth_adding"] = bool(key["excludes_zero"])
    print(f"\n  VERDICT: labs are "
          + ("worth adding on top of history and function." if key["excludes_zero"] else
             "not shown to add anything to history and function."))
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
