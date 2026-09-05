"""
Do waist circumference and exercise help the general panel?

Why ask
-------
The general panel is the weakest thing in the application. It adds about 0.006
over simply knowing someone's age and sex, which is why its card carries the
"barely beats age and sex" warning. Serum cotinine, CRP and the whole lab report
have already been tried on it and moved that to 0.009: see general_rescue.py.

Two things had never been offered to it, and both were sitting in NHANES:

    waist       BMI cannot tell a heavy-set person from a centrally obese one,
                and it is central adiposity the obesity-cancer literature ties
                to risk. A clinic measures it with a tape.
    exercise    the application already ASKS for hours of exercise per week and
                then no panel reads the answer. Either it earns its place or the
                question should stop being asked.

Cycle coverage
--------------
NHANES moved to the Global Physical Activity Questionnaire in 2007, so exercise
is present for 2007-2014 and absent for 2005-2006 — missing by cycle, not at
random. Median imputation across that gap would hand 4,147 people a made-up
value and let the model learn "no exercise recorded" as a cycle marker.

So every arm is run twice: once on the full pooled cohort the panel ships on,
and once restricted to the four cycles that actually asked. If an arm only wins
on the imputed version, it found the cycle, not the risk factor.

Judged on gain over age and sex, not on raw AUC, because raw AUC on this panel
is mostly a measure of how old the cohort is.

Run:  python experiments/general_body_activity.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

REPEATS = 5
OUT = "experiments/general_body_activity_result.json"

SHIPPED = ["age", "gender", "bmi", "smoking", "alcohol_intake"]
ARMS = {
    "A shipped today": SHIPPED,
    "B + waist": SHIPPED + ["waist"],
    "C + exercise": SHIPPED + ["physical_activity"],
    "D + waist and exercise": SHIPPED + ["waist", "physical_activity"],
}


def oof_auc(X, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def run(df, label):
    y = df["recent_cancer"].astype(int).reset_index(drop=True)
    print(f"\n--- {label} ---")
    print(f"{len(y)} adults, {int(y.sum())} with a cancer diagnosed within four years")

    base = np.array([
        oof_auc(df[["age", "gender"]].apply(pd.to_numeric, errors="coerce")
                .reset_index(drop=True).pipe(lambda f: f.fillna(f.median())), y, s)
        for s in range(REPEATS)])
    print(f"  age and sex alone      AUC {base.mean():.3f}")

    out = {}
    for name, feats in ARMS.items():
        X = df[feats].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        arr = np.array([oof_auc(X, y, s) for s in range(REPEATS)])
        gain = arr - base
        out[name] = {"auc": round(float(arr.mean()), 3),
                     "gain_over_age_sex": round(float(gain.mean()), 3)}
        print(f"  {name:<24} AUC {arr.mean():.3f}   gain over age/sex "
              f"{gain.mean():+.3f}", flush=True)

    shipped = out["A shipped today"]["gain_over_age_sex"]
    for name, v in out.items():
        if name == "A shipped today":
            continue
        v["improvement"] = round(v["gain_over_age_sex"] - shipped, 3)
    return out


def main():
    df = pd.read_csv("data/nhanes_screening_general.csv")

    full = run(df, "full pooled cohort, exercise imputed for 2005-2006")
    asked = df[df["physical_activity"].notna()].reset_index(drop=True)
    sub = run(asked, "2007-2014 only, everyone was actually asked")

    print("\n" + "=" * 78)
    verdict = {}
    for name in ARMS:
        if name == "A shipped today":
            continue
        f, s = full[name]["improvement"], sub[name]["improvement"]
        # Has to survive on the cycles that actually asked, not just on the
        # pooled cohort where a missing answer marks the cycle.
        real = f > 0.005 and s > 0.005
        verdict[name] = bool(real)
        print(f"  {name:<24} pooled {f:+.3f}   asked-only {s:+.3f}   "
              f"-> {'REAL' if real else 'not enough'}")

    adopt = [k for k, v in verdict.items() if v]
    print(f"\n-> {'adopt: ' + max(adopt, key=lambda k: sub[k]['improvement']) if adopt else 'neither earns a place'}")
    if not adopt:
        print("   the exercise question is asked and read by nothing, so it should go")

    with open(OUT, "w") as f:
        json.dump({"full_cohort": full, "cycles_that_asked": sub,
                   "verdict": verdict, "adopt": adopt}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
