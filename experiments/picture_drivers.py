"""
What in the whole picture separates a cancer death from another death?

whole_picture.py found that, among people who died within ten years, the whole
picture tells a cancer death from another death better than routine labs do (0.685
against 0.647, and 0.678 with no lab values at all), while for cancer death against
everyone it adds nothing over age and sex. The two results fit one story and a
second, and they need telling apart:

    CANCER MARKERS      things that run abnormal in someone heading for a cancer
                        death: weight loss, a smoking history, a cough, anaemia
    OTHER-DEATH MARKERS things that identify someone heading for a heart, lung,
                        kidney or diabetes death: a history of heart failure,
                        coronary disease or stroke, treated blood pressure,
                        diabetes, limitations in walking. Among people who die,
                        having these makes the death probably NOT cancer.

Only the first would support a cancer panel. The second is information about the
kind of death and says nothing about a tumour. They are told apart by the sign of
each variable's standardised coefficient, by whether that sign survives bootstrap
refits, and by whether it holds when the model is trained on earlier cycles and
read on later ones.

Run:  python experiments/picture_drivers.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = "experiments/picture_drivers_result.json"
DATA = "data/nhanes_wide.csv.gz"
CANCER, HORIZON = 2, 120
BOOT, C = 150, 0.003
EARLY = ("1999-2000", "2001-2002", "2003-2004")
META = {"age", "gender", "cycle", "followup_months", "died", "ucod_leading", "SEQN"}

# What a variable is, for reading the list. Written from the NHANES codebooks and
# used only for display; nothing is selected by it. Coding is 1 yes, 2 no for the
# questionnaire items, so a NEGATIVE coefficient on a yes/no item means "yes"
# pushes toward the outcome.
MEANING = {
    "MCQ160B": "told had congestive heart failure", "MCQ160C": "told had coronary heart disease",
    "MCQ160D": "told had angina", "MCQ160E": "told had heart attack",
    "MCQ160F": "told had stroke", "MCQ160G": "told had emphysema",
    "MCQ160K": "told had chronic bronchitis", "MCQ160L": "told had liver condition",
    "MCQ160M": "told had thyroid problem", "MCQ010": "told had asthma",
    "BPQ020": "told had high blood pressure", "BPQ050A": "now taking BP medicine",
    "BPQ080": "told had high cholesterol", "BPQ090D": "told to take cholesterol medicine",
    "DIQ010": "told had diabetes", "SMQ020": "smoked at least 100 cigarettes",
    "WHD020": "current weight (lb)", "WHD050": "weight a year ago (lb)",
    "WHQ030": "how they see their weight", "WHQ040": "want to weigh more or less",
    "WHD010": "height (in)", "HSD010": "self-rated general health",
    "BMXBMI": "body mass index", "BMXWT": "weight (kg)", "BMXWAIST": "waist (cm)",
    "RDQ050": "coughing most days", "CDQ001": "chest pain or discomfort",
    "KIQ020": "told had weak or failing kidneys", "OSQ010A": "broken hip",
    "PFQ061B": "trouble walking a quarter mile",
}


def label(c):
    base = c.split("__")[0]
    s = MEANING.get(base, "")
    return f"{c}" + (f" ({s})" if s else "") + (" [missing]" if c.endswith("__missing") else "")


def fit(X, y):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=C, max_iter=3000))
    m.fit(X, y)
    return m[-1].coef_[0]


def main():
    df = pd.read_csv(DATA)
    months, died = df["followup_months"], df["died"] == 1
    d = df[died & (months <= HORIZON)].reset_index(drop=True)
    d["label"] = (d["ucod_leading"] == CANCER).astype(int)
    feats = ["age", "gender"] + [c for c in df.columns if c not in META]
    X, y = d[feats], d["label"]
    print(f"{len(d):,} decedents within ten years, {int(y.sum())} cancer, {len(feats)} inputs\n")

    base = fit(X, y)
    rng = np.random.default_rng(0)
    agree = np.zeros(len(feats))
    n_ok = 0
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if y.iloc[i].nunique() < 2:
            continue
        agree += np.sign(fit(X.iloc[i], y.iloc[i])) == np.sign(base)
        n_ok += 1
    agree /= max(n_ok, 1)

    early = d["cycle"].isin(EARLY).values
    e_coef, l_coef = fit(X[early], y[early]), fit(X[~early], y[~early])
    same_across_cycles = np.sign(e_coef) == np.sign(l_coef)

    rows = [{"input": c, "meaning": MEANING.get(c.split("__")[0], ""),
             "coef": round(float(b), 3), "sign_agreement": round(float(a), 2),
             "same_sign_early_vs_late": bool(s)}
            for c, b, a, s in zip(feats, base, agree, same_across_cycles)]
    solid = [r for r in rows if r["sign_agreement"] >= 0.9 and r["same_sign_early_vs_late"]
             and r["input"] not in ("age", "gender")]
    solid.sort(key=lambda r: -abs(r["coef"]))
    toward = [r for r in solid if r["coef"] > 0][:12]
    away = [r for r in solid if r["coef"] < 0][:12]

    print("pushing TOWARD a cancer death (stable across refits and across cycles):")
    for r in toward:
        print(f"  {r['coef']:+.3f}  {label(r['input'])}")
    print("\npushing AWAY from cancer, toward another cause:")
    for r in away:
        print(f"  {r['coef']:+.3f}  {label(r['input'])}")

    json.dump({"n": int(len(d)), "cancer": int(y.sum()), "toward_cancer": toward,
               "away_from_cancer": away, "all": sorted(rows, key=lambda r: -abs(r["coef"]))[:40]},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
