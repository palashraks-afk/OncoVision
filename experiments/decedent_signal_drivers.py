"""
What is the cancer-versus-other-death signal actually made of?

cancer_specific_signal_search.py found that, among people who died, routine labs
separate a cancer death from another death better than age and sex do, and that
it replicated on NHANES III at a ten-year horizon. That is a lead. It is not yet
evidence of anything a patient could use, because there are two very different
ways to get it:

    CANCER MARKERS      values that run abnormal in people heading for a cancer
                        death: a low haemoglobin, a high platelet count, a raised
                        alkaline phosphatase or calcium.
    OTHER-DEATH MARKERS values that run abnormal in people heading for a heart,
                        kidney, lung or diabetes death: a high glucose, a high
                        creatinine, a low HDL. Among decedents, someone WITHOUT
                        those looks relatively like a cancer death by elimination.

The second is a real signal about which way a death will go and says nothing
about a tumour. Only the first would support a cancer panel. They are told apart
by the sign of each value's standardised coefficient and by what the value is
known to mean, so this lists them without choosing the story.

Stability matters as much as size: a coefficient that flips sign across
bootstrap refits is noise. Each is refit on 200 bootstrap samples and reported
with the share of refits that agree on its sign.

Run:  python experiments/decedent_signal_drivers.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = "experiments/decedent_signal_drivers_result.json"
TRAIN = "data/nhanes_mortality_full.csv"
EXTERNAL = "data/nhanes3_mortality_full.csv"
CANCER = 2
HORIZON = 10
CYCLES = ("1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008")
BOOT = 200
DEMO = ["age", "gender"]

# What a raised value usually points to, for reading the list. Not used to
# select anything.
MEANING = {
    "hemoglobin": "low runs with anaemia, chronic disease, bleeding",
    "rdw": "high runs with nutritional deficiency and chronic disease",
    "platelets": "high is a known cancer-associated finding",
    "wbc": "high runs with infection and inflammation",
    "albumin": "low runs with chronic illness of every kind",
    "glucose": "high runs with diabetes, a heart and kidney risk",
    "creatinine": "high runs with kidney disease",
    "bun": "high runs with kidney disease and dehydration",
    "calcium": "high is a known cancer-associated finding",
    "alkaline_phosphatase": "high runs with liver and bone disease and some cancers",
    "ggt": "high runs with liver disease and alcohol",
    "mcv": "high runs with alcohol and B12 deficiency",
}


def fit(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    m.fit(X, y)
    return m[-1].coef_[0]


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(EXTERNAL)
    tr = tr[tr["cycle"].isin(CYCLES)]
    skip = {"cancer_death", "followup_months", "cycle", "race_ethnicity", "died",
            "ucod_leading", "smoking", "alcohol_intake", "bmi"}
    labs = [c for c in tr.columns if c in te.columns and c not in skip and c not in DEMO
            and pd.api.types.is_numeric_dtype(tr[c])]
    cols = DEMO + labs

    def decedents(df):
        d = df[(df["died"] == 1) & (df["followup_months"] <= HORIZON * 12)].copy()
        d["label"] = (d["ucod_leading"] == CANCER).astype(int)
        return d

    d_tr, d_te = decedents(tr), decedents(te)
    med = d_tr[cols].median()
    X = d_tr[cols].fillna(med).reset_index(drop=True)
    y = d_tr["label"].reset_index(drop=True)
    print(f"{len(d_tr):,} decedents within {HORIZON} years ({int(y.sum())} cancer), "
          f"{len(cols)} inputs\n")

    base = fit(X, y)
    rng = np.random.default_rng(0)
    signs = np.zeros(len(cols))
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if y.iloc[i].nunique() < 2:
            continue
        signs += (np.sign(fit(X.iloc[i], y.iloc[i])) == np.sign(base))
    agree = signs / BOOT

    # The same coefficients on the external cohort, as a replication of direction.
    X2 = d_te[cols].fillna(med).reset_index(drop=True)
    ext = fit(X2, d_te["label"].reset_index(drop=True))

    rows = []
    for c, b, a, e in zip(cols, base, agree, ext):
        rows.append({"input": c, "coef": round(float(b), 3), "sign_agreement": round(float(a), 2),
                     "external_coef": round(float(e), 3),
                     "same_sign_externally": bool(np.sign(b) == np.sign(e)),
                     "meaning": MEANING.get(c, "")})
    rows.sort(key=lambda r: -abs(r["coef"]))

    print(f"{'input':<22}{'coef':>7}{'stable':>8}{'NHANES III':>12}  reading (positive = pushes toward cancer)")
    for r in rows[:14]:
        print(f"{r['input']:<22}{r['coef']:>+7.3f}{r['sign_agreement']:>8.0%}"
              f"{r['external_coef']:>+12.3f}  {r['meaning']}")

    towards = [r for r in rows if r["coef"] > 0 and r["sign_agreement"] >= 0.9
               and r["same_sign_externally"] and r["input"] not in DEMO]
    away = [r for r in rows if r["coef"] < 0 and r["sign_agreement"] >= 0.9
            and r["same_sign_externally"] and r["input"] not in DEMO]
    print(f"\n  stable and replicated, pushing TOWARD cancer: "
          f"{', '.join(r['input'] for r in towards) or 'none'}")
    print(f"  stable and replicated, pushing AWAY from cancer: "
          f"{', '.join(r['input'] for r in away) or 'none'}")
    with open(OUT, "w") as f:
        json.dump({"horizon_years": HORIZON, "n": int(len(d_tr)), "cancer": int(y.sum()),
                   "drivers": rows,
                   "toward_cancer": [r["input"] for r in towards],
                   "away_from_cancer": [r["input"] for r in away]}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
