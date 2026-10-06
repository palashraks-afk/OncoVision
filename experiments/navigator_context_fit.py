"""
Fit the navigator's "how much of this is just your age" context layer and write it to
backend/navigator_context.json.

The layer answers one question: among US adults of this age, sex and smoking history,
roughly how many died of cancer within five years of a health examination. It is a
population base rate and not a diagnosis, and it does not use the person's labs.

It was validated out of era first (experiments/navigator_evidence.py, question 6: fitted on
1999-2006, tested on 2007-2014, calibration slope required to be between 0.8 and 1.2). The
deployed version is then refitted on every cycle with enough follow-up (1999-2014).

Run:  python experiments/navigator_context_fit.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_navigator.csv.gz"))
    d = d[(d["age"] >= 40) & (d["cycle"].str[:4].astype(int) <= 2013)]
    d = d[(d["months"] >= 60) | (d["died"] == 1)]
    y = ((d["died"] == 1) & (d["ucod"] == 2) & (d["months"] <= 60)).astype(int)
    smoked_mean = float(d["ever_smoked"].mean())
    X = np.column_stack([d["age"] - 60, (d["age"] - 60) ** 2 / 20, (d["sex"] == "male").astype(float),
                         d["ever_smoked"].fillna(smoked_mean)])
    m = LogisticRegression(C=1e9, max_iter=2000).fit(X, y)
    ev = json.load(open(os.path.join(ROOT, "experiments", "navigator_evidence_result.json")))["q6"]
    out = {
        "what": "Share of US adults who died of cancer within five years of a health examination, by age, sex and smoking history",
        "outcome": "death from cancer (underlying cause, National Death Index) within 60 months",
        "data": "NHANES 1999-2014 adults 40 and over, linked mortality through 2019",
        "n": int(len(d)), "cancer_deaths": int(y.sum()), "age_range": [40, 85],
        "mean_ever_smoked": round(smoked_mean, 4),
        "coefficients": {"intercept": float(m.intercept_[0]), "age_minus_60": float(m.coef_[0][0]),
                         "age_minus_60_sq_over_20": float(m.coef_[0][1]), "male": float(m.coef_[0][2]),
                         "ever_smoked": float(m.coef_[0][3])},
        "validation_out_of_era": {"fit_on": "1999-2006", "tested_on": "2007-2014",
                                  "calibration_slope": ev["calibration_slope"], "auc": ev["auc"]},
    }
    with open(os.path.join(ROOT, "backend", "navigator_context.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"fitted on {out['n']:,} adults, {out['cancer_deaths']} cancer deaths in five years")
    for age in (45, 60, 75):
        for sex in ("female", "male"):
            for sm in (0, 1):
                z = (m.intercept_[0] + m.coef_[0][0] * (age - 60) + m.coef_[0][1] * (age - 60) ** 2 / 20
                     + m.coef_[0][2] * (sex == "male") + m.coef_[0][3] * sm)
                print(f"  {age} {sex:<6} smoked={sm}: {1000 / (1 + np.exp(-z)):.1f} per 1,000")


if __name__ == "__main__":
    main()
