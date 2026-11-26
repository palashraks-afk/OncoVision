"""
External check of the MCED paper's yield assumption against the NHS-Galleri trial (PMID 42773209).
Pass rules: docs/MCED_LAW_PREREG.md, section "External check against the NHS-Galleri trial" (committed first).

Run:  python experiments/mced_nhsgalleri_check.py
"""

import json
import os

import numpy as np

from cancer_age import Fit
from mced_law import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20261125)
TRIAL = {"age 70-79 vs 50-59": {"ratio": 0.78 / 0.18, "rates": [(0.18, 0.14, 0.22), (0.78, 0.71, 0.85)]},
         "men vs women": {"ratio": 0.58 / 0.38, "rates": [(0.38, 0.34, 0.42), (0.58, 0.53, 0.63)]}}


def trial_ci(rates):
    (a, al, ah), (b, bl, bh) = rates
    se = np.hypot(np.log(ah / al) / 3.92, np.log(bh / bl) / 3.92)
    return [float(b / a * np.exp(-1.96 * se)), float(b / a * np.exp(1.96 * se))]


def main():
    d = load()
    p_model = Fit(d[d["year"] <= 2002].reset_index(drop=True), 1)
    te = d[(d["year"] >= 2005) & d["weight"].notna()].reset_index(drop=True)
    y, w, age, male = te["y"].to_numpy(float), te["weight"].to_numpy(float), te["age"].to_numpy(), te["male"].to_numpy()
    root = np.sqrt(p_model.predict(te))
    groups = {"age 70-79 vs 50-59": ((age >= 50) & (age <= 59), (age >= 70) & (age <= 78)),
              "men vs women": ((age >= 50) & (age <= 78) & (male == 0), (age >= 50) & (age <= 78) & (male == 1))}
    out = {}
    for name, (g0, g1) in groups.items():
        lo, hi = trial_ci(TRIAL[name]["rates"])
        row = {"trial_ratio": round(TRIAL[name]["ratio"], 2), "trial_ci": [round(lo, 2), round(hi, 2)]}
        for label, v in (("proportional_to_death_rate", y), ("square_root_sensitivity", root)):
            def ratio(i, v=v):
                a, b = g0[i], g1[i]
                return (v[i][b] * w[i][b]).sum() / w[i][b].sum() / ((v[i][a] * w[i][a]).sum() / w[i][a].sum())
            point = float(ratio(np.arange(len(y))))
            ci = np.percentile([ratio(RNG.integers(0, len(y), len(y))) for _ in range(1000)], [2.5, 97.5])
            row[label] = {"ratio": round(point, 2), "ci": [round(float(x), 2) for x in ci], "pass": bool(ci[0] <= hi and ci[1] >= lo)}
        row["cancer_deaths"] = [int(y[g0].sum()), int(y[g1].sum())]
        out[name] = row
        print(name, json.dumps(row))
    json.dump(out, open(os.path.join(ROOT, "experiments", "mced_nhsgalleri_check_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
