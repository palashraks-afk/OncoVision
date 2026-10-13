"""
Exports the five-year cancer-death model used in experiments/mced_law.py as a lookup table for the
in-browser multi-cancer test checker (frontend/src/app/multicancer/model.json), so the page needs no server.

Grid: sex x age 35-84 x smoking (never; former by years since quitting 0-50 in 5s; current by cigarettes a day
5-40 in 5s). Also exports the anchor that turns predicted risk into expected multi-cancer test results
(PATHFINDER 2: detection 0.54% in adults 50+, specificity 99.64%) and the risk distribution of the people the
enacted law covers in 2029, for a percentile.

Run:  python experiments/export_mced_app.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from cancer_age import Fit
from mced_law import DETECT, PRICE, SPEC, load

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "frontend", "src", "app", "multicancer", "model.json")


def main():
    d = load()
    tr = d[d["year"] <= 2002].reset_index(drop=True)
    te = d[(d["year"] >= 2005) & d["weight"].notna()].reset_index(drop=True)
    m = Fit(tr, 1)
    te["p"] = m.predict(te)
    w, age, p, mc = te["weight"].to_numpy(), te["age"].to_numpy(), te["p"].to_numpy(), te["medicare"].to_numpy()
    a50 = age >= 50
    k = DETECT / float((p[a50] * w[a50]).sum() / w[a50].sum())
    cov = (age == 65) | ((age >= 50) & (age < 65) & (mc == 1))     # enacted law, 2029
    o = np.argsort(p[cov])
    cw = np.cumsum(w[cov][o]) / w[cov].sum()
    pct = [float(np.interp(q, cw, p[cov][o])) for q in np.linspace(0, 1, 101)]

    rows = []
    for male in (0.0, 1.0):
        for a in range(35, 85):
            for kind, vals in (("never", [0]), ("former", list(range(0, 51, 5))), ("current", list(range(5, 41, 5)))):
                for v in vals:
                    rows.append({"age": a, "male": male, "smoke_status": {"never": 0, "former": 1, "current": 2}[kind],
                                 "cigs_per_day": v if kind == "current" else 0.0, "quit_years": v if kind == "former" else 0.0,
                                 "bmi": 27.0, "kind": kind, "v": v})
    g = pd.DataFrame(rows)
    g["p"] = m.predict(g)
    table = {}
    for r in g.itertuples():
        table.setdefault(("m" if r.male else "f"), {}).setdefault(str(r.age), {})[f"{r.kind}:{r.v}"] = round(float(r.p), 5)
    out = {"source": "NHIS 1997-2002 fit, five-year cancer death, age spline + sex + smoking (experiments/mced_law.py)",
           "k": round(k, 4), "specificity": SPEC, "price": PRICE, "covered_percentiles": [round(x, 5) for x in pct], "risk": table}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT, f"k={k:.3f}", "sizes", {s: len(v) for s, v in table.items()})


if __name__ == "__main__":
    main()
