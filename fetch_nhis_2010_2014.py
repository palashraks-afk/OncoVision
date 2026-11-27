"""
NHIS sample adults 2010-2014 (replication years for docs/MCED_LAW_PREREG.md), built with exactly the same code as
the 1997-2009 cohort (fetch_nhis_cancer_risk.py). Writes data/nhis_2010_2014.csv.gz.

Run:  python fetch_nhis_2010_2014.py
"""
import os

import numpy as np
import pandas as pd

from fetch_nhis_cancer_risk import clean, one_year

OUT = os.path.join("data", "nhis_2010_2014.csv.gz")


def main():
    frames = []
    for y in range(2010, 2015):
        cache = os.path.join("data", "audit", "nhis_cache", f"{y}.csv.gz")
        if os.path.exists(cache):
            frames.append(pd.read_csv(cache))
            continue
        try:
            c = clean(one_year(y))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            c.to_csv(cache, index=False, compression="gzip")
            frames.append(c)
        except Exception as e:  # a year that cannot be read is left out and said so
            print(f"  {y}: SKIPPED ({type(e).__name__}: {e})", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["age"] >= 18) & (d["elig"] == 1)]
    d["months"] = np.where(d["mortstat"] == 1, (d["tdeath"] - d["t0"]) * 12, (2020.0 - d["t0"]) * 12)
    d.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT}: {len(d):,} adults; years {sorted(d['year'].unique().tolist())}")


if __name__ == "__main__":
    main()
