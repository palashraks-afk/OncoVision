"""
NHANES 2003-2004 and 2005-2006 hip-accelerometer files (minute-level activity counts, 7 days) reduced to a few
per-person daily summaries of the kind a phone or watch records for free.

Non-wear: a run of 60 or more minutes of zero counts (allowing up to 2 minutes of 1-100 counts), as in Troiano
et al. 2008. A valid day has at least 10 hours of wear; a person needs at least 4 valid days.

Per person (averaged over valid days):
  tac            total activity counts per day (log-transformed later)
  mvpa_min       minutes at 2,020 counts or more (moderate-to-vigorous)
  sed_frac       share of wear minutes under 100 counts (sedentary)
  frag           active-to-sedentary transition probability (how broken up activity is)
  wear_hours     hours of wear per valid day

Run:  python fetch_nhanes_activity.py      reads data/audit/pax/paxraw_{c,d}.zip, writes data/nhanes_activity.csv
"""

import os
import zipfile

import numpy as np
import pandas as pd

PAX = os.path.join("data", "audit", "pax")
OUT = os.path.join("data", "nhanes_activity.csv")


def nonwear(c):
    """Boolean non-wear mask for one person-day series of counts (Troiano 2008 rule)."""
    n = len(c)
    nw = np.zeros(n, bool)
    i = 0
    while i < n:
        if c[i] == 0:
            j, spikes = i, 0
            k = i
            while k < n and (c[k] == 0 or (c[k] <= 100 and spikes < 2)):
                if c[k] > 0:
                    spikes += 1
                k += 1
                if c[k - 1] == 0:
                    j = k
            if j - i >= 60:
                nw[i:j] = True
            i = max(j, i + 1)
        else:
            i += 1
    return nw


def summarise(seqn, day, counts):
    out = []
    for d in np.unique(day):
        c = counts[day == d]
        nw = nonwear(c)
        wear = ~nw
        if wear.sum() < 600:
            continue
        cw = c[wear]
        act = cw >= 100
        trans = ((act[:-1]) & (~act[1:])).sum()
        out.append((cw.sum(), (cw >= 2020).sum(), (~act).mean(), trans / max(act[:-1].sum(), 1), wear.sum() / 60))
    if len(out) < 4:
        return None
    a = np.array(out)
    return {"SEQN": int(seqn), "valid_days": len(out), "tac": a[:, 0].mean(), "mvpa_min": a[:, 1].mean(), "sed_frac": a[:, 2].mean(), "frag": a[:, 3].mean(), "wear_hours": a[:, 4].mean()}


def main():
    rows = []
    for s in ("c", "d"):
        z = zipfile.ZipFile(os.path.join(PAX, f"paxraw_{s}.zip"))
        name = z.namelist()[0]
        print("reading", name, flush=True)
        with z.open(name) as fh:
            rd = pd.read_sas(fh, format="xport", chunksize=2_000_000)
            carry = None
            for chunk in rd:
                chunk = chunk[["SEQN", "PAXDAY", "PAXINTEN", "PAXSTAT", "PAXCAL"]]
                if carry is not None:
                    chunk = pd.concat([carry, chunk])
                last = chunk["SEQN"].iloc[-1]
                carry = chunk[chunk["SEQN"] == last]
                done = chunk[chunk["SEQN"] != last]
                for sq, g in done.groupby("SEQN", sort=False):
                    if (g["PAXCAL"] != 1).any() or (g["PAXSTAT"] != 1).any():
                        continue                      # monitor not in calibration or data unreliable
                    r = summarise(sq, g["PAXDAY"].to_numpy(), g["PAXINTEN"].fillna(0).to_numpy())
                    if r:
                        r["cycle"] = "2003-2004" if s == "c" else "2005-2006"
                        rows.append(r)
                print(f"  {len(rows):,} people summarised", flush=True)
            if carry is not None and len(carry):
                g = carry
                if not ((g["PAXCAL"] != 1).any() or (g["PAXSTAT"] != 1).any()):
                    r = summarise(g["SEQN"].iloc[0], g["PAXDAY"].to_numpy(), g["PAXINTEN"].fillna(0).to_numpy())
                    if r:
                        r["cycle"] = "2003-2004" if s == "c" else "2005-2006"
                        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
