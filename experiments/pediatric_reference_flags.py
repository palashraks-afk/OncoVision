"""
How wrong is an adult-style lab flag for a healthy child, and what does a
blood-count red flag cost in false alarms when cancer is this rare?

Why this exists
---------------
Reading a child's lab report is a different problem from reading an adult's, for a
reason that is easy to miss and has nothing to do with cancer: a child's normal
changes with age. A one-year-old's haemoglobin, white count and red cell size sit in
ranges that would be abnormal in an adult, and they drift through childhood and then
split by sex at puberty. Any tool that flags "high" and "low" against fixed adult
limits will mislabel healthy children, and a parent reading the result will be told
something false in one direction or the other. Dedicated paediatric reference
intervals exist (the CALIPER programme is the main one); a tool has to use them.

This measures the size of that error on a nationally representative sample, so the
need is a number and not an assertion, and then does the arithmetic for a blood
count red flag in children, where the cancers it could point to are rare.

What is measured
----------------
NHANES 2005-2018 children aged 1 to 19 with a complete blood count. Nobody is
selected for health, so the sample includes children with ordinary illness, and the
percentiles are population percentiles and not healthy-volunteer reference intervals.
That is a limit and is stated as one.

    1. Adult limits (fixed, listed in ADULT below) applied to each child: the share of
       children flagged abnormal, by age band. This is the false flag a naive
       reader produces.
    2. The same children against age- and sex-specific limits (the 2.5th and 97.5th
       percentile of children of that age and sex): by construction about 5% sit
       outside, and that is the honest baseline.
    3. A multi-lineage pattern, AGE-ADJUSTED: at least two of haemoglobin, platelets
       and absolute neutrophils below the 2.5th percentile for the child's age and
       sex. Two or more lineages falling together is the classic blood-count red flag
       for marrow failure and leukaemia, and it is the one pattern that does not
       depend on a single value being extreme. How often does it occur in ordinary
       children?
    4. The false-alarm arithmetic. Leukaemia is about 4.8 per 100,000 children a year
       in the US. Applied to unselected children, a flag with false-positive rate f
       raises about f x 100,000 / 4.8 false alarms per leukaemia case found, before
       sensitivity is even considered.

No sensitivity is measured, because NHANES has no children with a leukaemia diagnosis
and a blood count. That needs hospital data, and is the point of a chart review.

Run:  python experiments/pediatric_reference_flags.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_nhanes_mortality as fm

OUT = "experiments/pediatric_reference_flags_result.json"
CYCLES = [("2005", "D"), ("2007", "E"), ("2009", "F"), ("2011", "G"), ("2013", "H"),
          ("2015", "I"), ("2017", "J")]
LEUKAEMIA_PER_100K = 4.8       # US children 0-19, per year (SEER, 2017-2018)

# Fixed adult limits, the kind a laboratory prints when no paediatric range is applied.
# Haemoglobin is sex-specific in adults. These are widely used values and are stated
# here as the assumption they are, not as a particular laboratory's.
ADULT = {
    "hemoglobin": {"low": {"F": 12.0, "M": 13.5}, "high": {"F": 15.5, "M": 17.5}},
    "wbc": {"low": 4.5, "high": 11.0},
    "platelets": {"low": 150.0, "high": 450.0},
    "mcv": {"low": 80.0, "high": 100.0},
}
BANDS = [(1, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17), (18, 19)]
COLS = {"LBXHGB": "hemoglobin", "LBXWBCSI": "wbc", "LBXPLTSI": "platelets",
        "LBXMCVSI": "mcv", "LBDNENO": "neutrophils"}


def load():
    frames = []
    for year, suf in CYCLES:
        demo = fm.take(fm.grab_xpt(year, suf, "DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR"])
        cbc = fm.first_of(year, suf, ["CBC"])
        if demo is None or cbc is None:
            print(f"  {year}: missing, skipped")
            continue
        keep = ["SEQN"] + [c for c in COLS if c in cbc.columns]
        d = demo.merge(cbc[keep], on="SEQN")
        d = d[(d["RIDAGEYR"] >= 1) & (d["RIDAGEYR"] <= 19)]
        d = d.rename(columns={**COLS, "RIDAGEYR": "age", "RIAGENDR": "sex"})
        d["sex"] = d["sex"].map({1: "M", 2: "F"})
        d["cycle"] = f"{year}"
        frames.append(d)
        print(f"  {year}-{int(year) + 1}: {len(d):,} children aged 1-19 with a blood count")
    return pd.concat(frames, ignore_index=True)


def adult_flag(df):
    """True where any of the four values is outside the fixed adult limit."""
    f = pd.DataFrame(False, index=df.index, columns=list(ADULT))
    for sx in ("F", "M"):
        m = df["sex"] == sx
        f.loc[m, "hemoglobin"] = ((df.loc[m, "hemoglobin"] < ADULT["hemoglobin"]["low"][sx])
                                  | (df.loc[m, "hemoglobin"] > ADULT["hemoglobin"]["high"][sx]))
    for k in ("wbc", "platelets", "mcv"):
        f[k] = (df[k] < ADULT[k]["low"]) | (df[k] > ADULT[k]["high"])
    return f


def child_limits(df, col, lo=2.5, hi=97.5):
    """Age- and sex-specific percentile limits: each child against children within a
    year of the same age and of the same sex."""
    low = pd.Series(np.nan, index=df.index)
    high = pd.Series(np.nan, index=df.index)
    for sx in ("F", "M"):
        for a in range(1, 20):
            win = df[(df["sex"] == sx) & ((df["age"] - a).abs() <= 1) & df[col].notna()][col]
            if len(win) < 40:
                continue
            m = (df["sex"] == sx) & (df["age"] == a)
            low[m], high[m] = np.percentile(win, lo), np.percentile(win, hi)
    return low, high


def main():
    print("Children in NHANES 2005-2018\n")
    df = load().reset_index(drop=True)
    df = df.dropna(subset=["hemoglobin", "wbc", "platelets"])
    print(f"\n{len(df):,} children aged 1-19 with the counts needed\n")

    result = {"n": int(len(df)), "bands": {}}
    print("1. Adult limits applied to children: the share flagged abnormal")
    print(f"   {'age':<7} {'n':>6} {'any':>7} " + " ".join(f"{k:>11}" for k in ADULT))
    flags = adult_flag(df)
    for lo, hi in BANDS:
        m = (df["age"] >= lo) & (df["age"] <= hi)
        row = {k: float(flags.loc[m, k].mean()) for k in ADULT}
        any_ = float(flags.loc[m].any(axis=1).mean())
        result["bands"][f"{lo}-{hi}"] = {"n": int(m.sum()), "any_flag": round(any_, 4),
                                         **{k: round(v, 4) for k, v in row.items()}}
        print(f"   {lo}-{hi:<5} {int(m.sum()):>6} {any_:>7.0%} "
              + " ".join(f"{row[k]:>11.0%}" for k in ADULT))
    overall = float(flags.any(axis=1).mean())
    result["overall_adult_flag_rate"] = round(overall, 4)
    print(f"   {'all':<7} {len(df):>6} {overall:>7.0%}\n")

    print("2. The same children against limits for their own age and sex")
    own = pd.DataFrame(index=df.index)
    for col in ("hemoglobin", "wbc", "platelets", "mcv"):
        lo, hi = child_limits(df, col)
        own[col] = ((df[col] < lo) | (df[col] > hi)).where(lo.notna())
    own_any = float(own.fillna(False).astype(bool).any(axis=1).mean())
    result["own_age_flag_rate_any_of_four"] = round(own_any, 4)
    print(f"   share outside their own age and sex limits on at least one of four values: "
          f"{own_any:.1%} (about 5% per value is expected by construction)\n")

    print("3. Age-adjusted multi-lineage pattern: two or more of haemoglobin, platelets, "
          "absolute neutrophils below the 2.5th percentile for age and sex")
    low = pd.DataFrame(index=df.index)
    for col in ("hemoglobin", "platelets", "neutrophils"):
        l, _ = child_limits(df, col, lo=2.5)
        low[col] = (df[col] < l).where(l.notna() & df[col].notna())
    complete = low.notna().all(axis=1)
    n_low = low[complete].astype(bool).sum(axis=1)
    multi = float((n_low >= 2).mean())
    all3 = float((n_low >= 3).mean())
    result["multilineage_two_or_more"] = round(multi, 5)
    result["multilineage_all_three"] = round(all3, 5)
    result["multilineage_n"] = int(complete.sum())
    print(f"   {int(complete.sum()):,} children have all three counts")
    print(f"   two or more lineages low: {multi:.2%}   all three low: {all3:.2%}\n")

    print("4. False-alarm arithmetic for an unselected child population")
    per_case = 100_000 / LEUKAEMIA_PER_100K
    result["children_per_leukaemia_case_per_year"] = round(per_case)
    alarms = {}
    for name, f in (("any adult-limit flag", overall), ("two-or-more lineage pattern", multi)):
        alarms[name] = round(f * per_case, 1)
        print(f"   {name:<30} false positive rate {f:>6.2%} -> about {f * per_case:>8,.0f} "
              f"false alarms per leukaemia case, before sensitivity")
    result["false_alarms_per_case"] = alarms
    print(f"\n   (about {per_case:,.0f} children for each new leukaemia case in a year)")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
