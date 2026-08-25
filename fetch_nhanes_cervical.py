"""
An external cohort for the cervical panel, from NHANES.

Why this exists
---------------
The cervical panel trains on 858 women at one hospital in Caracas and had no
external test at all. "Single centre, no external cohort" was listed as one of
its two open limitations, so this builds the missing cohort rather than
restating the gap.

NHANES collects the same risk history the Caracas cohort does: lifetime sexual
partners, age at first intercourse, pregnancies, tobacco, hormonal
contraceptives, and diagnosed sexually transmitted infections. It also records
cervical cancer. So a model fitted in Venezuela can be asked to rank women in
the United States, which is a real transfer test across country, decade,
language and data collection method.

What this is honestly NOT
-------------------------
The targets are not identical and pretending otherwise would be the same
mistake this project has corrected elsewhere.

    Caracas   a POSITIVE CERVICAL BIOPSY. Mostly pre-cancer, CIN, found by
              looking. 6.4 percent of the cohort.
    NHANES    SELF-REPORTED CERVICAL CANCER, ever. Invasive disease, recalled
              in an interview, and far rarer.

So this does not test "does the panel reproduce its AUC". It tests the weaker
and still worthwhile question: does the risk ordering this model learned in
Caracas carry any signal about who develops cervical cancer in a nationally
representative US sample? A model that has learned real HPV exposure risk
should transfer. A model that has memorised one clinic should not.

Because invasive cervical cancer is rare and self-report is noisy, the expected
ceiling here is well below the internal number. That is a property of the
target, not of the model, and the comparison is framed as such.

Two features cannot be mapped
-----------------------------
NHANES has no IUD variable, so iud and iud_years arrive missing and are filled
with the training medians, exactly as the live application does when a user
leaves a field blank. 13 of 15 features map directly.

Run:  python fetch_nhanes_cervical.py
"""

import os

import numpy as np
import pandas as pd

from fetch_nhanes_screening import grab, CYCLES

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "nhanes_cervical_external.csv")

# Verified against the MCQ_D codebook rather than guessed. An earlier version of
# this project invented these codes and reported a cancer count that was wrong
# by more than an order of magnitude.
CERVIX = 15

# Sexually transmitted infections NHANES asks about directly.
STI_VARS = {
    "SXQ260": "genital herpes",
    "SXQ265": "genital warts",
    "SXQ270": "gonorrhea",
    "SXQ272": "chlamydia",
}
# Genital warts are caused by HPV. It is the only recorded HPV diagnosis NHANES
# carries for these cycles, and it matches how the Caracas cohort records
# STDs:HPV, which is likewise a documented diagnosis rather than a DNA assay.
HPV_VAR = "SXQ265"


def _col(frame, name):
    """A column as a Series, or an all-missing Series if the cycle lacks it.

    NHANES renames variables between cycles, and DataFrame.get returns a bare
    NaN scalar for a missing name rather than None, which silently poisons
    anything downstream that expects a Series.
    """
    if name in frame.columns:
        got = frame[name]
        # A duplicated name after a merge comes back as a DataFrame.
        if isinstance(got, pd.DataFrame):
            got = got.iloc[:, 0]
        return got
    return pd.Series(np.nan, index=frame.index)


def _yes(frame, name):
    """NHANES yes/no: 1 yes, 2 no, 7 refused, 9 don't know."""
    s = pd.to_numeric(_col(frame, name), errors="coerce")
    return s.where(s.isin([1, 2])).map({1: 1.0, 2: 0.0})


def _clean(frame, name, lo, hi):
    """Drop the 7777/9999 style refusal and don't-know sentinels."""
    s = pd.to_numeric(_col(frame, name), errors="coerce")
    return s.where((s >= lo) & (s <= hi))


def fetch_nhanes_cervical():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for year, suf, label in CYCLES:
        demo, mcq = grab(year, suf, "DEMO"), grab(year, suf, "MCQ")
        sxq, rhq, smq = grab(year, suf, "SXQ"), grab(year, suf, "RHQ"), grab(year, suf, "SMQ")
        if any(x is None for x in (demo, mcq, sxq, rhq, smq)):
            print(f"  {label}: a required file is missing, skipped")
            continue

        d = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        for extra in (mcq, sxq, rhq, smq):
            d = d.merge(extra, on="SEQN", how="left", suffixes=("", "_dup"))

        # Women only. RIAGENDR: 1 male, 2 female.
        d = d[pd.to_numeric(d["RIAGENDR"], errors="coerce") == 2]
        d = d[pd.to_numeric(d["RIDAGEYR"], errors="coerce") >= 18]

        out = pd.DataFrame(index=d.index)
        out["age"] = pd.to_numeric(d["RIDAGEYR"], errors="coerce")
        out["gender"] = 0

        out["sexual_partners"] = _clean(d, "SXQ101", 0, 200)
        out["first_intercourse_age"] = _clean(d, "SXD031", 8, 60)
        out["pregnancies"] = _clean(d, "RHQ160", 0, 20)

        # Tobacco. SMQ020 ever smoked 100 cigarettes, SMQ040 smoke now
        # (1 every day, 2 some days, 3 not at all), SMD030 age started.
        ever = _yes(d, "SMQ020")
        now = pd.to_numeric(_col(d, "SMQ040"), errors="coerce")
        started = _clean(d, "SMD030", 5, 80)
        per_day = _clean(d, "SMD650", 0, 100)

        current = now.isin([1, 2])
        out["smokes"] = np.where(current, 1.0, np.where(ever.notna(), 0.0, np.nan))

        # Current smokers: years since starting. Former smokers: subtract the
        # time since quitting. Never smokers: zero.
        quit_q = _clean(d, "SMQ050Q", 0, 80)
        quit_u = pd.to_numeric(_col(d, "SMQ050U"), errors="coerce")
        # SMQ050U units: 1 days, 2 weeks, 3 months, 4 years.
        since_quit = np.select(
            [quit_u == 1, quit_u == 2, quit_u == 3, quit_u == 4],
            [quit_q / 365.0, quit_q / 52.0, quit_q / 12.0, quit_q],
            default=np.nan,
        )
        smoked = out["age"] - started
        former = (ever == 1) & (now == 3)
        years = np.where(current, smoked,
                 np.where(former, smoked - pd.Series(since_quit, index=out.index), 0.0))
        years = pd.Series(years, index=out.index)
        years = years.where(years >= 0, 0.0)
        years = years.where(ever.notna(), np.nan)
        out["smoking_years"] = years
        out["smoking_packyears"] = years * (per_day / 20.0)
        out.loc[ever == 0, "smoking_packyears"] = 0.0

        # Hormonal contraceptives. RHQ460Q is a duration with RHQ460U as unit
        # (1 months, 2 years) in these cycles.
        out["hormonal_contraceptives"] = _yes(d, "RHQ420")
        hc_q = _clean(d, "RHQ460Q", 0, 600)
        hc_u = pd.to_numeric(_col(d, "RHQ460U"), errors="coerce")
        hc_years = np.select([hc_u == 1, hc_u == 2], [hc_q / 12.0, hc_q], default=np.nan)
        hc_years = pd.Series(hc_years, index=out.index)
        hc_years = hc_years.where(hc_years.notna(),
                                  np.where(out["hormonal_contraceptives"] == 0, 0.0, np.nan))
        out["hormonal_contraceptives_years"] = hc_years

        # NHANES carries no IUD question. Left missing on purpose so the model
        # fills it with its training median, which is what the live application
        # does for any field a user leaves blank.
        out["iud"] = np.nan
        out["iud_years"] = np.nan

        sti = pd.DataFrame({v: _yes(d, v) for v in STI_VARS if v in d.columns})
        if sti.empty:
            continue
        out["stds_number"] = sti.sum(axis=1, min_count=1)
        out["stds"] = (out["stds_number"] > 0).astype(float)
        out.loc[out["stds_number"].isna(), "stds"] = np.nan
        out["stds_diagnoses"] = out["stds_number"]
        out["stds_hpv"] = _yes(d, HPV_VAR)

        # Target: ever told you had cervical cancer, across the four slots.
        told = _yes(d, "MCQ220")
        sites = [pd.to_numeric(_col(d, f"MCQ230{L}"), errors="coerce")
                 for L in "ABCD" if f"MCQ230{L}" in d.columns]
        has_cervix = pd.concat([s.eq(CERVIX) for s in sites], axis=1).any(axis=1) if sites \
            else pd.Series(False, index=out.index)
        out["cervical_cancer"] = np.where(has_cervix, 1.0,
                                          np.where(told == 0, 0.0, np.nan))

        out["cycle"] = label
        # A row is only usable if the outcome is known and at least one of the
        # sexual-history features was actually answered.
        core = ["sexual_partners", "first_intercourse_age", "pregnancies"]
        out = out[out["cervical_cancer"].notna() & out[core].notna().any(axis=1)]
        frames.append(out)
        print(f"  {label}: {len(out)} women, {int(out.cervical_cancer.sum())} cervical cancer")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)

    n, pos = len(df), int(df["cervical_cancer"].sum())
    print(f"\nwrote {OUT}")
    print(f"  {n} women, {pos} with cervical cancer ({pos / n:.3%})")
    print("  13 of 15 features mapped; iud and iud_years are not collected by NHANES")
    return df


if __name__ == "__main__":
    fetch_nhanes_cervical()
