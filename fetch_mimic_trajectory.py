"""
Blood counts as a TRAJECTORY: how a person's own numbers moved before diagnosis.

The idea
--------
Every cohort in this project gives one blood draw per person, so every panel
answers "does this number look wrong today". A haemoglobin of 13 says little. A
haemoglobin that was 15 two years ago, then 14, now 13, says something quite
different, and slow blood loss from a right-sided bowel tumour looks exactly
like that. ColonFlag, the one commercially deployed model built on this idea,
reports AUCs of 0.74 to 0.82 on repeat blood counts in ordinary care.

Nothing public that this project has found pairs repeat blood counts with a
later cancer diagnosis -- except hospital records. MIMIC-IV has both: every lab
result with a timestamp, and diagnosis codes per admission. This builds the
cohort from it.

Two versions of the same file
-----------------------------
    demo    100 patients, openly downloadable, no credentials. Far too small
            for a result. It exists so the pipeline is finished, tested and
            reviewable BEFORE anyone waits on access.
    full    MIMIC-IV proper, ~300,000 patients. Free, but it requires a
            PhysioNet credentialed account and CITI "Data or Specimens Only
            Research" training. Point --dir at the unpacked hosp/ folder.

The traps this file is built to avoid
-------------------------------------
PREVALENT CANCER. Most cancer codes in these records are "personal history of
malignant neoplasm" -- someone treated years ago. Their bloodwork reflects
treatment, not an undetected tumour, and counting them as cases is the exact
error that withdrew three panels here. Only a FIRST active malignancy code
counts, and anyone carrying a history code before that point is dropped
entirely rather than used as either a case or a control.

LOOKING AFTER THE FACT. Labs drawn once someone is being worked up for cancer
are not early detection: they are the investigation itself. Every value used
must predate the first cancer code by at least LEAD_DAYS, which defaults to 90.

THE SICK-CONTROL PROBLEM, stated not solved. These are hospital patients. The
controls are people admitted for something else, so this cohort answers "among
people in hospital, whose blood count drifted like a cancer patient's", which
is a narrower question than screening the street. It is written on the cohort.

Run:  python fetch_mimic_trajectory.py            # demo, downloads itself
      python fetch_mimic_trajectory.py --dir path/to/mimiciv/3.1/hosp
"""

import argparse
import io
import os
import ssl
import sys
import urllib.request

import numpy as np
import pandas as pd

DEMO = "https://physionet.org/files/mimic-iv-demo/2.2/hosp/"
OUT = os.path.join("data", "mimic_trajectory.csv")

LEAD_DAYS = 90          # nothing within 90 days of the first cancer code is used
MIN_POINTS = 3          # a trajectory needs at least three readings to have a shape
LOOKBACK_DAYS = 1825    # five years of history before the index date

# The analytes a complete blood count prints, by their MIMIC labels.
CBC = {
    "Hemoglobin": "hemoglobin", "Hematocrit": "hematocrit", "MCV": "mcv",
    "RDW": "rdw", "Platelet Count": "platelets", "White Blood Cells": "wbc",
    "Red Blood Cells": "rbc", "MCH": "mch", "MCHC": "mchc",
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def load(directory, name):
    """One table, from a local MIMIC folder or the open demo."""
    if directory:
        for cand in (f"{name}.csv.gz", f"{name}.csv"):
            path = os.path.join(directory, cand)
            if os.path.exists(path):
                return pd.read_csv(path, low_memory=False)
        raise FileNotFoundError(f"{name} not found in {directory}")
    req = urllib.request.Request(DEMO + f"{name}.csv.gz",
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=_ctx, timeout=600).read()
    return pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False)


def is_active_cancer(code, version):
    """An active malignancy, not a history of one and not a benign growth."""
    c = str(code).strip().upper()
    if version == 10:
        return c.startswith("C") and not c.startswith("C4A")
    return c[:3].isdigit() and 140 <= int(c[:3]) <= 208


def is_history_of_cancer(code, version):
    c = str(code).strip().upper()
    return c.startswith("Z85") if version == 10 else c.startswith("V10")


def trajectory_features(g):
    """What a run of one analyte did: where it is, and where it was going.

    The level is what a single-draw panel already sees. The rest is what only
    repeat measurement can say -- how far the value has moved from this
    person's own baseline, and how fast.
    """
    g = g.sort_values("charttime")
    v = g["valuenum"].to_numpy(dtype=float)
    t = (g["charttime"] - g["charttime"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    out = {"last": v[-1], "baseline": v[0], "delta": v[-1] - v[0],
           "rel_delta": (v[-1] - v[0]) / v[0] if v[0] else np.nan,
           "min": v.min(), "max": v.max(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0,
           "n": len(v), "span_days": t[-1] - t[0]}
    # Slope per year, which is the number a clinician reads off a graph.
    out["slope_per_year"] = (np.polyfit(t, v, 1)[0] * 365.0
                             if len(v) > 2 and t[-1] > t[0] else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="a local MIMIC-IV hosp/ folder")
    ap.add_argument("--lead-days", type=int, default=LEAD_DAYS)
    args = ap.parse_args()
    where = args.dir or "the open 100-patient demo"
    print(f"Building blood-count trajectories from {where}\n")

    lab = load(args.dir, "labevents")
    items = load(args.dir, "d_labitems")
    dx = load(args.dir, "diagnoses_icd")
    adm = load(args.dir, "admissions")
    pat = load(args.dir, "patients")

    lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    lab = lab.merge(items[["itemid", "label"]], on="itemid", how="left")
    lab = lab[lab["label"].isin(CBC) & lab["valuenum"].notna() & lab["charttime"].notna()]
    lab["analyte"] = lab["label"].map(CBC)

    dx = dx.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
    dx["active"] = [is_active_cancer(c, v) for c, v in zip(dx["icd_code"], dx["icd_version"])]
    dx["history"] = [is_history_of_cancer(c, v) for c, v in zip(dx["icd_code"], dx["icd_version"])]

    first_cancer = (dx[dx["active"]].groupby("subject_id")["admittime"].min()
                    .rename("cancer_date"))
    first_history = (dx[dx["history"]].groupby("subject_id")["admittime"].min()
                     .rename("history_date"))
    print(f"  {dx.subject_id.nunique():,} patients with any diagnosis code")
    print(f"  {len(first_cancer):,} with an active malignancy code")
    print(f"  {len(first_history):,} carry a 'personal history of cancer' code")

    people = pd.DataFrame({"subject_id": lab["subject_id"].unique()})
    people = (people.merge(first_cancer, on="subject_id", how="left")
                    .merge(first_history, on="subject_id", how="left")
                    .merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"))

    # Anyone whose cancer history predates their first active code is a
    # survivor, not a new case, and is removed from both sides.
    survivor = people["history_date"].notna() & (
        people["cancer_date"].isna() | (people["history_date"] <= people["cancer_date"]))
    dropped = int(survivor.sum())
    people = people[~survivor].copy()

    # The index date: for a case, the first cancer code. For a control, their
    # last lab, so a control's history is as long as a case's can be.
    last_lab = lab.groupby("subject_id")["charttime"].max().rename("last_lab")
    people = people.merge(last_lab, on="subject_id", how="left")
    people["is_case"] = people["cancer_date"].notna()
    people["index_date"] = np.where(people["is_case"], people["cancer_date"], people["last_lab"])
    people["index_date"] = pd.to_datetime(people["index_date"])

    cutoff = people.set_index("subject_id")["index_date"] - pd.to_timedelta(args.lead_days, "D")
    lab = lab.join(cutoff.rename("cutoff"), on="subject_id")
    start = cutoff - pd.to_timedelta(LOOKBACK_DAYS, "D")
    lab = lab.join(start.rename("window_start"), on="subject_id")
    usable = lab[(lab["charttime"] <= lab["cutoff"]) & (lab["charttime"] >= lab["window_start"])]
    print(f"  {dropped:,} survivors dropped; {usable.subject_id.nunique():,} patients have "
          f"blood counts at least {args.lead_days} days before their index date")

    rows = []
    for (sid, analyte), g in usable.groupby(["subject_id", "analyte"]):
        if len(g) < MIN_POINTS:
            continue
        f = trajectory_features(g)
        rows.append({"subject_id": sid, **{f"{analyte}_{k}": v for k, v in f.items()}})
    if not rows:
        print("\n  No patient has enough repeat measurements in the window. On the "
              "100-patient demo that is the expected outcome: the pipeline is what "
              "is being checked here, not the result.")
    feat = (pd.DataFrame(rows).groupby("subject_id").first().reset_index()
            if rows else pd.DataFrame({"subject_id": []}))

    out = people.merge(feat, on="subject_id", how="inner")
    out["age"] = pd.to_numeric(out["anchor_age"], errors="coerce")
    out["gender"] = (out["gender"].astype(str).str.upper() == "M").astype(float)
    out["cancer"] = out["is_case"].astype(int)
    out = out.drop(columns=[c for c in ("anchor_age", "is_case", "history_date",
                                        "last_lab") if c in out.columns])
    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT, index=False)

    n, pos = len(out), int(out["cancer"].sum()) if len(out) else 0
    print(f"\nwrote {OUT}")
    print(f"  {n:,} patients, {pos} with a first cancer diagnosis after the window "
          f"({pos / n:.1%})" if n else "  0 patients")
    traj_cols = [c for c in out.columns if c.endswith(("_slope_per_year", "_delta"))]
    print(f"  {len(traj_cols)} trajectory features, and the same analytes' latest values")
    if n and pos < 96:
        print(f"\n  {pos} cases is below the ~96 event floor this project uses, so nothing "
              f"here is a result. Run it again on full MIMIC-IV:")
        print(f"      python fetch_mimic_trajectory.py --dir path/to/mimiciv/hosp")
        print(f"  which needs a PhysioNet credentialed account and CITI training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
