"""
How many symptom-free adults would the navigator's lab-driven rules alert?

Why this matters
----------------
The navigator applies guideline rules that were written for people who walked into a
surgery with a symptom. Some of those rules need no symptom at all once a lab result is
in hand: a high platelet count at 40 or over, anaemia at 60 or over, small red cells in
anaemia. If a patient uploads a routine blood count and ticks nothing, those rules can
still fire. Whether that is a useful nudge or a wall of false alarms is an empirical
question, and it decides whether the tool is usable at all.

This runs the REAL engine, not a re-implementation, on every NHANES adult 40 or over with
a blood count, with no symptoms entered, and counts what it says.

What is measured
----------------
    alert burden     the share of adults in each age band told to talk to a doctor
                     soon, to raise something, or to mention something, by labs alone
    which rules      what drives each alert
    what happened    for context only: the share who died of cancer within ten years in
                     each alert group, against those not alerted

What it does NOT show
---------------------
Whether an alert is correct. NHANES has no symptoms and records death, not diagnosis, so
the last column describes who these people are and is not a validation of any rule. The
guideline rules were derived from symptomatic patients and this applies them to people
who have none, which is exactly the extrapolation a patient-facing tool invites and the
reason the burden has to be known before one is released.

Ferritin and CA-125 are not in this cohort. Iron deficiency is therefore assumed from small
red cells (MCV under 80), which the engine does and discloses, and which over-alerts
relative to a ferritin-confirmed definition.

Run:  python experiments/navigator_lab_trigger_burden.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import navigator as nav  # noqa: E402

OUT = "experiments/navigator_lab_trigger_burden_result.json"
DATA = "data/nhanes_wide.csv.gz"
BANDS = [(40, 49), (50, 59), (60, 69), (70, 79), (80, 120)]
HORIZON = 120


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0, c - h)), 4), round(float(min(1, c + h)), 4)]


def main():
    cols = ["age", "gender", "LBXHGB", "LBXMCVSI", "LBXPLTSI", "LBXWBCSI",
            "died", "ucod_leading", "followup_months"]
    df = pd.read_csv(DATA, usecols=cols)
    df = df[df["age"] >= 40].dropna(subset=["LBXHGB", "LBXPLTSI"]).reset_index(drop=True)
    print(f"{len(df):,} adults aged 40 and over with a haemoglobin and platelet count "
          f"(NHANES 1999-2008)\n")

    states, rule_hits, skipped = [], [], 0
    for r in df.itertuples(index=False):
        labs = {"hemoglobin": r.LBXHGB, "platelets": r.LBXPLTSI}
        if pd.notna(r.LBXMCVSI):
            labs["mcv"] = r.LBXMCVSI
        if pd.notna(r.LBXWBCSI):
            labs["wbc"] = r.LBXWBCSI
        req = {"age": float(r.age), "sex": "male" if r.gender == 1 else "female", "labs": labs}
        try:
            out = nav.evaluate(req)
        except ValueError:
            skipped += 1
            states.append(None)
            rule_hits.append([])
            continue
        states.append(out["state"])
        rule_hits.append([m["id"] for m in out["matches"]])
    df["state"] = states
    df["rules"] = rule_hits
    df = df[df["state"].notna()].reset_index(drop=True)
    print(f"{len(df):,} evaluated, {skipped} skipped for an impossible value\n")

    months, died = df["followup_months"], df["died"] == 1
    cancer_in = died & (df["ucod_leading"] == 2) & (months <= HORIZON)
    known = (died & (months <= HORIZON)) | (months >= HORIZON)

    result = {"n": int(len(df)), "bands": {}, "assumption": "iron deficiency inferred from MCV < 80 (no ferritin in cohort)"}
    print(f"{'age':<8}{'n':>7}{'talk soon':>11}{'worth raising':>15}{'mention':>9}{'any alert':>11}")
    for lo, hi in BANDS:
        m = (df["age"] >= lo) & (df["age"] <= hi)
        n = int(m.sum())
        row = {"n": n}
        for st in ("talk_soon", "worth_raising", "mention"):
            k = int((df.loc[m, "state"] == st).sum())
            row[st] = {"n": k, "share": round(k / n, 4), "ci": wilson(k, n)}
        any_k = int((df.loc[m, "state"].isin(["talk_soon", "worth_raising", "mention"])).sum())
        row["any_alert"] = {"n": any_k, "share": round(any_k / n, 4), "ci": wilson(any_k, n)}
        result["bands"][f"{lo}-{hi if hi < 120 else '+'}"] = row
        print(f"{lo}-{hi if hi < 120 else '+':<5}{n:>7}"
              f"{row['talk_soon']['share']:>11.1%}{row['worth_raising']['share']:>15.1%}"
              f"{row['mention']['share']:>9.1%}{row['any_alert']['share']:>11.1%}")

    flat = pd.Series([r for rs in df["rules"] for r in rs]).value_counts()
    result["rule_counts"] = {k: int(v) for k, v in flat.items()}
    print("\nWhat fires, across all adults 40+ (a person can trigger several):")
    for k, v in flat.items():
        print(f"   {k:<28}{int(v):>7,}  ({v / len(df):.1%} of adults)")

    # Context: who are the alerted people. Not a validation.
    print("\nFor context only: ten-year cancer death among those with a known outcome")
    ctx = {}
    for label, mask in (("talk soon", df["state"] == "talk_soon"),
                        ("worth raising or mention", df["state"].isin(["worth_raising", "mention"])),
                        ("no alert", df["state"].isin(["nothing_meets", "need_info"]))):
        sel = mask & known
        n, k = int(sel.sum()), int((cancer_in & sel).sum())
        ctx[label] = {"n": n, "cancer_deaths": k, "share": round(k / max(n, 1), 4), "ci": wilson(k, n)}
        print(f"   {label:<26} n={n:>6,}  cancer deaths {k:>4}  ({k / max(n, 1):.1%}, "
              f"95% CI {ctx[label]['ci'][0]:.1%} to {ctx[label]['ci'][1]:.1%})")
    result["context_ten_year_cancer_death"] = ctx
    result["note"] = ("Descriptive. NHANES has no symptoms and records death rather than diagnosis, so this "
                      "does not validate any rule.")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
