"""
Why the liver panel scores 0.442 in Germany. It is not a bug.

The claim this replaces
-----------------------
The documentation carried "Germany 0.442, below chance, probably a units or
encoding mismatch, unresolved" for a long time. Two things were wrong with that.
It was never investigated, and the guess was incorrect.

What was checked
----------------
First suspicion: ucimlrepo mangles UCI 571. It returns the columns in a
different order from the published CSV, with ALT moved to the end. So the raw
file was downloaded from archive.ics.uci.edu and compared value by value.
ucimlrepo is FINE. The values and the per-category medians match the original
exactly. That suspicion was wrong.

Second suspicion: a unit mismatch. Also wrong. Albumin, total protein and
bilirubin are converted in fetch_external.py, and the remaining markers are U/L
in both cohorts.

What is actually happening
--------------------------
Two of the eight shared markers point in OPPOSITE directions in the two
cohorts. Measured as single-feature AUC for liver disease, where below 0.5 means
the marker runs backwards:

    marker                  NHANES   Germany
    AST                      0.657     0.938   agree
    bilirubin                0.516     0.836   agree
    albumin                  0.449     0.358   agree
    total protein            0.530     0.514   agree
    age                      0.606     0.546   agree
    alkaline phosphatase     0.591     0.306   INVERTED
    ALT                      0.654     0.218   INVERTED

The model learned from NHANES that a raised ALT means liver disease, which is
correct there. In the German cohort a raised ALT means the patient is HEALTHY.

That is not a data error, it is case mix. The two cohorts mean different things
by "liver disease":

  NHANES   self-reported "ever told you had a liver condition" in a general
           population, dominated by fatty liver and mild chronic hepatitis,
           where ALT is elevated. Cases median ALT 27 against 20 in controls.

  Germany  screened blood donors against hepatitis, fibrosis and cirrhosis
           patients, 30 of the 75 with cirrhosis. In advanced cirrhosis ALT
           FALLS as hepatocyte mass is lost while AST stays high, which is the
           familiar AST/ALT ratio inversion. Median ALT is 23.1 in donors,
           15.2 in hepatitis and 5.6 in cirrhosis.

So the panel transfers badly to Germany because it was trained on early, mild,
self-reported liver disease and tested on advanced histology-confirmed disease,
and ALT genuinely reverses between those two populations.

What is NOT done about it
-------------------------
Dropping ALT and ALP lifts the German transfer from 0.425 to 0.764. That number
is not reported as external validation anywhere, because those two features were
chosen by looking at the test result. That is fitting the test set, and quoting
it would be exactly the kind of thing this project keeps catching elsewhere.

The honest position is that 0.442 stands as the measured transfer, now with a
mechanism attached instead of a shrug, and that it bounds the claim: this panel
describes mild self-reported liver disease in a US population and should not be
expected to rank advanced cirrhosis.

Run:  python experiments/liver_germany.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_external as fe

warnings.filterwarnings("ignore")

OUT = "experiments/liver_germany_result.json"
SHARED = ["age", "bilirubin", "alkaline_phosphatase", "alt", "ast",
          "protein_total", "albumin"]


def main():
    nh = pd.read_csv("data/nhanes_liver_multicycle.csv")
    de = fe.fetch_hcv()
    y_nh = nh["liver_disease"].astype(int)
    y_de = de["liver_disease"].astype(int)

    print(f"NHANES {len(nh)} adults, {int(y_nh.sum())} with a liver condition")
    print(f"Germany {len(de)} patients, {int(y_de.sum())} with hepatitis, fibrosis "
          f"or cirrhosis\n")

    print("Single-feature AUC for liver disease in each cohort.")
    print("Below 0.5 means the marker runs backwards in that cohort.\n")
    print(f"  {'marker':<24}{'NHANES':>9}{'Germany':>10}   direction")

    rows = {}
    inverted = []
    for f in SHARED:
        a = roc_auc_score(y_nh, pd.to_numeric(nh[f], errors="coerce").fillna(nh[f].median()))
        b = roc_auc_score(y_de, pd.to_numeric(de[f], errors="coerce"))
        agree = (a - 0.5) * (b - 0.5) > 0
        if not agree:
            inverted.append(f)
        print(f"  {f:<24}{a:>9.3f}{b:>10.3f}   {'agree' if agree else 'INVERTED'}")
        rows[f] = {"nhanes_auc": round(float(a), 3), "germany_auc": round(float(b), 3),
                   "same_direction": bool(agree)}

    print(f"\nMarkers pointing opposite ways: {inverted}")

    # The clinical mechanism, shown rather than asserted.
    print("\nALT by disease stage in the German cohort:")
    raw = fe.fetch_hcv.__doc__  # noqa: F841  (kept so the docstring travels with the file)
    from ucimlrepo import fetch_ucirepo
    X = fetch_ucirepo(id=571).data.features.copy()
    X["cat"] = fetch_ucirepo(id=571).data.targets.iloc[:, 0].astype(str)
    med = X.assign(ALT=pd.to_numeric(X["ALT"], errors="coerce"),
                   AST=pd.to_numeric(X["AST"], errors="coerce")) \
           .groupby("cat")[["ALT", "AST"]].median().round(1)
    print(med.to_string())
    print("\nALT falls as disease advances while AST climbs, which is the AST/ALT")
    print("ratio inversion of cirrhosis. NHANES cases are mild, so ALT rises there.")

    payload = {
        "measured_transfer_auc": 0.442,
        "cause": ("Case mix, not a bug. ALT and alkaline phosphatase point in opposite "
                  "directions in the two cohorts because NHANES captures mild self-reported "
                  "liver disease where ALT is raised, and the German cohort captures "
                  "advanced histology-confirmed disease where ALT falls as hepatocyte mass "
                  "is lost."),
        "ruled_out": ["ucimlrepo column misalignment (values match the original CSV exactly)",
                      "unit mismatch (albumin, protein and bilirubin are converted; the rest "
                      "are U/L in both)"],
        "inverted_markers": inverted,
        "per_marker": rows,
        "not_reported_as_validation": ("Dropping ALT and ALP raises the German transfer to "
                                       "0.764, but those features were selected by looking "
                                       "at the test result, so that number is fitting the "
                                       "test set and is not quoted as validation."),
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
