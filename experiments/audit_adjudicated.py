"""
Step 2 of the audit: the adjudicated codes for every record that met the inclusion rule.

Each decision was made by reading the abstract and the sentences the cue rules pulled out of
the full text (experiments/audit_extract.py). Inclusion rule (docs/AUDIT_PREREG.md): a US
NHANES paper that builds a prediction or classification model with reported performance,
whose target is cancer (any site) or death from cancer.

Records are keyed by position in data/audit/records.csv (the order the fixed search returned
them in) and by PMID. Everything not listed here was excluded; the reason for each exclusion
is in EXCLUDED below.

Fields
    site         the cancer site the model targets ("any" = any cancer)
    task         xs = cancer status at the same visit as the predictors; mort = cancer death or
                 survival in cancer patients; comorb = diabetes-cancer or CVD-cancer comorbidity (xs)
    validation   random = one random split; cv = cross-validation only; none = no validation
                 described in the text; split_unstated = a test set is mentioned, how it was
                 made is not assessable (abstract only)
    ext          True if any external cohort (non-NHANES) or held-out NHANES cycles are used
    heldout      True if held-out NHANES cycles were used
    agesex       an age-only or age-and-sex-only baseline is reported
    comparator   a logistic-regression, Cox or clinical-variables comparator is reported
    auc          best test-set AUC (or C-index) the authors report for the cancer target
    weights      NHANES survey weights used when FITTING the model (None = not assessable)
    calib        calibration (plot, Brier, Hosmer) reported
    resample     SMOTE or other resampling used
    guard        data-leakage guard reported (None = not applicable / not assessable)
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# idx, site, task, validation, ext, heldout, agesex, comparator, auc, weights, calib, resample, guard, abstract_only
R = [
    (1, "any", "comorb", "none", False, False, False, False, 0.996, False, True, True, False, False),
    (2, "colorectal", "xs", "random", False, False, False, False, 0.787, False, True, True, True, False),
    (5, "uterine", "xs", "random", False, False, False, True, 0.964, False, True, False, None, False),
    (7, "breast", "xs", "random", True, True, False, False, 0.832, False, True, False, None, False),
    (10, "digestive", "xs", "random", True, False, False, True, 0.852, False, True, False, None, False),
    (13, "prostate", "xs", "none", False, False, False, False, 0.90, False, False, False, None, False),
    (16, "any", "xs", "random", False, False, False, True, 0.92, False, True, True, None, False),
    (17, "any", "mort", "random", False, False, False, False, 0.891, False, False, False, None, False),
    (20, "any", "mort", "random", False, False, False, True, 0.84, None, False, True, None, False),
    (23, "breast", "xs", "random", False, False, False, False, 0.727, False, False, False, None, False),
    (25, "any", "xs", "random", False, False, False, False, 0.765, False, False, False, None, False),
    (27, "bladder", "xs", "none", False, False, False, False, 0.78, False, False, False, None, False),
    (28, "any", "mort", "cv", False, False, False, True, 0.674, None, True, False, None, True),
    (29, "prostate", "xs", "split_unstated", False, False, False, False, 0.869, None, None, None, None, True),
    (30, "prostate", "mort", "none", True, False, False, True, 0.751, None, True, False, None, False),
    (36, "any", "comorb", "cv", False, False, False, True, 0.951, False, False, True, None, False),
    (37, "breast", "xs", "none", False, False, False, False, 0.84, False, True, False, None, False),
    (39, "prostate", "xs", "random", False, False, False, False, 0.768, False, True, False, None, False),
    (42, "breast", "xs", "cv", False, False, False, False, 0.797, False, True, False, None, False),
    (45, "any", "xs", "cv", False, False, False, True, 0.812, None, False, False, None, True),
    (48, "colorectal", "xs", "random", False, False, False, True, 0.87, False, True, True, True, False),
    (59, "any", "xs", "random", False, False, False, False, None, False, True, True, None, False),
    (62, "digestive", "xs", "random", False, False, False, False, 0.827, False, False, False, None, False),
    (64, "any", "mort", "none", False, False, False, True, 0.850, None, None, None, None, True),
]
COLS = ["idx", "site", "task", "validation", "ext", "heldout", "agesex", "comparator", "auc", "weights", "calib", "resample", "guard", "abstract_only"]

EXCLUDED = {
    0: "association analysis; ML only ranks variables, no model performance",
    3: "Korea NHANES (KNHANES), a different survey; target is psychological distress",
    4: "target is sarcopenic obesity, not cancer",
    6: "target is insulin resistance",
    8: "target is dietary fat",
    9: "targets stroke and mortality",
    11: "target is fatty liver",
    12: "target is cardiovascular death among cancer survivors",
    14: "target is all-cause mortality in CVD-cancer comorbidity",
    15: "mechanistic study, no prediction performance",
    18: "target of the models not clearly cancer; abstract only",
    19: "target is multimorbidity",
    21: "target is CVD in cancer survivors",
    22: "target is benign prostatic hyperplasia",
    24: "target is osteoporosis",
    26: "single-marker association; no model performance reported in the abstract",
    31: "target is depression",
    32: "Mendelian randomisation and omics; unclear",
    33: "association study; single marker",
    34: "mortality association in kidney disease",
    35: "association study",
    38: "association study",
    40: "target is asthma",
    41: "clustering and association",
    43: "target is mortality in rheumatoid arthritis",
    44: "association study (environment-wide)",
    46: "omics",
    47: "mechanistic cell study",
    49: "target is asthma",
    50: "omics and genetics",
    51: "statistical method paper",
    52: "association, no model performance in the abstract",
    53: "chronic disease burden and all-cause mortality",
    54: "association study",
    55: "target is GI health in cancer survivors",
    56: "single-index ROC; no ML model evident (abstract only)",
    57: "hyperuricaemia in breast-cancer patients",
    58: "hospital data primary; NHANES only supporting",
    60: "target is lean mass",
    61: "omics",
    63: "biological age",
    65: "target is emphysema",
    66: "omics",
    67: "microbiome age",
    68: "target is sleep apnoea",
    69: "target is kidney disease",
    70: "target is albuminuria",
    71: "target is benign prostatic hyperplasia",
    72: "PSA association",
    73: "Korea NHANES (KNHANES), a different survey",
    74: "target is a high PSA level, not a cancer diagnosis",
    75: "obesity drug use",
    76: "association study",
    77: "global burden and omics; unclear",
    78: "simulation of HPV-related cancer, not NHANES prediction",
}


def table():
    df = pd.DataFrame(R, columns=COLS)
    rec = pd.read_csv(os.path.join(ROOT, "data", "audit", "records.csv"))
    df["pmid"] = df["idx"].map(rec["pmid"])
    df["pmcid"] = df["idx"].map(rec["pmcid"])
    df["year"] = df["idx"].map(rec["year"])
    return df


if __name__ == "__main__":
    df = table()
    df.to_csv(os.path.join(ROOT, "data", "audit", "adjudicated_codes.csv"), index=False)
    ex = pd.DataFrame([{"idx": k, "pmid": pd.read_csv(os.path.join(ROOT, "data", "audit", "records.csv")).iloc[k]["pmid"], "reason": v} for k, v in EXCLUDED.items()])
    ex.to_csv(os.path.join(ROOT, "data", "audit", "excluded.csv"), index=False)
    print(f"{len(df)} included, {len(ex)} excluded, {len(df) + len(ex)} of 79 accounted for")
