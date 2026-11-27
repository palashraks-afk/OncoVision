# Pre-registration: do flags on other routine tests point to cancer specifically?

Written and committed on 27 November 2026, before any model of these flags has been fitted in this form.
**Not blind:** this project has already found that routine values, taken together as continuous inputs (337 NHANES
variables, `experiments/whole_picture.py`; blood-count markers, `docs/CHEAP_MARKERS_PREREG.md`; checkup labs), add
little for cancer death and often behave as frailty markers. Abnormal-result flags at clinical cut-offs have not been
tested this way here.

## Flags (cut-offs fixed now)

| Flag | Routine test | Definition |
|---|---|---|
| High calcium | metabolic panel | albumin-corrected calcium (Ca + 0.8 x (4 - albumin)) above 10.5 mg/dL |
| High alkaline phosphatase | liver panel | above 130 U/L |
| Low albumin | metabolic panel | below 3.5 g/dL |
| High GGT | liver panel | above 60 U/L |
| High white count | blood count | above 11.0 x10^9/L |
| High platelets | blood count | above 400 x10^9/L |
| High FIB-4 liver score | calculated from age, AST, ALT, platelets | above 2.67 |
| Unrecognised diabetes-range HbA1c | HbA1c | 6.5% or higher, never told they have diabetes |
| Albumin in urine | urine albumin/creatinine | 30 mg/g or higher |
| High C-reactive protein | CRP | above 1.0 mg/dL |
| Two or more of the above | | secondary combined flag |

Blood pressure, pulse and lung function are not included: they are not in the files used here for both datasets.

## Data and analysis

Adults 50-84 without reported earlier cancer. **Discovery:** NHANES 1999-2014 (`data/nhanes_wide_8c.csv.gz`).
**Replication:** NHANES III 1988-1994, lab values read at the positions in CDC's lab.sas. Deaths through 2019; five-year
outcome in three classes: alive, cancer death, other-cause death. Each flag is added on its own to a multinomial logistic
model with age (restricted cubic spline), sex and ever smoking (plus survey wave in discovery). People missing a test are
left out of that flag's model only. Maximum likelihood with Wald intervals.

Measures: relative-risk ratio (RRR) for cancer death, RRR for other-cause death, and their **ratio** (cancer / other).

## Bars (fixed now)

- **Discovery:** a flag passes if the cancer RRR **and** the ratio are both above 1 with two-sided p below 0.05 / 11
  (Bonferroni over 11 flags). The hit list is written before replication is looked at.
- **Replication:** a passing flag is confirmed if the cancer RRR and the ratio both have 95% intervals above 1 in NHANES III.
- Reported for every flag, pass or fail: counts, cancer deaths among the flagged, both RRRs, the ratio, and in both datasets.
  Replication results of non-hits are labelled exploratory. No cut-off, flag or rule changes after results.

## Limits

Death from any cancer, not site. Unweighted. Single measurement. Lab methods differ between eras. A cancer-specific flag
would still not be a diagnostic test; a miss means these cut-offs do not single out cancer in these data.


## Result (27 November 2026)

No flag passed discovery, so none was confirmed. Full numbers: `experiments/routine_test_flags_discovery.json` and
`..._replication.json`; paper Section 3.8. Departure: the first NHANES III run read the wrong column for diagnosed diabetes
(a width from CDC's LENGTH list was taken as the position). It showed as 14.5% unrecognised diabetes; after the fix it is
5.5%. The change affects only that flag and the combined flag in NHANES III; discovery does not use that code.
