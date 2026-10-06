# Risk-ranking study: decided before it was run

Written and committed BEFORE `experiments/risk_ranking.py` was run.

## Why this exists

The project's original idea was screening in the sense of finding risk: use everything a
person has (labs, habits, history, questionnaire answers) to rank who is more likely to be
affected by cancer, so that scarce follow-up goes to them first. The navigator study showed
that the guideline's lab rules do not do this. This asks the broader question properly,
with much more information per person and a later era held out.

## Cohort

`fetch_nhanes_wide.py` with `WIDE_CYCLES=8`: NHANES 1999-2014, adults 40 and over, every
laboratory, examination and questionnaire column recorded in at least six of eight cycles
(over 300 columns), linked to deaths through 2019. Nobody dropped for how they died.
Outcome: death from cancer within 60 months of examination. Non-cancer deaths inside the
window are non-cases (a competing-risk simplification, stated).

## Split

Fit on cycles 1999-2006. Test on cycles 2007-2014, which the model never saw and whose lab
methods and population differ. No tuning on the test era. Hyperparameters fixed in advance:
ridge logistic regression with C = 0.05 after standardising and median imputation, and
gradient boosting with depth 3, 300 trees, learning rate 0.05, minimum 50 per leaf.

## Feature sets, from simple to broad

| Set | Contents |
|---|---|
| S0 | age, sex |
| S1 | S0 + ever smoked + body mass index. **This is the strong baseline.** |
| S2 | S1 + the 22 routine blood and chemistry values |
| S3 | the whole picture: every column, with missing-value indicators |

## Questions and bars

| # | Question | Bar to call it a success |
|---|---|---|
| R1 | Does the whole picture (S3, better of the two models) beat the strong baseline S1 for five-year cancer death in the held-out era? | AUC gain of at least 0.02 AND bootstrap 95% interval above 0 |
| R2 | Is the score specific to cancer? Among people who died within five years in the held-out era, can it tell cancer deaths from other deaths better than age and sex can? | Decedent AUC of S3 exceeds that of S1 with a bootstrap 95% interval above 0 |
| R3 | Is it calibrated in the held-out era? | Calibration slope between 0.8 and 1.2 |
| R4 | Does more data help? Fit on 25%, 50% and 100% of the training cycles. | Reported, no bar |
| R5 | Triage value: what share of cancer deaths fall in the top 10% and 20% of scores, against S1? | Reported, no bar |

## Expectation, written down

Earlier work in this project found the whole picture told which way a death goes more than
whether it was cancer. R1 and R2 are therefore expected to be close calls or misses. If a bar
is missed it is reported as missed in the paper and nothing is rescued by changing the
split, the features or the bar.

## What this cannot show

That a score is useful for screening. The outcome is death, not diagnosis; people whose cancer
was found and cured are non-cases; the sample is a US survey; and the score has never been
used to decide anything.

## Addendum, written before running the three follow-up analyses

The first run missed R1 and R3 for the 380-variable model and suggested why (missing-value
flags that encode the survey cycle). Three follow-ups are fixed here, before they are run.

### F1. A curated, cycle-stable feature set (S4)

Not "everything", but variables a clinician would plausibly think relevant AND that were
recorded in all eight cycles for at least 80% of people (checked on the file, before fitting):
age, sex, smoking, body mass index, waist circumference, weight now, weight change over the
past year (computed as (weight a year ago minus weight now) / weight a year ago), HbA1c,
albumin, cholesterol, GGT, uric acid, haemoglobin, MCV, platelets, white count, told they have
diabetes, high blood pressure, asthma, congestive heart failure, coronary heart disease,
angina, heart attack, stroke, emphysema, chronic bronchitis, liver condition, and a
self-rated health item. No missing-value indicators. Ridge logistic regression, C = 0.05, median
imputation, same split (fit 1999-2006, test 2007-2014).

Bars: AUC gain over S1 of at least 0.01 with a bootstrap 95% interval above 0, AND
calibration slope between 0.8 and 1.2, AND a decedent-only AUC gain over S1 with an interval
above 0.

### F2. Is the failure a survey-cycle artefact?

(a) Predict era (1999-2006 versus 2007-2014) from the missing-value flags alone, five-fold
cross-validated. (b) Fit the 380-variable model on a random half of ALL cycles pooled and test
it on the other half. If the era-split failure is an artefact the flags alone will identify the
era well (AUC above 0.9) and the random split will score far higher than the era split (0.629).
No bar; this is a diagnostic of an explanation already given.

### F3. Is the pooled cancer-specificity result different from NHANES III?

A z-test on the difference of the two log odds ratios (1999-2014 labs-only, NHANES III
labs-only), reported with its p-value. No bar.
