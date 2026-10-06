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
