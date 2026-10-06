# Personal risk estimate: decided before it was fitted

Written and committed BEFORE `experiments/risk_estimate.py` was run.

## What is being built

A personal estimate of the chance of dying of cancer within five years, from age, sex, whether
the person has smoked, and body mass index (BMI), shown with a plausible range. These are the
four inputs the risk-ranking study found to be the best available in this survey: nothing added
to them reliably (`docs/RISK_RANKING_PREREG.md`, S1 and the follow-ups).

It is an estimate of **death from cancer**, not of getting cancer, because that is what the
data record. It says so in the words shown to the user.

## Data and split

NHANES 1999-2014 adults 40 and over with five-year status known (`data/nhanes_wide_8c.csv.gz`).
Fit on cycles 1999-2006, test on 2007-2014, as before. The deployed model is refitted on all
cycles only after the bars below are met.

## Candidate forms (fixed in advance)

| Model | Terms |
|---|---|
| M1 | age, age squared, sex, ever smoked |
| M2 | M1 + BMI (linear) |
| M3 | M1 + BMI and BMI squared (U-shaped; low BMI often reflects illness) |

The form is chosen by the lower Brier score in the held-out era. Ties (within 0.00005) go to
the simpler model. Missing smoking or BMI is filled with the training mean, and the person is
told when that happened.

## Bars (all in the held-out era)

1. AUC of at least 0.74.
2. Calibration slope between 0.8 and 1.2, and intercept between -0.3 and 0.3.
3. In ten equal-sized groups ordered by predicted risk, the observed rate lies inside the 95%
   Wilson interval of the group in at least 8 of 10 groups.
4. Calibration within each of four age bands (40-49, 50-59, 60-69, 70+): the observed total
   number of cancer deaths is within 25% of the predicted total, or within the Poisson 95%
   interval of it, in every band.

## Plausible range

For the deployed model, 300 bootstrap refits give a 90% interval for a person's predicted
risk. It describes uncertainty in the fitted model, not the person's biology, and it will be
labelled so.

## What it cannot be

A risk of getting cancer. A personal prediction that beats age, sex, smoking and BMI. A
statement about people outside the US, outside 40 to 85, or about cancers found early and cured.
Anyone with a family history, a known gene change, or earlier cancer has a different baseline.

## If a bar is missed

The estimate is not shown as a percentage. The group context line stays as it is.
