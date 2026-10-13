# Pre-registration: a free-data ladder for multi-cancer risk (questions, checkup, phone/watch activity)

Written and committed on 12 October 2026, before the analysis is run and before the activity summaries
exist (they are still being computed from the raw minute files). Companion to `docs/MCED_LAW_PREREG.md`.

## Question

Does information that people already get for free add to simple questions in ranking who will die of
cancer (any cancer, the target of multi-cancer screening)? The layers, in the order a person meets them:

- **Q, questions**: age, sex, smoking (never / former / current), BMI from self-reported height and weight,
  and **weight change over the past year** (self-reported weight now against a year ago) together with whether
  the person was trying to lose weight. Lost 5% or more without trying is the "unintentional weight loss" flag.
- **C, routine checkup**: blood pressure and the standard blood panel already done at a yearly physical
  (complete blood count including haemoglobin, white cells, platelets, red-cell distribution width; albumin,
  alkaline phosphatase, glucose, creatinine, total cholesterol). Earlier work here found such labs add almost
  nothing to age, sex, smoking and weight for cancer death; that is the expectation and it is tested again.
- **W, phone/watch-type activity**: from 7 days of hip accelerometer minutes (NHANES 2003-2006): total
  activity per day (log), minutes of moderate-to-vigorous activity, share of wear time sedentary, and the
  active-to-sedentary transition probability (fragmentation). These approximate what consumer devices record;
  that a research hip monitor equals a phone is an assumption and is stated as one.

## Data

NHANES 2003-2004 and 2005-2006, adults 40-84, no reported earlier cancer, with at least 4 valid accelerometer
days, linked to the National Death Index through 2019. Outcome: cancer death (underlying cause 2) within
ten years. `fetch_nhanes_activity.py` builds the activity summaries; `data/nhanes_wide.csv.gz` holds the rest.

## Analysis

Logistic models Q, Q+C, Q+W, Q+C+W (age as a cubic spline, by-sex intercept). Because there are only a few
hundred cancer deaths, performance is estimated by **repeated 5-fold cross-validation (20 repeats)** and AUC
gains get paired bootstrap 95% intervals over the out-of-fold predictions. Calibration slope out of fold.
Exploratory: the same in adults 65 and older (the Medicare age band); which single W feature carries any gain;
whether W still adds when other-cause death is the outcome (a check that W measures frailty in general).

## Bars (fixed now)

1. W adds to Q: AUC gain of at least 0.01 with an interval above zero.
2. C adds to Q: AUC gain of at least 0.01 with an interval above zero (expected to be missed).
3. If W adds, it adds for cancer death and not only for other-cause death: the W gain for cancer death is at
   least half its gain for other-cause death. (If W only measures general frailty it is not a cancer screen.)

## Prior work, stated now (three recorded search rounds, `data/audit/novelty_freedata.csv`)

Wearable data predicting cancer mortality has been shown before: Tedesco et al. 2021 (2,291 Swedish
70-year-olds, AUC 0.857 with demographics, self-report and wearable data), Xu et al. 2022 (NHANES 2011-2014
wrist rest-activity rhythm and cancer mortality), and UK Biobank pattern studies. So "a wearable predicts cancer
death" is **not** new. What this study adds, if anything, is the head-to-head ladder (does the wearable add
beyond free questions including weight change, and beyond the routine checkup), the specificity check against
other-cause death, and the link to whom a multi-cancer blood test should go.
