# Cancer-risk age: how much cancer-death risk can be read from free information?
## Pre-registration, written and committed before any model was fitted (9 October 2026)

## Why this question, and why it is the original idea

The original idea was cheap cancer risk prediction that an ordinary person can read. Earlier work
here found that routine blood tests add almost nothing to what a few free facts already say
(`docs/marker_paper/`), and that the free facts (age, sex, smoking, weight) reach AUC about 0.74
for ten-year cancer death. That leaves three open questions that matter for a cheap, readable tool:

1. **How far can free information go**, once the sample is large enough to ask it properly? The
   earlier model was too imprecise to calibrate (it under-predicted people in their fifties).
2. **Can the answer be put in terms anyone understands?** Not "a risk of 0.8%" but "the cancer-death
   risk of an average 58-year-old": a **cancer-risk age**, in the way "heart age" or "lung age" are used.
3. **Does it matter for who should start screening earlier?** What share of adults aged 40 to 49 already carry
   the cancer-death risk of an average 50-year-old, and who are they?

## Data

NHIS 1997-2002 and 2005-2009 (2003 was built from its loose file; 2004 has no layout in the standard
place and is left out), adults 35 to 84, linked to the National Death Index through 2019
(`fetch_nhis_cancer_risk.py`). About 350,000 adults, ten or more years of follow-up, no prior cancer
reported. Outcome: death from cancer within ten years of interview. Non-cases: alive at ten years or dead of
another cause. External check: NHANES 1999-2008 (`data/nhanes_wide_8c.csv.gz`) and NHANES III
(`data/nhanes3_markers.csv.gz`), different surveys, different instruments and years.

## Free information (the only predictors)

Age, sex, body mass index, smoking (never, former with years since quitting, current with cigarettes
a day), schooling (four groups), race and ethnicity (non-Hispanic White, non-Hispanic Black, Hispanic,
other), marital status, self-rated health, vigorous activity per week. Race and education enter only as
**described groups for checking fairness**; whether they also enter the risk model is decided by the model
comparison below, and race is never used in the version shown to users.

## Models (fixed in advance)

| Model | Terms |
|---|---|
| F0 | age (spline), sex |
| F1 | F0 + smoking (status, intensity, years since quitting) |
| F2 | F1 + BMI |
| F3 | F2 + schooling, marital status, self-rated health, vigorous activity |
| F4 | F3 + race and ethnicity (for the fairness comparison only) |

Logistic regression with a natural spline in age (4 degrees of freedom) separately by sex, so the age
curve can bend where the earlier model failed. **Fit** on interview years 1997-2002 (6 years);
**temporal test** on 2005-2009 (5 years); **external test** in NHANES 1999-2008 and NHANES III using the
variables each has (age, sex, BMI, smoking status; schooling in NHANES).

## Cancer-risk age (definition, fixed now)

Let R0(a, s) be the model F0 predicted ten-year cancer-death risk for sex s at age a. A person's
**cancer-risk age** is the age a* with R0(a*, s) equal to the risk their own model (F2, the version using
only age, sex, smoking and BMI, or F3 if it is pre-registered below to win) predicts. The **risk advancement
period** is a* minus their age. A person is "aged 45 with a cancer-risk age of 58" when their free
information puts them where an average 58-year-old of their sex stands.

## Bars (all must be met in the temporal test; the first two also in the external data)

1. **Overall calibration.** Slope between 0.8 and 1.2, intercept between -0.3 and 0.3.
2. **Calibration in every age band** (35-49, 50-59, 60-69, 70-84): observed cancer deaths within 20% of
   predicted, or inside the Poisson 95% interval of the prediction. **The band that failed before, 50-59, is
   named here.**
3. **Discrimination.** F2 beats F0 by an AUC of at least 0.02 with a bootstrap 95% interval above 0.
4. **Fairness.** Calibration slope between 0.8 and 1.2 within each of the four race and ethnicity groups,
   and the AUC difference between the best and worst group no larger than 0.03.
5. **Cancer-risk age is calibrated.** In the temporal test, among people whose risk advancement period is 5
   years or more, observed ten-year cancer death must lie within the 95% interval of the age-and-sex
   curve at their cancer-risk age, in each sex.

Which of F2 and F3 is the "cheapest model that suffices" is decided by this rule: the simpler model unless the
richer one gains at least 0.01 AUC in the temporal test with an interval above zero.

## Screening-start question (descriptive, no bar)

Among adults 40 to 49 in the temporal test: the share whose cancer-risk age is 50 or more; their composition
by sex, smoking, schooling and race and ethnicity; the share of all cancer deaths in 40-49-year-olds that
occur among them; and the same for flagging the oldest 40-49-year-olds at the same rate. The measure is all
cancer deaths, which includes cancers with no screening test, so this describes risk concentration and is not
an estimate of what screening would achieve.

## Expectation, written down

Smoking will dominate (hazard of smoking-related cancers); BMI will add a little; schooling will add a small
amount. The cancer-risk age of a current heavy smoker in their forties will be a decade or more above their
age. The earlier calibration failure at 50-59 may or may not recur: a spline age curve and ten times the data
should help, and if it still fails the paper says so. Race is expected to add little once smoking and schooling are in.

## What this cannot show

That a person will or will not get cancer; anything about cancer incidence (the outcome is death); effects of
screening; effects of changing a risk factor (these are associations). All cancers are pooled.
