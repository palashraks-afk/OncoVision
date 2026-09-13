# Draft: NCI CDAS data request for PLCO ovarian and prostate data

**Status: draft for review.** Nothing here has been submitted. A CDAS project must
be submitted and owned by a named investigator; that person has not been
confirmed and is left as a placeholder below.

## Why this request exists

Oncovision ships four panels built on case-control cohorts — ovarian (349 operated
masses), prostate (212 biopsied men), pancreatic (600 tissue-bank samples) and a
biopsy breast panel (569 aspirates). Their AUCs are the highest in the project and
their evidence is the weakest, because each cohort was assembled from people who
already had a reason to be tested. No sample size fixes that; only a screening
population does.

Two alternatives were tried and failed:

- **NHANES PSA for prostate.** NHANES 2005–2010 measured PSA in 4,697 men and 253
  reported a prostate cancer, but only 17 of those 253 had PSA drawn — men with a
  prior diagnosis were excluded from the PSA subsample — and those 17 had a *lower*
  median PSA than controls (0.85 vs 1.0 ng/mL) because they had been treated.
- **No public ovarian screening cohort** with CA-125 and an outcome was found.

The breast case shows what a screening population changes. Rebuilt on 2,392,998
BCSC screening mammograms, the breast panel scores AUC 0.623 rather than 0.997 —
and beats age alone by +0.028 (95% CI +0.021 to +0.035) on a held-out split of
597,859 mammograms, with a rule-out cut that excludes 9.8 more women per hundred
than a cut on age at the same sensitivity. That is the kind of result PLCO would
make possible for ovarian and prostate.

## Data requested

From the PLCO trial as distributed through CDAS:

| Dataset | Why |
|---|---|
| PLCO Ovarian (one record per woman, ~78,000) | outcome, incidence and mortality |
| PLCO Ovarian Screening (one record per screen, ~151,000) | serial CA-125 and transvaginal ultrasound results |
| PLCO Prostate and Prostate Screening | serial PSA and DRE results, outcome |
| Baseline questionnaire | age, family history, menopausal status, BMI, smoking |

Per the CDAS documentation, cancer data are available through 31 December 2009
and mortality through 2022.

## Aims

1. **Ovarian.** Estimate how well CA-125 with age and menopausal status separates
   women later diagnosed with ovarian cancer from women who are not, *in a screening
   population at its real prevalence*, and whether a rule-out cut keeps its promise.
2. **Prostate.** The same for PSA with age and family history.
3. **Against age alone.** For both, measure the gain over the strongest age-only
   model on identical folds, and test any rule-out cut against a cut on age alone at
   matched sensitivity. This is the test that withdrew Oncovision's bowel panel,
   whose lab values turned out to add nothing to age.

## Methods (summary)

- Train on the PLCO intervention arm's early screening rounds; hold out later rounds
  and, separately, one or more screening centres, so the external test is between
  sites rather than a random split.
- Model kinds: logistic regression and a calibrated tree ensemble; the ensemble
  ships only if it leads by more than 0.005 cross-validated AUC.
- Baseline: the better of logistic and ensemble on age (and sex where relevant)
  alone, on the same folds. Gains reported with paired bootstrap intervals.
- Race and ethnicity recorded as stratifiers for subgroup accuracy, never features.
- All code is already public in the Oncovision repository and would be applied
  unchanged apart from the cohort loader.

## Data handling

PLCO data would be held only on the investigator's approved system, never committed
to the public repository, never used in the public web application's training
without separate approval, and destroyed at the end of the approved access period.
Only aggregate results would be published.

## What the named investigator would need to do

1. Register on CDAS and open a new PLCO project with the aims above.
2. Accept responsibility for the Data Transfer and Use Agreement.
3. Confirm institutional requirements (at UCI, whether a determination of
   non-human-subjects research or an IRB exemption is needed for de-identified
   trial data).

A PLOS Medicine report on data sharing through CDAS ("Data sharing in clinical
trials: An experience with two large cancer screening trials") gives 199 of 215 PLCO
data requests approved between November 2012 and October 2016. Verify the figure
against the paper before quoting it in the submission.

**Named investigator:** _[to be confirmed]_
**Student researcher:** _[name]_
