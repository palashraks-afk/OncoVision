# The angle for the paper

## The short version

**Keep the idea. Change the claim.**

The idea is unchanged: read the labs, reports and information a person already has,
across several cancers, and explain the result in plain language. What changes is
what the paper says about it. It cannot honestly say "this screens for cancer from
routine blood work". It can say, with external validation on every claim:

> Here is exactly where routine labs, existing clinical reports and patient history
> do carry cancer-relevant information, where they provably do not, and why most
> published-style results in this area look better than they are.

That is a stronger paper than the original claim, because every number in it
survived a test designed to break it, and the failures are as reusable as the wins.

## Why the original claim cannot be the paper

A blood-only multi-cancer screening tool would have to show that routine lab values
add information beyond a patient's age and sex, on people the model never saw, with
blood drawn before the diagnosis. Four panels built that way were withdrawn in this
project, each on a bar written down before the test was run. No public dataset
allows the claim to be tested properly. A paper that asserted it would not survive a
reviewer who knew to ask for the age-and-sex baseline.

## What the paper does claim: three findings

**A. Reports a patient already holds. This is the positive result.**

- *Mammogram report plus history (breast).* On the consortium's own validation split
  of 597,859 mammograms (2,871 cancers), AUC 0.623 (95% CI 0.613 to 0.633). It beats
  age by +0.028 (0.021 to 0.035) and, the comparison that matters, beats age plus the
  density grade already printed on the report by +0.014 (0.007 to 0.020). At the same
  share of cancers caught, its rule-out cut excludes 11.6% of women against 1.8% for
  a cut on age alone (+9.8 points, 95% CI 8.3 to 10.0).
- *PSA plus MRI score (prostate).* Fitted on 212 men at one centre, scored unchanged
  on 1,500 men at three Dutch hospitals: 0.857 against 0.583 for the PSA number alone
  (+0.274, 0.245 to 0.304). Restricted to the 1,032 men with a real biopsy result,
  which removes a verification bias, 0.764 against 0.542 (+0.221, 0.180 to 0.262).
- Evidence: `experiments/bcsc_validation.py`, `breast_vs_clinic.py`,
  `bcsc_rule_out_vs_age.py`, `prostate_external.py`.

**B. Organ-specific chemistry. A scoped positive.**

- *Liver.* The panel predicts liver DISEASE, not liver cancer. On a survey cycle
  withheld from training (4,887 adults, 269 cases) it beats age and sex by +0.091
  (0.057 to 0.125). Pooled over two unseen cycles, 582 cases, +0.064 (0.042 to
  0.087), AUC 0.704. It fails on hospital patients (0.442 on a German cohort), so the
  claim is limited to adults in the general population.
- Evidence: `temporal_validation.py`, `fresh_cycle_2021.py`.

**C. Blood-only screening for named cancers. The negative result, and the main
contribution.**

| Panel | What happened | Evidence |
|---|---|---|
| Bowel | Best panel against best age-and-sex model: −0.010 inside the survey, −0.000 externally (95% CI −0.012 to +0.013) | `baseline_strength.py`, `external_baseline_strength.py` |
| General (any cancer) | Rule-out cut excluded +2.1% more adults than a cut on age (CI −3.0 to +4.3); external CI −5.6 to +8.6 | `general_rule_out_vs_age.py` |
| Lung | Two unseen cycles, 32 cases: +0.001 (−0.055 to +0.055) | `fresh_cycle_2021.py` |
| Rest-of-checkup (HbA1c, lipids, urine, blood pressure, waist) | General panel +0.016 inside the survey, −0.026 on NHANES III (−0.054 to +0.001) | `checkup_panels.py`, `checkup_external.py` |
| Five-year cancer-death panel | 0.875 against cancer deaths and 0.873 against other deaths on NHANES III: it measured being unwell | `mortality_cause_specificity.py` |
| Whole picture (337 inputs, pattern matching) | Among people who died 0.685, but for cancer death against everyone +0.003 | `whole_picture.py` |

## The five ways a result in this area goes wrong (the reusable part)

Each is a mechanism that produced a convincing number here and was caught. This is
the section a reader can take away and apply to someone else's model.

1. **A weak baseline.** A tree ensemble given only age and sex ranks age coarsely, so
   every panel measured against it looks better. Always compare against the stronger
   of a regression and an ensemble on age and sex.
2. **Internal validation cannot see a survey-specific effect.** Cross-validation, and
   even leave-one-cycle-out, holds the laboratory, instruments and protocol fixed. The
   rest-of-checkup gain passed internally and reversed on a cohort from another decade.
3. **Selecting people by what happens after the index date.** The cancer-death cohort
   dropped everyone who died of another cause, so every early death was a cancer
   death. The external cohort was built by the same rule and could not catch it:
   **a test that shares a flaw with the thing it tests proves nothing about the flaw.**
4. **Blood drawn after the diagnosis.** Survey cohorts that ask survivors whether they
   ever had cancer measure treatment and illness, not early disease.
5. **One favourable split.** Repeating the identical 80/20 split 30 times moved one
   panel's AUC from the published 0.725 to a mean of 0.594.

## Proposed structure

1. **Introduction.** Routine labs and reports are universal. The question is what
   they carry about cancer beyond age and sex. Why published AUCs mislead.
2. **Methods.** The constraint that a model may only use what the tool can collect;
   the pre-registered bars; the stronger-baseline rule; paired repeated validation;
   external cohorts; the tool's design for patient readability.
3. **Results A, B, C** as above.
4. **The five mechanisms.** The generalisable contribution.
5. **Discussion.** What the tool can responsibly claim per panel; the data that would
   settle the open question (blood before diagnosis, with a diagnosis date: CHARLS,
   HRS, MIMIC-IV, CHOC records).
6. **Limitations.** Stated first, not last (list below).

`PAPER.md` already contains the evidence for every section. It is about 900 lines and
reads as a project log, so the manuscript is a condensation, not new analysis.

## Claims the paper must not make

- That the tool screens for, or detects, a specific cancer from routine blood work.
- That liver detects liver cancer. It detects liver disease.
- That ovarian, pancreatic or the biopsy-breast panel is validated. They rest on
  case-control cohorts and have no external test. Say so in the results table.
- That prostate is a screening test. The 1,500 men were already referred.
- That any panel saves money, except the narrow supplemental-MRI result for dense
  breasts, which is small at Medicare prices.

## Limitations to put up front

NHANES records death, not diagnosis, so no cohort here pairs blood with a later
diagnosis date. Case-control panels have no screening population. No patient has used
the tool and been followed, and there is no IRB approval. The cost model is
illustrative. Race and ethnicity are reported as a stratifier and never as a feature.

## Before it goes anywhere

- **Reporting guidance.** TRIPOD+AI for prediction-model studies, and PROBAST for risk
  of bias, are the standard checklists. Fill them in; reviewers will look for them.
- **Venue.** Post a preprint first. Journals that publish methodology and negative
  results of this kind include BMC Medical Research Methodology, Diagnostic and
  Prognostic Research, the Journal of Clinical Epidemiology and PLOS ONE. These are
  examples to discuss with Dr. Chavan, not a guarantee of acceptance.
- **Ethics.** Every dataset used is public and de-identified, which usually means no
  IRB review is needed for the analysis. That determination belongs to the
  institution, so ask Dr. Chavan to confirm it before submission. PI-CAI is licensed
  CC-BY-NC, so cite it as required.
- **One claim to verify before writing it.** I found, but did not confirm from the
  papers themselves, that commercial blood-count models such as ColonFlag do not
  report an age-and-sex baseline. Check the primary papers before saying so.

## What to ask Dr. Chavan

1. Is the clinical framing right? Which of A, B, C would a clinician want to read first?
2. Can CHOC support a retrospective chart review, and who would sign as principal
   investigator? That is the natural next paper, and the protocol is drafted.
3. Is a methods-and-negative-results paper the right first publication, with the
   CHARLS or MIMIC-IV trajectory test as a second?
