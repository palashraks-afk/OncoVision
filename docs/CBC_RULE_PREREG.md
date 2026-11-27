# Pre-registration: can the rest of the blood count decide who needs a ferritin test?

Written and committed on 27 November 2026, before any of these blood-count indices has been compared with ferritin here
(only the MCV-below-80 shortcut has been: it missed about 3 in 4 cases, `docs/RED_FLAG_LABS_PREREG.md`, T1).
Predicting iron deficiency from blood-count indices is a well-studied topic; the question here is narrow and practical.

**Also recorded:** the public NHANES variable lists (questionnaire, examination, laboratory; all cycles, read 27 November
2026) contain no general question on colonoscopy, sigmoidoscopy or stool testing, so the endoscopy gap cannot be measured
in this survey.

## Question

Among anaemic adults aged 50 and older, can a rule using only numbers already on the blood count (and age and sex) skip a
useful share of ferritin tests without missing iron-deficiency anaemia (ferritin below 45 ng/mL)?

## Data

Same definitions as before. **Derivation:** NHANES III (1988-1994). **Validation, untouched until the rules are frozen:**
NHANES 2001-2002 and 2017-March 2020 (women 50+ reporting regular periods excluded), pooled and separately, with each
survey's exam weights and design (pooled weights are not rescaled; strata are kept distinct by survey).

## Rules (fixed now)

1. MCV below 80 fL.  2. MCH below 27 pg.  3. MCHC below 32 g/dL.  4. RDW above 14.5%.
5. Any of rules 1-4.
6. A logistic model of ferritin below 45 on MCV, MCH, MCHC, RDW, log platelets, age and sex, fitted on NHANES III
   anaemic adults. Its cut-off is the value giving 95% sensitivity in NHANES III.

NHANES III cut-offs are frozen and written to disk before validation data are read.

## Bar

A rule is **usable** if, in pooled validation, its sensitivity is at least 95% with a 95% interval lower bound of at least
90%, **and** it avoids at least 25% of ferritin tests (share of anaemic adults it would not test). All rules are reported
whatever happens. If none is usable, the conclusion is to add ferritin for every anaemic adult 50+, without a filter.

## Limits

Survey, not patient, samples; small numbers of cases in validation; analysers differ between eras (RDW especially).
