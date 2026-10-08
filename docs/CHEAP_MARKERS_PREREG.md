# Cheap blood markers of cancer risk: cancer signal or general frailty?
## Pre-registration, written and committed before the analysis was run (7 October 2026)

## Why this question (and why it is the original idea)

The original idea: use the cheap tests a person already has, above all the routine blood count,
to say something about cancer risk. A large literature does exactly this with blood-count
indices: the neutrophil-to-lymphocyte ratio (NLR), the systemic immune-inflammation index
(SII), the neutrophil-percentage-to-albumin ratio (NPAR), the advanced lung cancer
inflammation index (ALI) and others. A search of Europe PMC on 7 October 2026 found 103
NHANES papers with these indices and cancer in the abstract, but only 5 that also mention
incremental prediction (C-index, net reclassification) and 6 that mention specificity, negative
controls or competing causes. The ones we read report associations, and several report that the
same index predicts cardiovascular death at least as well as cancer death.

An index that rises with illness in general will "predict cancer" in any sample where sick
people die of cancer, and it will also predict heart disease, diabetes and everything else. Two
properties decide whether a cheap marker is a **cancer** risk marker: it must add to what age, sex,
smoking and body mass index already tell us, in data it was not built on, and it must be more
strongly related to dying of cancer than to dying of other things.

## Cohort

`data/nhanes_wide_8c.csv.gz`: NHANES 1999-2014, adults 40 and over, no cancer diagnosis reported at
the exam, complete blood count with differential (and albumin where an index needs it), linked
to deaths through 2019. Outcomes use a **ten-year horizon** in cycles 1999-2008, which all have
at least ten years of follow-up: death from cancer (case), death from another cause, alive. A
secondary five-year analysis uses all eight cycles.

## Indices (fixed in advance)

NLR (neutrophils / lymphocytes), PLR (platelets / lymphocytes), MLR (monocytes / lymphocytes),
SII (platelets x neutrophils / lymphocytes), SIRI (neutrophils x monocytes / lymphocytes), NPAR
(neutrophil percentage / albumin), ALI (BMI x albumin / NLR), PNI (albumin in g/L plus five times the
lymphocyte count), red cell distribution width (RDW), white count, albumin, haemoglobin. Each is
log-transformed and standardised on the fitting data, so an effect is "per one standard deviation".
C-reactive protein is analysed separately because it was not measured in 2011-2014.

## Models

M0 = age, age squared, sex, ever smoked, BMI (the strong baseline from earlier work). Each index is
added to M0 one at a time. Logistic regression throughout.

## Questions and bars

| # | Question | A marker passes if |
|---|---|---|
| T1 | Does it add to M0 for ten-year cancer death in an era it was not fitted on? Fit 1999-2004, test 2005-2008 | AUC gain of at least 0.01 with a bootstrap 95% interval above 0 |
| T2 | Is it specific to cancer? Among people who died within ten years (all of 1999-2008), odds of dying of cancer rather than another cause per SD, adjusted for M0 and cycle | Odds ratio above 1, with a Holm-adjusted p below 0.05 across the 12 indices |
| T3 | Negative control: does it predict death from other causes? AUC gain and odds ratio per SD for ten-year non-cancer death | Reported. An index with a larger effect on non-cancer than on cancer death is labelled a frailty marker |
| T4 | Do the indices together add? M0 plus all 12 (ridge) | AUC gain of at least 0.01 with the interval above 0 |
| T5 | Do the claims in the literature reproduce as associations? Per-SD odds ratio of self-reported cancer history at the same visit (the usual NHANES design), adjusted for M0, and the same split-based AUC gain | Reported |

**An index qualifies as a cheap cancer-specific risk marker only if it passes T1 and T2.**

## Expectation, written down

Earlier work found that routine blood values add nothing reliable to age, sex, smoking and BMI
for cancer death and that blood-count-based alerts marked non-cancer death more than cancer
death. The expected result is that no index qualifies and most behave as frailty markers. A
qualifying index would be a positive finding and is reported as one.

## What this cannot show

That an index is useless in people who already have cancer (prognosis is a different question); that
it cannot detect specific cancers; that it fails outside the US or in other eras. Death from
cancer, not a diagnosis, is the outcome. Multiple indices are correlated, so the 12 tests are not
independent; Holm correction is conservative.

---

## Addendum, 8 October 2026: three additions, fixed before they are run

The main analysis (above) is complete and unchanged: 0 of 12 markers qualified. To raise the
evidence from "one survey era" toward an independent check, to show what a marker-based rule
would do in practice, and to make the novelty claim rest on a recorded search, three further
pieces are fixed here BEFORE they are run.

### R. Independent replication in NHANES III (1988-1994)

NHANES III is a separate survey, run by different staff on different analysers five to twenty years
before the continuous survey used above, linked to deaths through 2019. It is not a registry
cohort and it is the same national programme, so it is described as an **independent replication in
a separate survey, not in a separate programme**.

- Adults 40 and over, no earlier non-skin cancer reported, complete blood count and albumin, BMI
  and smoking; ten-year outcomes as above.
- NHANES III has a three-part differential (lymphocyte, mononuclear, granulocyte), not the
  five-part one. The twelve markers are therefore **analogues**: NLR becomes granulocyte-to-
  lymphocyte (GLR), SII and SIRI use granulocytes, NPAR uses the granulocyte percentage, ALI uses GLR.
  Granulocytes include eosinophils and basophils, which is stated as a limitation.
- **Test 1 externally**: fit M0 and M0 plus marker in the 1999-2008 cohort; predict in NHANES III,
  with each marker standardised within its own survey. Test 2 and Test 3 repeat within NHANES III.
- **Replication bar.** The null replicates if no marker passes Tests 1 and 2 in NHANES III AND at
  least 5 of the 6 inflammation analogues show both a decedent odds ratio below 1 and a larger
  association with other-cause than cancer death. If any marker qualifies in NHANES III, the paper
  reports that the finding did not replicate.

### D. What a marker-based rule would do (decision analysis)

Fixed rules: flag NLR at or above 3 (a common cut-off) and flag the top fifth of NLR, SII, RDW. For each,
in the 2005-2008 test cycles and in NHANES III, among adults 60 and over: share flagged, sensitivity and
positive predictive value for ten-year cancer death, and the sensitivity of flagging the **same
number of the oldest people** by age alone. Net benefit by decision-curve analysis at risk thresholds
of 2% to 10%. No bar; the question is whether a marker rule finds more cancer deaths than age does
at the same flag rate, with a bootstrap interval.

### N. A recorded novelty search

A fixed Europe PMC query (any access, 2005-2026) for papers on blood-count or inflammation
markers (NLR, SII, PLR, MLR, SIRI, NPAR, ALI, PNI, RDW, albumin) and cancer outcomes, from which every
record whose abstract mentions a negative-control or non-cancer-death comparison, a competing-risk or
decedent-only design, AND any validation in separate data is read in full. The count, the query and the
papers read are recorded in `data/audit/novelty_search.csv`. A close prior study would lower the
novelty score; none found raises the claim to "none found in a recorded search", not to proof.
