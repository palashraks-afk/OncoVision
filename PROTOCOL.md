# Prospective validation protocol and IRB submission draft

Two gaps in this project cannot be closed by writing code: nobody has run a real
patient's report through the system and followed it to an outcome, and there is
no ethics approval. Those need a study and an institution.

What code cannot do, a protocol can. This document is the actual next step: a
study design specific enough to submit, with the statistics fixed in advance so
the result cannot be massaged afterwards.

It is a draft prepared by the developer. It has not been submitted, reviewed, or
approved by any board, and nothing in it should be read as approval having been
obtained.

---

## 1. Why prospective testing is not optional

Every number reported in `EVALUATION.md` is retrospective. The model saw records
that had already been collected, already been labelled, and already been cleaned.
Three things go wrong when that is used to predict the future:

**The outcome was already known when the data was recorded.** In the pancreatic
cohort, CA 19-9 was measured because someone suspected pancreatic disease. The
model learns the consequence of clinical suspicion, not a signal that precedes it.

**The population is wrong.** Cohorts here run 21 to 71 percent positive. A
screening population runs a fraction of a percent. `EVALUATION.md` projects
precision onto real prevalence, but a projection assumes sensitivity and
specificity transfer unchanged, and the external validation in this project
shows they do not.

**Measured generalisation loss is large.** Training on German liver patients and
testing on Indian ones cost 0.297 AUC. Testing the general panel against NHANES
cost 0.370. Those are the honest error bars on every internal number here, and
prospective performance should be expected at or below them.

---

## 2. Study design

**Title.** Prospective evaluation of an automated lab-report interpretation tool
for multi-cancer and liver disease risk stratification.

**Design.** Prospective, observational, non-interventional, single arm. No
clinical decision is made or withheld on the basis of a model output.

**Primary question.** Among adults presenting for routine outpatient bloodwork,
what are the sensitivity, specificity and positive predictive value of each panel
against a documented clinical outcome at 12 months?

**Primary endpoint.** Per panel, PPV at the pre-specified operating threshold,
with a 95% confidence interval, against outcomes ascertained from the medical
record at 12 months.

**Secondary endpoints.** Sensitivity, specificity, AUC with confidence intervals;
calibration slope and Brier score in the prospective sample; performance by sex,
age band, and race and ethnicity; PDF parser field-level accuracy against manual
transcription; the difference between prospective performance and the
retrospective estimate in `EVALUATION.md`.

**Setting.** One outpatient clinic or a single health system's laboratory service.

**Population.** Adults aged 20 and over with a complete metabolic panel and
complete blood count drawn as part of routine care.

**Exclusion.** Active cancer treatment at enrolment; a cancer diagnosis within
the preceding 12 months; incomplete panels where fewer than three features are
available for any model.

**Comparator.** Logistic regression on the same inputs, and age plus sex alone.
The ensemble must beat both to justify its complexity, which is the same standard
applied retrospectively, where it already fails on two panels.

---

## 3. Sample size

Fixed before enrolment, and it is the part that determines whether the study can
answer anything at all.

Estimating a proportion to within a half-width of 0.10 at 95% confidence requires
roughly 96 events. Number of participants needed is therefore driven by the
prevalence of the outcome, not by convenience:

| Panel | Outcome | Assumed prevalence | Events needed | Participants needed |
|---|---|---|---|---|
| Liver disease | Chronic liver disease or cirrhosis | 3.1% (NHANES) | 96 | ~3,100 |
| General cancer | Any incident cancer at 12 months | 0.45% (SEER) | 96 | ~21,300 |
| Breast | Malignant on biopsy, among those biopsied | 30% | 96 | ~320 biopsied |
| Pancreatic | Pancreatic adenocarcinoma | 0.014% (SEER) | 96 | ~690,000 |

**This table is the finding.** A single clinic can realistically run the liver
panel study and, through a breast imaging service, the breast panel study. The
general panel needs a health-system-scale registry. **The pancreatic panel cannot
be prospectively validated at any single site**, because at real incidence the
required enrolment is around 690,000 people. That is why a rare-disease screening
claim is so hard to make, and it is a stronger argument for caution than any
disclaimer.

Recommended staging:

1. **Phase A, liver panel, n ≈ 3,100.** Achievable. Highest prevalence, real
   external validation already exists, and the panel is honest about detecting
   disease rather than cancer.
2. **Phase B, breast panel, n ≈ 320 biopsied.** Achievable through one imaging
   service. Note this validates interpretation of a biopsy already taken, not
   screening.
3. **Phase C, general panel.** Registry-scale, or a nested case-control design
   with matched sampling instead.
4. **Pancreatic panel.** Not prospectively testable at feasible scale. It should
   remain labelled as not validated, or be withdrawn on the same reasoning that
   withdrew prostate.

---

## 4. Pre-specification

Fixed before any data is collected, so nothing can be tuned to the result:

- Operating thresholds are frozen at the values in the shipped model bundles.
- No retraining, refitting or recalibration on prospective data. Any change
  produces a new model and a new study.
- The primary endpoint is PPV. Reporting AUC alone if PPV disappoints is
  explicitly disallowed.
- Subgroup analyses by sex, age band, and race and ethnicity are pre-specified,
  not selected after inspection.
- The analysis script is committed and tagged before enrolment opens.
- The protocol is registered on ClinicalTrials.gov before the first participant.
- Results are published regardless of direction. A negative result is the more
  likely outcome and is worth publishing.

---

## 5. IRB submission

Expected review category: **Expedited**, or **Exempt** under 45 CFR
46.104(d)(4) if the study uses only existing records and the investigator records
no identifiers. Non-interventional, no change to clinical care, minimal risk. The
board makes this determination, not the investigator.

### Risks

**Incidental finding.** The tool may flag a participant whose care team has not
flagged them. Mitigation: outputs are not returned to participants or clinicians
during the study. Analysis is retrospective to the care episode.

**False reassurance.** A low score could discourage a participant from seeking
care. Mitigation: the same blinding. No participant sees any output.

**Privacy.** Lab values and history are identifiable in combination. Mitigation
in section 6.

**Anxiety.** Not applicable while outputs are blinded.

### Benefits

None to the individual participant. The societal benefit is an honest measurement
of whether automated lab interpretation works prospectively, including if it does
not.

### Consent

Written informed consent, or a waiver under 45 CFR 46.116(f) if the study is
records-only, since the research could not practicably be carried out without a
waiver and involves no more than minimal risk. Consent language must state that
the tool is investigational, that results will not be returned, and that
participation does not affect care.

### Data monitoring

No stopping rules apply, because there is no intervention to stop. The
investigator reviews parser accuracy at 10% enrolment; a field-level error rate
above 5% halts enrolment for correction, because a validation study of a model
fed by a broken parser measures the parser.

---

## 6. Data handling

- Lab values and history extracted to a study database keyed by a random study
  ID. The identifier link is held separately, encrypted, by the site.
- No PDFs retained. Parsed values only, matching the production service, which
  discards uploads within the request.
- Race and ethnicity collected deliberately, because no existing cohort in this
  project carries it and subgroup accuracy is currently unmeasured for every
  panel except the general one against NHANES.
- Outcomes ascertained by chart review at 12 months by a reviewer blinded to the
  model output.
- Data retained 7 years, then destroyed.
- No cloud processing outside the site's own environment.

---

## 7. What would make this project trustworthy

Ordered by how much each would change the picture:

1. **Phase A completed and published**, whichever direction it lands.
2. **A second external cohort for the general panel**, ideally NHANES from a
   different survey cycle, to separate cohort effects from era effects.
3. **Any external cohort at all for pancreatic.** There is currently none in the
   public domain sharing its feature set.
4. **Parser accuracy measured** against manual transcription across at least
   three lab report formats. This is currently unmeasured and sits upstream of
   every model.
5. **A named clinical collaborator** with institutional standing to sponsor the
   IRB submission.

Item 4 is achievable immediately and does not need a board. It should be done
first.

---

## Status

| Item | State |
|---|---|
| Protocol drafted | Yes, this document |
| IRB submitted | No |
| IRB approved | No |
| ClinicalTrials.gov registered | No |
| Site identified | No |
| Clinical sponsor identified | No |
| Participants enrolled | 0 |

Nothing in this document should be represented as an approval, a registration, or
a completed study.
