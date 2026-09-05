# What routine blood work can and cannot tell you about cancer risk

**A feature-constrained multi-panel evaluation on eight public cohorts, with a prospective
mortality analysis**

Oncovision AI. Mentored research project, UCI CHOC.

---

## Abstract

**Background.** Most adults have routine blood work drawn every year. A comprehensive metabolic
panel and a complete blood count together produce roughly thirty numbers, which are typically read
one at a time against a reference range. The hypothesis behind this work is that early-detection
signal, where it exists, lies in combinations of values that are individually unremarkable, and
that this is the kind of pattern a model finds and a person scanning a printout does not.

**Methods.** We built eight risk panels under a strict constraint: a model may only use features
the application can actually collect from a patient's own lab report and a short history. Every
dataset column was mapped onto one canonical input schema or dropped before fitting. Panels were
calibrated, evaluated on a held-out split, and then re-evaluated by repeated cross-validation on
identical folds, because a single split proved unreliable. We additionally measured operating
points at real population incidence, subgroup performance by race and ethnicity where the cohort
recorded it, and split stability across many random partitions. The central hypothesis was then
tested on a design none of the training cohorts can support: 33,834 NHANES participants linked by
NCHS to the National Death Index, where blood was drawn years before the outcome existed.

**Results.** The premise holds strongly for some questions and barely at all for others, and the
distinction is the main finding. Where the organ's own chemistry appears on the lab report, the
combination of values adds a great deal over knowing age and sex: +0.106 AUC for liver disease and
+0.498 for pancreatic adenocarcinoma. Where the question is undifferentiated — will this person be
diagnosed with any cancer — it adds +0.006. On the prospective cohort of 33,834 adults with
NDI-confirmed outcomes, the full blood panel adds +0.013 over age and sex for cancer death within
five years, winning 5 of 5 paired repeats. **That gain did not survive external validation.**
Trained on NHANES 1999–2014 and tested on NHANES III (1988–1994, 14,630 adults, 254 events), age
and sex alone reached 0.852 while the full blood panel reached 0.839: a transferred gain of −0.013.
Separately, discrimination and usability diverge sharply: three panels with AUCs between 0.79 and
0.97 flag between 55 and 768 people per true case at real incidence, and a threshold sweep shows
no operating point repairs any of them.

**Conclusions.** Routine blood work carries usable signal about organ-specific disease when the
relevant analytes are on the panel, and essentially none about undifferentiated cancer risk that
generalises beyond the survey it was fitted on. The ceiling is a property of the question rather
than of the model: serum cotinine, C-reactive protein, the complete blood count, the full metabolic
panel, waist circumference and physical activity were each tested and each rejected. Two
methodological cautions follow. Resampling within one survey — including leave-one-cycle-out, which
gave an encouraging 0.837 here — measures stability and not generalisation, and only a genuinely
external cohort distinguished the two. And for a tool of this kind, the number of healthy people
flagged per true case matters more than AUC, with which it frequently disagrees.

---

## 1. Background

Screening is the part of cancer care that most obviously works and least often happens. Only a
handful of cancers have a recommended screening test at all, and those tests reach a minority of
the people eligible for them. The majority of cancers are diagnosed after symptoms appear, which
is usually later than it needed to be.

Meanwhile, a large fraction of adults already have blood drawn annually for entirely unrelated
reasons. That sample is paid for, already collected, and mostly unexamined beyond a scan for
out-of-range flags. If it carries early-detection signal, that signal is free.

The question this project set out to answer is whether it does — and, just as importantly, where
it does not. A tool that claims signal it does not have is worse than no tool, because the cost of
a false positive in cancer screening is an invasive work-up performed on a healthy person.

## 2. Methods

### 2.1 The feature-constraint rule

The rule that shaped everything else: **a model may only train on features the application can
actually collect.**

The failure this prevents is common and quiet. A public dataset contains a column that is only
knowable after diagnosis — tumour stage, surgical findings, a specialist assay — and a model
trained on it reports an excellent AUC describing a situation that never occurs at the moment the
tool would be used. Every dataset column here was either mapped onto the canonical input schema or
dropped before fitting. The pancreatic panel drops `stage`, which would leak the answer. The
prostate panel drops tumour volume and capsular penetration, which are surgical findings, and
converts the dataset's log PSA back into the ng/mL a patient reads off a report.

The rule cuts both ways, and the second direction produced one of the more useful findings here.
The breast panel originally used four of the thirty available nuclear measurements, because the
application asked for four numbers. But breast is an *interpretation* panel: it requires a fine
needle aspirate that has already been taken and imaged, and anyone holding that report is holding
all thirty. The restriction was not a property of the setting, it was a stale assumption about the
form. Correcting it is reported below.

### 2.2 Panel taxonomy

Not every panel answers the same kind of question, and conflating them is the main way an
application like this misleads. Each panel is labelled, in the code and on its own card in the
interface, as one of:

- **Screening** — runs from a routine lab report and nothing else.
- **Triage** — requires that something has already been found. It asks whether that thing is
  malignant, not whether it exists.
- **Interpretation** — reads a diagnostic test that has already been performed.

A panel with no declared type raises at training time rather than defaulting to the strongest
label, after one panel was found to have been silently claiming to screen.

### 2.3 Cohorts

Eight cohorts, four population-based and four case-control. Cohort design bounds what a number
means, so it is stated on every panel rather than kept in a methods section.

<!-- AUTOGEN:paper_cohorts -->
| Panel | Design | n | Events | Prevalence | Cohort |
|---|---|---|---|---|---|
| Breast | Case-control | 569 | 212 | 37.30% | Wisconsin fine needle aspirates, post-biopsy |
| Pancreatic | Case-control | 600 | 130 | 21.70% | 3 tissue banks, adenocarcinoma vs benign hepatobiliary |
| Ovarian | Case-control | 349 | 171 | 49.00% | operated ovarian masses, malignant vs benign |
| Prostate | Case-control | 212 | 121 | 57.10% | biopsied men, adenocarcinoma vs benign biopsy |
| Lung | Population | 21,916 | 110 | 0.50% | NHANES, adults with measurable tobacco exposure |
| Bowel | Population | 23,794 | 95 | 0.40% | NHANES, colon or rectal cancer within 8 years |
| Liver | Population | 35,511 | 1,420 | 4.00% | NHANES, 7 cycles, clinical liver disease |
| General | Population | 23,923 | 742 | 3.10% | NHANES 2005-2014, cancer diagnosed within 4 years |
<!-- /AUTOGEN:paper_cohorts -->

Race and ethnicity are carried as a **stratifier and never as a model feature**, following the
precedent set by the removal of the race coefficient from eGFR. A model that uses race as an input
encodes the consequences of unequal care as if they were biology. A model that is *measured*
across racial groups reports whether it works equally well, which is the question that matters.

### 2.4 Model and calibration

Each panel is a soft-voting ensemble of XGBoost and extremely randomised trees, with a plain
logistic regression fitted alongside as a baseline. Whichever scores higher on cross-validated AUC
is the one that ships, and for several panels that is the logistic regression — reported rather
than hidden, because an ensemble that does not beat a linear model has not earned its complexity.

Probabilities are calibrated by isotonic regression inside a cross-validation loop. Calibration
method was itself tested across three options on every small panel, and no panel moved by more
than a trivial amount, so the choice is documented as not mattering rather than defended.

### 2.5 Evaluation

**A single split is not an estimate.** This was learned rather than assumed. The cervical panel
reported 0.725 on its held-out split; run across many random partitions of the same data, its mean
was 0.594 with a spread from 0.421 to 0.789, placing the shipped number at the 97th percentile of
its own distribution. It was a lucky split. The panel was withdrawn, and every panel is now
reported with its split-stability distribution alongside its point estimate.

Panels are therefore evaluated by:

1. A held-out 20% split, cut before anything was fitted, with bootstrap confidence intervals.
2. Repeated stratified cross-validation on identical folds, for every comparison between two
   feature sets, so that paired differences are measured rather than inferred from two separate
   numbers.
3. Split-stability across many random partitions, reporting where the shipped split falls in its
   own distribution.
4. Baseline comparisons against age and sex alone, and against a single best marker, because a
   panel that does not beat either has not justified reading a lab report.

### 2.6 Operating points and what a score is worth

AUC is a ranking statistic and says nothing about whether acting on a model is defensible. Every
panel is therefore projected onto real population incidence to compute positive predictive value
and, more legibly, **the number of people flagged per true case found**.

This is the number that decides whether something is a screening instrument. A panel with an
excellent AUC that flags several hundred healthy people per cancer found is not a screening test,
whatever its discrimination, because the harm of the resulting work-ups exceeds the benefit. Where
that is true, it is stated on the panel's own card, and a threshold sweep is reported showing
whether *any* operating point fixes it.

### 2.7 The prospective analysis

Every cohort above shares one weakness that no modelling choice can repair: the blood and the
answer were recorded at the same visit, or the cases were assembled after the fact. Such a design
cannot distinguish "the bloodwork predicts the cancer" from "the cancer has already changed the
bloodwork."

NCHS links NHANES participants to the National Death Index and publishes the linkage, with vital
status and underlying cause of death through 31 December 2019. This yields a genuine cohort design:
the sample is drawn, years pass, and the death certificate arrives later from a different agency.

- **Population.** 33,834 adults aged 20 and over, NHANES 1999–2014, with no reported cancer
  diagnosis at baseline, so the outcome is incident rather than recurrent.
- **Outcome.** Death with malignant neoplasm as underlying cause within 60 months of the blood
  draw. 339 events, a 1.00% rate stable across all eight cycles.
- **Censoring.** Participants still alive with fewer than 60 months of follow-up are excluded,
  because their status at the horizon is genuinely unknown. Cycles that could not complete the
  horizon (2015–2018) are excluded entirely: they can contribute deaths but never survivors, and
  pooling them lets cross-cycle assay drift masquerade as cancer signal.

The outcome is death from cancer, not detection of it. This is a later and harsher endpoint, it
counts survivors as non-cases because they are, and it is confounded by everything determining
whether a cancer is survivable. It is a different question from detection, honestly labelled, on a
design the other cohorts cannot offer.

---

## 3. Results

<!-- AUTOGEN:paper_results -->
### 3.1 Discrimination, and what it adds over knowing age and sex

| Panel | Model | Logistic | Age and sex alone | Gain over age and sex |
|---|---|---|---|---|
| Breast | 0.997 | 0.995 | — | not measurable |
| Pancreatic | 0.969 | 0.968 | 0.5 | +0.498 |
| Ovarian | 0.949 | 0.911 | 0.813 | +0.174 |
| Prostate | 0.840 | 0.876 | 0.661 | +0.258 |
| Lung | 0.829 | 0.785 | 0.778 | +0.044 |
| Bowel | 0.793 | 0.8 | 0.817 | +0.039 |
| Liver | 0.760 | 0.74 | 0.602 | +0.106 |
| General | 0.732 | 0.731 | 0.727 | **+0.006** |

The gain column is measured by repeated paired cross-validation on identical folds, not from the held-out split, because a single split proved unreliable.

### 3.2 What a score is worth at real incidence

| Panel | Test AUC | PPV at population incidence | Flagged per true case | Usable as screening? |
|---|---|---|---|---|
| Breast | 0.997 | 95.91% | 1.0 | not a screening panel |
| Pancreatic | 0.969 | 0.47% | 210.4 | not a screening panel |
| Ovarian | 0.949 | 79.33% | 1.3 | not a screening panel |
| Prostate | 0.840 | 70.59% | 1.4 | not a screening panel |
| Lung | 0.829 | 1.81% | 55.1 | no |
| Bowel | 0.793 | 0.13% | 768.4 | no |
| Liver | 0.760 | 9.95% | 10.0 | yes, with caveats |
| General | 0.732 | 5.92% | 16.9 | yes, with caveats |

This is the table that decides whether a panel is a screening instrument. Discrimination and usability are different properties, and three panels have the first without the second.

### 3.3 Stability across resampling

| Panel | Rows | Events | Mean AUC | Spread across splits | Shipped split | Percentile |
|---|---|---|---|---|---|---|
| Breast | 569 | 212 | 0.992 | 0.970 to 1.000 | 0.997 | 77th |
| Pancreatic | 600 | 130 | 0.969 | 0.939 to 0.995 | 0.969 | 50th |
| Ovarian | 349 | 171 | 0.928 | 0.852 to 0.969 | 0.949 | 70th |
| Lung | 21,916 | 104 | 0.839 | 0.822 to 0.860 | 0.829 | 40th |
| Prostate | 212 | 121 | 0.822 | 0.732 to 0.909 | 0.840 | 70th |
| Bowel | 23,794 | 96 | 0.799 | 0.785 to 0.817 | 0.793 | 40th |
| Liver | 35,511 | 1,436 | 0.754 | 0.740 to 0.764 | 0.760 | 80th |
| General | 23,923 | 750 | 0.743 | 0.692 to 0.772 | 0.732 | 20th |
| Cervical | 858 | 55 | **0.594** | **0.421 to 0.789** | 0.725 | **97th** |

### 3.4 The prospective test

33,834 adults, 339 deaths from malignant neoplasm within 60 months of the blood draw (1.00%).

| Feature set | Features | AUC | 95% CI | Gain over age and sex | Wins |
|---|---|---|---|---|---|
| A age and sex | 2 | 0.816 | 0.796 to 0.838 | +0.000 | 0/5 |
| B + lifestyle | 5 | 0.827 | 0.812 to 0.849 | +0.011 | 5/5 |
| C + blood count | 15 | 0.832 | 0.813 to 0.85 | +0.016 | 5/5 |
| D + chemistry | 16 | 0.825 | 0.805 to 0.841 | +0.010 | 5/5 |
| E everything | 26 | 0.829 | 0.804 to 0.84 | +0.013 | 5/5 |

Leave-one-cycle-out, full feature set. Cycles differ in assay method, field staff and population, so this approximates external validation within one survey.

| Held-out cycle | AUC |
|---|---|
| 1999-2000 | 0.849 |
| 2001-2002 | 0.808 |
| 2003-2004 | 0.834 |
| 2005-2006 | 0.893 |
| 2007-2008 | 0.822 |
| 2009-2010 | 0.844 |
| 2011-2012 | 0.836 |
| 2013-2014 | 0.809 |
| **mean** | **0.837** |

**Routine bloodwork adds little beyond age and sex on this endpoint.**

### 3.5 Does that gain survive a different decade?

Trained on NHANES 1999-2014 (33,834 adults, 339 deaths) and tested on NHANES III 1988-1994 (14,630 adults, 254 deaths). Identical features on both sides. Nothing from the test cohort touches fitting, calibration or imputation.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| age and sex only | 2 | 0.852 | 0.831 to 0.873 |
| full blood work | 22 | 0.839 | 0.819 to 0.858 |

Gain over age and sex, transferred: **-0.013**. The same gain measured inside the training survey: +0.013.

**The gain does not survive the transfer.** Age and sex transfer well, at 0.852. Adding twenty blood values makes the prediction *worse* on a cohort measured in a different decade than using age and sex alone. Whatever the blood panel contributed inside NHANES 1999-2014 was specific to that survey rather than to human physiology.

This is also a caution about the leave-one-cycle-out result above. Holding out one cycle of the same survey gave a mean of 0.837 and looked like evidence of transfer. It was not. Cycles of one survey share protocols, instruments and laboratory methods, and resampling within a survey measures stability rather than generalisation. Only the genuinely external cohort distinguished them.
<!-- /AUTOGEN:paper_results -->

---

## 4. Discussion

### 4.1 The answer is not the same for every question

The premise was that early-detection signal lives in combinations of routine values. Stated that
broadly, the results neither confirm nor refute it, because it turns out to be two different
questions with two different answers.

**Where the organ's own chemistry is on the panel, the combination carries a great deal.** The
liver panel adds 0.106 of AUC over knowing age and sex, and pancreatic adds 0.498. These are not
marginal effects. They are what happens when the disease being asked about changes the very
analytes the lab report contains, and a model reads the pattern across them rather than one value
at a time. The ovarian panel behaves the same way for the same reason.

**Where the question is undifferentiated, the combination carries almost nothing — and what little
it carries does not generalise.** Asked "will this person be diagnosed with any cancer", the
general panel adds 0.006 over age and sex. Asked prospectively — will this person die of cancer
within five years — the full blood work adds 0.013 on 33,834 people with NDI-confirmed outcomes,
winning every paired repeat.

Then it was tested on a cohort measured in a different decade, and the gain reversed. On NHANES III
age and sex alone reach 0.852; adding twenty blood values gives 0.839. The transferred gain is
−0.013 against an internal +0.013. Whatever those twenty values contributed inside NHANES
1999–2014 belonged to that survey, not to human physiology.

This is not a failure of the method, and it was not for want of trying. Serum cotinine, C-reactive
protein, the complete blood count, the full metabolic panel, waist circumference and physical
activity were each measured against that panel and each rejected. The ceiling is a property of the
question, not of the model — and the external test shows the ceiling is lower still than the
internal estimate suggested.

### 4.2 Within-survey resampling is not external validation

The leave-one-cycle-out result deserves separate attention, because it was wrong in an instructive
way.

Holding out an entire NHANES cycle and training on the other seven gave a mean AUC of 0.837 across
eight folds, with no fold below 0.808. By any ordinary reading that is evidence the model
generalises: different years, different participants, different field teams.

It was not. Cycles of one survey share a protocol, a laboratory contract, instrument calibration
procedures and an analytic pipeline. Resampling across them measures stability under participant
variation while holding the *measurement process* fixed. The genuinely external cohort — a
different decade, different analysers, different assay chemistry — reversed the sign of the effect
entirely.

The practical implication is uncomfortable and worth stating plainly: every internal validation
strategy used in this project, including the repeated paired cross-validation that serves as its
arbiter throughout, would have reported this panel as working. Only an external cohort caught it.
The panels here that have no external cohort should be read with that in mind.

### 4.3 Why that distinction matters more than any AUC here

The headline numbers in this project run from 0.73 to 0.997, and the highest are the least
informative. Breast at 0.997 and pancreatic at 0.969 are case-control designs: they separate known
cases from selected controls, on cohorts of 569 and 600. Read as screening performance they are
badly misleading, and the taxonomy exists to stop them being read that way.

The tool ships eight panels. Two of them screen for a named cancer from a lab report alone. Four
require the patient to already be inside the diagnostic pathway, and one detects liver disease
rather than liver cancer. That sentence is a more honest summary of the work than any table of
discrimination statistics.

### 4.4 Discrimination is not usability

Three panels have good discrimination and no usable operating point. Projected onto real
incidence, the bowel panel flags roughly 768 people for every true case, pancreatic 210 and lung
55. A threshold sweep confirms that no operating point brings any of them to a defensible ratio:
raising specificity far enough to fix precision destroys sensitivity first.

This is arithmetic, not a modelling deficiency. At an incidence of 36.5 per 100,000, no classifier
with achievable specificity produces a tolerable positive predictive value. Reporting AUC without
reporting this is the single most common way a paper of this kind overstates itself, which is why
the flagged-per-case figure appears on every panel's own card rather than in an appendix.

### 4.5 Methodological findings

Three results here are about method rather than about cancer, and generalise beyond this project.

**A single train-test split is not an estimate.** The cervical panel reported 0.725 on its held-out
split and had a mean of 0.594 across resamples of the same data, with its shipped number at the
97th percentile of its own distribution. Nothing about that split was improper; it was simply one
draw, reported as though it were a measurement. Any model selected or reported on one split of a
small cohort is subject to the same error.

**Constraining features to what is collectable cuts both ways.** The rule prevents leakage from
post-diagnosis columns, which is its purpose. But it was also applied too aggressively to the
breast panel, which read four of thirty available measurements because the form asked for four
numbers. Since that panel requires a biopsy report that carries all thirty, the restriction
described the form rather than the setting. Correcting it moved performance on the smallest third
of lesions from 0.680 to 0.952 — the subgroup where an earlier answer is worth anything.

**Negative results have to be kept to be worth anything.** Roughly half the experiments in this
repository changed nothing: rebuilding breast on blood markers failed at chance, reweighting did
not close fairness gaps, exercise made the general panel slightly worse, waist circumference fell
below its pre-registered bar. Each is committed with its result. Without them, the positive
findings are unfalsifiable.

---

## 5. Limitations

**No prospective use, and no IRB.** The NDI analysis is prospective in *design* — exposure measured
before outcome — but it is a secondary analysis of an existing survey. No patient has used this
tool and had the result followed to an outcome. That requires ethics approval granted by an
institution to a named investigator, and no amount of analysis substitutes for it.

**The prospective endpoint is death, not detection.** People who developed cancer and survived it
count as non-cases, because they did. Cancer death is also confounded by everything determining
survivability — stage at presentation, treatment access, insurance, comorbidity — so a model
trained on it partly learns who gets treated. Deaths from other causes inside the window are
treated as non-cases, which is true as stated but is a competing risk that a cause-specific hazard
model would handle more carefully than a binary classifier.

**Four cohorts are case-control.** Breast, pancreatic, ovarian and prostate assemble cases and
match controls to them. Their discrimination does not transfer to a screening population and is
not claimed to.

**Small test sets on the highest-scoring panels.** Prostate is measured on 43 held-out patients,
ovarian on 70, breast on 114. Prostate's confidence interval runs from 0.705 to 0.952, which spans
"barely useful" to "excellent". These are not settled numbers.

**External validation is uneven.** The liver panel is tested across three countries and transfers
badly to one of them — 0.442 on the German cohort, below chance, because ALT and alkaline
phosphatase run in opposite directions between a mild-disease population and an advanced-disease
one. Several panels have no external cohort at all, and no external cohort exists in public data
for the case-control panels.

**Fairness is measured where it can be and unmeasured where it cannot.** Subgroup performance by
race and ethnicity is reported for the NHANES panels. Reweighting was tested and did not close the
gaps. The Wisconsin breast cohort records neither race nor age nor sex, so for that panel the
question is unanswerable rather than answered acceptably.

**The ensemble is often unnecessary.** On several panels a plain logistic regression matches or
beats the ensemble, and where it does, it is what ships.

---

## Data availability

Every cohort is public. Fetchers that reconstruct each one from its original source are in the
repository, so the datasets need not be redistributed. Every experiment reported here is a script
under `experiments/` with its result committed as JSON, including the experiments whose answer was
negative and changed nothing.

## Disclaimer

This is a research prototype, not a medical device. It has no regulatory clearance and no IRB
approval, and no patient has been followed prospectively through the tool itself. Analyses and
interpretations are the author's; NCHS is responsible only for the initial data.
