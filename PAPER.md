# What routine blood work can and cannot tell you about cancer risk

**A feature-constrained multi-panel evaluation on eight public cohorts, with a prospective
mortality analysis and a cost model of triage before expensive diagnostics**

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
The same external test applied to an organ-specific panel gave the opposite answer. The bowel panel,
trained on NHANES 2005–2016 and tested on 14,499 NHANES III adults with 56 colorectal cancers,
transferred a gain of **+0.028** against +0.033 internally — around five sixths of the effect
survived a fifteen-year gap and a change of analysers. Separately, discrimination and usability
diverge sharply: three panels with AUCs between 0.79 and
0.97 flag between 55 and 768 people per true case at real incidence, and a threshold sweep shows
no operating point repairs any of them.

**Cost.** Discrimination is not the outcome that matters for a tool whose purpose is to reduce
spending on diagnostics, so we modelled it directly: per 100,000 people, sending everyone for the
confirmatory procedure against sending only those a panel flags, charging missed cancers the
difference between early and late-stage treatment. At the balanced operating points these panels
ship, triage appears to save $192M per 100,000 on colorectal — by missing 190 of 400 cancers.
Break-even is $1.08M per missed cancer against roughly $2.25M for fifteen life-years at
conventional willingness-to-pay, so the saving disappears once a life is priced. Choosing instead
the point on each panel's ROC curve that maximises net benefit **after** charging a missed cancer
$2.25M: colorectal avoids 36,052 colonoscopies per 100,000 while missing 8.3 of 400 cancers
(+$68.2M), and lung avoids 21,561 CT scans while missing none (+$6.5M).

**Conclusions.** Routine blood work carries usable signal about organ-specific disease when the
relevant analytes are on the panel, and essentially none about undifferentiated cancer risk that
generalises beyond the survey it was fitted on. The two questions differ not merely in effect size
but in whether the effect exists at all: under an identical external test, one gain kept three
quarters of its magnitude and the other reversed sign. Separately, and independently of any model
improvement, the operating point matters more than the model: Youden's J treats a false positive
and a false negative as equally costly, and before an expensive diagnostic they are not. A panel
too weak to screen with can still be strong enough to **rule out** with, and that is a different
and more useful claim. The ceiling is a property of the question rather
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
| Bowel | Population | 28,527 | 114 | 0.40% | NHANES 2005-2016, colon or rectal cancer within 8 years |
| General | Population | 28,711 | 890 | 3.10% | NHANES 2005-2016, cancer diagnosed within 4 years |
| Liver | Population | 35,511 | 1,420 | 4.00% | NHANES, 7 cycles, clinical liver disease |
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

### 2.8 Refusing to extrapolate

A tree has no splits beyond the range of its training data, so past that edge it returns whichever
leaf it lands in, at full confidence, in whichever direction it happens to point. On a clinical
cohort this is not a theoretical concern.

The liver panel scored a coherent acute-hepatitis picture — ALT 300, AST 260, GGT 200, bilirubin
2.5, all rising together as they do in a real patient — at 3.0%, *below* a completely healthy
patient at 3.3% and far below a mild abnormality at 14.1%. The cause was in the data rather than
the code: only 19 of 35,511 people in that cohort have an ALT above 250, and among the 1,436 with
liver disease the highest is 232. Every high-ALT person in the training data is a non-case, so the
model learned that a very high ALT means no liver disease. That is true of NHANES, whose label is
self-reported *"were you ever told you had a liver condition"*, and false of medicine: someone in
acute hepatitis at survey time has not been told yet.

The model is right about its data and wrong about the world, and no retraining on the same cohort
repairs it. Each panel therefore ships the 1st and 99th percentile of every feature as observed;
inputs are clipped to that range before scoring, and anything outside it is declared to the user as
a value the panel cannot rank. The cost of this is measured rather than assumed: across all eight
panels the largest AUC loss is 0.016, on liver, and that loss is the artefact being removed.

### 2.9 Two operating points, because two questions

AUC is a ranking statistic and a threshold is a decision. These panels originally shipped one
threshold each, chosen by Youden's J, which weights a false positive and a false negative equally.

For a tool used **before** an expensive diagnostic that weighting is wrong. A false positive costs a
colonoscopy; a false negative costs a life. Each panel therefore carries a second, looser threshold
and reports both: whether the person is flagged, and separately whether there is enough evidence to
*exclude* them from further testing. The second is stored with what it buys, so a cut is never
presented without its consequences.

The sensitivity that second threshold targets is **not a round number**. An earlier version fixed it
at 95% for every panel, which is an assumption dressed as a standard. It is now read from the cost
model in section 2.10, which computes the sensitivity that maximises net benefit once a missed case
is priced at a life — a figure that differs per panel because it depends on prevalence and on what
the confirmatory procedure costs.

Where that analysis concludes that everyone should be tested, the panel ships **no rule-out at all**
and says so on its own card. Offering one there would invite a person to skip a test the analysis
says they should have, and an absent feature with no explanation reads as an oversight rather than
as a finding.

### 2.10 A cost model

The marginal cost of running a panel is zero: the blood is drawn, the analyser has run, the report
exists. So the relevant comparison is not test against no test, but

    universal   send everyone eligible for the confirmatory procedure
    triaged     send only those the panel flags

per 100,000 people at real incidence, with missed cancers charged the difference between early and
late-stage treatment. Procedure and treatment costs are taken from published US figures and listed
in the source file so they can be argued with. Every input is swept, and the model is reported as
illustrative rather than as a cost-effectiveness analysis: no discounting, no quality-adjusted life
years beyond a single sensitivity figure, and no price on the harm of an unnecessary procedure.

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
| Bowel | 0.815 | 0.82 | 0.843 | +0.033 |
| General | 0.794 | 0.78 | 0.779 | **+0.006** |
| Liver | 0.760 | 0.74 | 0.602 | +0.106 |

The gain column is measured by repeated paired cross-validation on identical folds, not from the held-out split, because a single split proved unreliable.

### 3.2 What a score is worth at real incidence

| Panel | Test AUC | PPV at population incidence | Flagged per true case | Usable as screening? |
|---|---|---|---|---|
| Breast | 0.997 | 95.91% | 1.0 | not a screening panel |
| Pancreatic | 0.969 | 0.47% | 210.4 | not a screening panel |
| Ovarian | 0.949 | 79.33% | 1.3 | not a screening panel |
| Prostate | 0.840 | 70.59% | 1.4 | not a screening panel |
| Lung | 0.829 | 1.81% | 55.1 | no |
| Bowel | 0.815 | 0.13% | 795.0 | no |
| General | 0.794 | 7.79% | 12.8 | yes, with caveats |
| Liver | 0.760 | 9.95% | 10.0 | yes, with caveats |

This is the table that decides whether a panel is a screening instrument. Discrimination and usability are different properties, and three panels have the first without the second.

### 3.3 Stability across resampling

| Panel | Rows | Events | Mean AUC | Spread across splits | Shipped split | Percentile |
|---|---|---|---|---|---|---|
| Breast | 569 | 212 | 0.992 | 0.970 to 1.000 | 0.997 | 77th |
| Pancreatic | 600 | 130 | 0.969 | 0.939 to 0.995 | 0.969 | 50th |
| Ovarian | 349 | 171 | 0.928 | 0.852 to 0.969 | 0.949 | 70th |
| Lung | 21,916 | 104 | 0.839 | 0.822 to 0.860 | 0.829 | 40th |
| Bowel | 28,527 | 113 | 0.823 | 0.788 to 0.859 | 0.815 | 40th |
| Prostate | 212 | 121 | 0.822 | 0.732 to 0.909 | 0.840 | 70th |
| General | 28,711 | 897 | 0.758 | 0.735 to 0.779 | 0.794 | 100th |
| Liver | 35,511 | 1,436 | 0.754 | 0.740 to 0.764 | 0.760 | 80th |
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

### 3.6 The same test on an organ-specific panel

The bowel panel is one of only two here that screen for a named cancer from a routine lab report alone, so it carries more of the application's claim than the case-control panels do. NHANES III recorded both the site of any reported cancer and the age at which it was first told, which reconstructs the same eight-year window the training cohort uses.

Train: NHANES 2005-2014, 28,527 adults, 113 cases. Test: NHANES III, 14,499 adults, 56 cases. The two prevalences agree to within a hundredth of a percent, which is a check that the window was reconstructed the same way on both sides.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| age and sex only | 2 | 0.764 | 0.705 to 0.815 |
| full panel | 16 | 0.792 | 0.74 to 0.841 |

Gain over age and sex, transferred: **+0.028**, against +0.033 measured inside the training survey.

**This gain survives.** Roughly three quarters of it is still there on a cohort measured a decade and a half earlier, on different analysers. Set beside section 3.5, where the undifferentiated panel's gain reversed sign under the same test, this is the sharpest form of the paper's main result: the two questions do not merely differ in effect size, they differ in whether the effect is real at all.

### 3.7 Does triage on free bloodwork save money?

Per 100,000 people at real incidence, sending everyone for the confirmatory procedure against sending only those the panel flags.

| Panel | Procedure | Sent everyone | Sent if flagged | Cases missed | Apparent saving |
|---|---|---|---|---|---|
| Bowel | colonoscopy | 100,000 | 14,036 | 208.8 of 400 | $193,356,533 |
| Lung | low-dose chest CT | 100,000 | 14,999 | 201.6 of 470 | $13,402,557 |
| Liver | FibroScan (transient elastography) | 100,000 | 24,535 | 1575.6 of 4040 | $13,513,852 |

That apparent saving counts only treatment dollars. Charging a missed case what a life is conventionally worth changes the answer. Each panel is valued on its own endpoint: fifteen life-years for a cancer, five for liver disease, at $150,000 per QALY.

| Panel | Break-even per missed case | A case, valued | Verdict |
|---|---|---|---|
| Bowel | $993,037 | $2,250,000 | **stops saving** |
| Lung | $126,481 | $2,250,000 | **stops saving** |
| Liver | $23,948 | $750,000 | **stops saving** |

**The operating point, not the model, decides this.** Choosing the point on each panel's real ROC curve that maximises net benefit once a missed case is priced at a life:

| Panel | Sensitivity | Specificity | Procedures avoided per 100,000 | Cases missed | Net benefit |
|---|---|---|---|---|---|
| Bowel | 0.947 | 0.467 | **46,551** | 21.2 | $64,492,872 |
| Lung | 1.0 | 0.217 | **21,561** | 0.0 | $6,468,172 |
| Liver | 1.0 | 0.0 | **0** | 0.0 | $0 |

**The liver row is the interesting one.** That panel has the largest gain over age and sex of anything in this project, +0.106, and its best operating point is to send everyone: no triage threshold beats universal testing once a missed case is priced. Liver disease is common at 4% and a FibroScan is cheap at $500, so the scans a threshold saves are worth less than the cases it misses. **Discrimination did not decide this; prevalence and procedure cost did.** The panel that separates best is the one where triage helps least, which is the clearest available demonstration that AUC and decision value are different quantities.

An illustrative model, not a cost-effectiveness analysis: no discounting, no quality-adjusted life years beyond the per-panel figure above, and no price on the harm of an unnecessary procedure. The treatment costs are first-year figures and understate the late-stage penalty, which biases the model *towards* triage.
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

The corollary is that the external test is not simply a harsher grader that marks everything down.
Applied to the bowel panel, the identical procedure — same survey, same decade gap, same imputation
discipline — returned +0.029 against +0.039 internally. An organ-specific panel kept three quarters
of its effect while the undifferentiated one reversed sign. That is what distinguishes a transfer
test from a difficulty penalty, and it is why the contrast rather than either number alone is the
finding.

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

### 4.5 The operating point matters more than the model

Sections 4.1 to 4.4 are about what these models know. This one is about what is done with it, and
on the evidence here it is the larger lever.

Every panel originally shipped a single threshold chosen by Youden's J. That statistic maximises
sensitivity plus specificity, which is to say it treats a false positive and a false negative as
equally costly. Before an expensive diagnostic they are not remotely equal: a false positive costs
a colonoscopy, and a false negative costs a life.

The consequence is measurable. At Youden, the colorectal panel flags 14,852 people per 100,000 and
misses 190 of 400 cancers. It appears to save $192M, and the appearance survives only while a
missed cancer is priced at the $67,000 difference between early and late-stage treatment. Priced at
fifteen life-years, the break-even is $1.08M against roughly $2.25M and the saving evaporates.

Move along the same ROC curve — the same model, the same features, the same data — to the point
that maximises net benefit once a missed cancer costs $2.25M, and the panel avoids 36,052
colonoscopies per 100,000 people while missing 8.3 of 400 cancers. The lung panel avoids 21,561 CT
scans and misses none.

Nothing about the model changed. What changed was the question asked of it. **A panel too weak to
screen with can be strong enough to rule out with**, and those are different claims with different
thresholds and different evidentiary burdens.

The converse also holds, and the liver panel demonstrates it. That panel has the largest gain over
age and sex in this project, +0.106, comfortably ahead of colorectal's +0.039. Its optimal
operating point is to send everyone: no threshold beats universal testing once a missed case is
priced. Liver disease is common in this cohort at 4%, and transient elastography costs $500, so the
scans a threshold saves are worth less than the cases it misses.

**Discrimination did not decide either result. Prevalence and procedure cost did.** The panel that
separates best is the one where triage helps least, and the panel with an unremarkable 0.793 is the
one that avoids 36,052 colonoscopies. Any account of a model's usefulness that stops at AUC has no
way of telling those two apart. A literature that reports AUC and a single balanced
operating point is systematically failing to find this, and it is available for free in models that
already exist.

The corollary matters for how this application presents itself. "You are flagged" is a claim about
a person. "There is not enough here to exclude you" is a claim about the evidence, it is the one
these panels can actually support, and it is the one a patient deciding whether to push for a
colonoscopy needs.

### 4.6 A tool can be useful below the accuracy at which it is interesting

The colorectal panel reaches 0.793. By the standards of a modelling paper that is unremarkable, and
it is the number that would be reported. At a rule-out threshold the same panel excludes 40% of
people from consideration for a $2,412 procedure while missing 4 cases in 100, and that is a useful
thing to be able to do with a blood test somebody has already paid for.

The two statements describe one model. The first is what gets published and the second is what
would matter to a health system, and the gap between them is not a modelling problem. It is a
reporting convention.

This does not rescue the panels that fail on other grounds. The general panel's gain does not
survive external validation, and no threshold repairs a signal that is not there. But for the
panels that do transfer, the honest summary is not "moderately accurate". It is: this can safely
take a substantial fraction of people out of an expensive queue, and here is exactly how many
cancers that costs.

### 4.7 Methodological findings

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

**The cost model is illustrative, not a cost-effectiveness analysis.** It has no discounting, no
quality-adjusted life years beyond the single figure used to price a missed cancer, and no price on
the harm, anxiety or complication risk of an unnecessary procedure — only its invoice. Treatment
costs are first-year figures and understate the late-stage penalty, which biases it *towards*
triage. It compares triage against universal screening, not against current practice, which is
neither. The sign of the lung result flips if incidence is five times higher or the late-stage
penalty three times worse, and both sweeps are reported.

**Pricing a life at fifteen years and $150,000 per QALY is a convention, not a fact.** It is the
figure conventionally used in US health economics, it is contested, and the break-even table is
presented so a reader who prefers a different number can read off their own answer rather than
accept ours.

**The rule-out thresholds are chosen on the training cohorts, and only one has been tested outside
one.** They are computed from out-of-fold predictions rather than from the fitted model's own
scores, so they are not fit to the data they are evaluated on. The colorectal cut was then applied
unchanged to NHANES III: it promised to catch 95.6% of cases while excluding 38.6% of people, and
delivered 94.6% and 42.5%, ruling out 3 of 56 cancers that it should not have. That is inside the
5-point tolerance set beforehand and it is a real degradation, so the interface quotes the rate as
approximate rather than exact. The lung, liver and general cuts have no such test, and given
section 4.2 that should be read seriously.

**Clipping to the observed range does not make a panel right about an extreme patient.** It stops
it being confidently backwards. A patient whose ALT is 900 gets the score of a patient at the edge
of the data and a statement that the panel cannot rank them, which is honest and is not the same as
useful. Only a cohort containing such patients would fix that.

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
