# What routine blood work can and cannot tell you about cancer risk

**A feature-constrained multi-panel evaluation on nine public cohorts, with a prospective
mortality analysis and a cost model of triage before expensive diagnostics**

Oncovision AI. Mentored research project, UCI CHOC.

---

## Abstract

**Background.** Most adults have routine blood work drawn every year. A comprehensive metabolic
panel and a complete blood count together produce roughly thirty numbers, which are typically read
one at a time against a reference range. The hypothesis behind this work is that early-detection
signal, where it exists, lies in combinations of values that are individually unremarkable, and
that this is the kind of pattern a model finds and a person scanning a printout does not.

**Methods.** We built nine risk panels under a strict constraint: a model may only use features
the application can actually collect from a patient's own lab report and a short history. Every
dataset column was mapped onto one canonical input schema or dropped before fitting. Panels were
calibrated, evaluated on a held-out split, and then re-evaluated by repeated cross-validation on
identical folds, because a single split proved unreliable. We additionally measured operating
points at real population incidence, subgroup performance by race and ethnicity where the cohort
recorded it, and split stability across many random partitions. The central hypothesis was then
tested on a design none of the training cohorts can support: 33,834 NHANES participants linked by
NCHS to the National Death Index, where blood was drawn years before the outcome existed.

**Results.** The premise holds for some questions and not at all for others, and telling them apart
required a stronger baseline than this work first used. Every gain below is measured against the
better of a logistic and an ensemble model on age and sex alone, on identical folds. Where the
organ's own chemistry is on the lab report, the combination adds a great deal: +0.114 AUC for liver
disease, confirmed on a withheld later survey cycle (+0.091, 95% CI +0.057 to +0.125), and +0.505
for pancreatic adenocarcinoma in a case-control cohort. Lung adds +0.026 inside its survey but was not
confirmed on a withheld cycle holding thirteen cases. For the undifferentiated question — will this
person be diagnosed with any cancer — risk factors add +0.002. **A bowel panel built on sixteen blood
values added nothing**: −0.010 inside its survey and −0.000 on 14,499 NHANES III adults (95% CI
−0.012 to +0.013). An earlier analysis had reported that gain as surviving external transfer; it had
been measured against a tree ensemble given only age and sex, which cannot use age smoothly. 

<!-- AUTOGEN:prospective_short -->
On the prospective cohort of 33,834 adults with NDI-confirmed outcomes, the full panel, which adds BMI, smoking and alcohol to routine blood work, gains +0.025 over the stronger age-and-sex model for cancer death within five years inside its survey; blood work alone gains +0.011 (95% CI +0.002 to +0.020) on NHANES III. That gain survives the transfer.
<!-- /AUTOGEN:prospective_short -->

 A breast panel fitted on
2,392,998 screening mammograms adds +0.028 over age on the consortium's own 597,859-mammogram
validation split (95% CI +0.021 to +0.035). Discrimination and usability diverge sharply: panels with
good AUCs flag dozens to hundreds of healthy people per true case at real incidence.

<!-- AUTOGEN:abstract_cost -->
**Cost.** Discrimination is not the outcome that matters for a tool whose purpose is to reduce spending on diagnostics, so we modelled it directly: per 100,000 people, sending everyone for the confirmatory procedure against sending only those a panel flags, charging missed cancers the difference between early and late-stage treatment. At the balanced operating point the bowel panel ships, triage appears to save $202M per 100,000 — by missing 191 of 400 cancers. Break-even is $1.13M per missed cancer against $2.25M for fifteen life-years at conventional willingness-to-pay, so the saving disappears once a life is priced. Choosing instead the point on each panel's ROC curve that maximises net benefit **after** charging a missed cancer at a life: bowel avoids 36,129 procedures per 100,000 while missing 14.2 of 400 cases (+$55.3M). **But the lab values are not what pays.** Triage on age and sex alone, with no lab values at all, avoids 42,863 procedures, misses 10.6 cancers and nets $79.5M, against $64.5M for the best version of the panel on the same folds. For lung and liver, no threshold beats sending everyone.
<!-- /AUTOGEN:abstract_cost -->

**Conclusions.** Routine blood work carries usable signal about organ-specific disease when the
organ's own chemistry is on the panel — liver most clearly — and a small signal about cancer death
within five years that held on a cohort measured in a different decade. It carried none we could
confirm about a diagnosis of any cancer, or about bowel cancer, and both of those panels were
withdrawn: the one whose triage appeared to save money owed that saving to age. Serum cotinine,
C-reactive protein, the complete blood count, the full metabolic panel, waist circumference and
physical activity were each tested against the diagnosis panel and each rejected. Three
methodological cautions follow, and all concern what a result is compared against. Resampling within
one survey — including leave-one-cycle-out — measures stability and not generalisation. **An
external test validates a model, not a hypothesis**: the prospective blood-work signal first appeared
to reverse on the external cohort, and the reversal belonged to an overfitted tree ensemble rather
than to the signal. And **a baseline is a claim too**: an age-and-sex model that cannot use age
smoothly flatters every panel measured against it, and here it manufactured the bowel panel's
apparent external gain. For the panels that remain, the operating point matters more than the model,
and the number of healthy people flagged per true case matters more than AUC, with which it
frequently disagrees.

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

Nine cohorts, five population-based and four case-control. Cohort design bounds what a number
means, so it is stated on every panel rather than kept in a methods section.

<!-- AUTOGEN:paper_cohorts -->
| Panel | Design | n | Events | Prevalence | Cohort |
|---|---|---|---|---|---|
| Breast (biopsy) | Case-control | 569 | 212 | 37.30% | Wisconsin fine needle aspirates, post-biopsy |
| Pancreatic | Case-control | 600 | 130 | 21.70% | 3 tissue banks, adenocarcinoma vs benign hepatobiliary |
| Ovarian | Case-control | 349 | 171 | 49.00% | operated ovarian masses, malignant vs benign |
| Prostate | Case-control | 212 | 121 | 57.10% | biopsied men, adenocarcinoma vs benign biopsy |
| Lung | Population | 19,866 | 99 | 0.50% | NHANES, adults with measurable tobacco exposure |
| Bowel | Population | 28,527 | 114 | 0.40% | NHANES 2005-2016, colon or rectal cancer within 8 years |
| General | Population | 28,711 | 890 | 3.10% | NHANES 2005-2016, cancer diagnosed within 4 years |
| Liver | Population | 30,624 | 1,164 | 3.80% | NHANES, 7 cycles, clinical liver disease |
| Breast (mammogram) | Population | 400,000 | 2,000 | 0.50% | BCSC screening mammograms, cancer within 1 year |
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
| Breast (biopsy) | 0.997 | 0.995 | — | not measurable |
| Pancreatic | 0.966 | 0.968 | 0.5 | +0.505 |
| Ovarian | 0.949 | 0.911 | 0.813 | +0.174 |
| Prostate | 0.880 | 0.876 | 0.661 | +0.222 |
| Lung | 0.872 | 0.867 | 0.842 | +0.024 |
| ~~Bowel~~ withdrawn | 0.821 | 0.82 | 0.843 | **-0.011** |
| ~~General~~ withdrawn | 0.781 | 0.78 | 0.779 | **+0.002** |
| Liver | 0.780 | 0.761 | 0.623 | +0.114 |
| Breast (mammogram) | 0.628 | 0.628 | 0.608 | +0.028 |

The gain column is measured by repeated paired cross-validation on identical folds, not from the held-out split, because a single split proved unreliable.

### 3.2 What a score is worth at real incidence

| Panel | Test AUC | PPV at population incidence | Flagged per true case | Usable as screening? |
|---|---|---|---|---|
| Breast (biopsy) | 0.997 | 95.91% | 1.0 | not a screening panel |
| Pancreatic | 0.966 | 0.50% | 200.0 | not a screening panel |
| Ovarian | 0.949 | 79.33% | 1.3 | not a screening panel |
| Prostate | 0.880 | 69.51% | 1.4 | not a screening panel |
| Lung | 0.872 | 1.99% | 50.3 | no |
| Bowel | 0.821 | 0.18% | 559.8 | no |
| General | 0.781 | 7.39% | 13.5 | yes, with caveats |
| Liver | 0.780 | 12.47% | 8.0 | yes, with caveats |
| Breast (mammogram) | 0.628 | 1.35% | 74.0 | no |

This is the table that decides whether a panel is a screening instrument. Discrimination and usability are different properties, and three panels have the first without the second.

### 3.3 Stability across resampling

| Panel | Rows | Events | Mean AUC | Spread across splits | Shipped split | Percentile |
|---|---|---|---|---|---|---|
| Breast (biopsy) | 569 | 212 | 0.993 | 0.972 to 1.000 | 0.997 | 43th |
| Pancreatic | 600 | 130 | 0.973 | 0.938 to 0.991 | 0.966 | 27th |
| Ovarian | 349 | 171 | 0.928 | 0.852 to 0.969 | 0.949 | 70th |
| Prostate | 212 | 121 | 0.832 | 0.730 to 0.928 | 0.880 | 80th |
| Lung | 19,866 | 91 | 0.827 | 0.768 to 0.869 | 0.872 | 100th |
| Liver | 30,624 | 1,167 | 0.759 | 0.740 to 0.776 | 0.780 | 100th |
| General | 28,711 | 897 | 0.758 | 0.745 to 0.779 | 0.781 | 100th |
| Breast (mammogram) | 400,000 | 1,974 | 0.616 | 0.600 to 0.633 | 0.628 | 80th |
| Cervical | 858 | 55 | **0.594** | **0.421 to 0.789** | 0.725 | **97th** |

### 3.4 The prospective test

33,834 adults, 339 deaths from malignant neoplasm within 60 months of the blood draw (1.00%).

| Feature set | Features | AUC | 95% CI | Gain over age and sex | Wins |
|---|---|---|---|---|---|
| A age and sex | 2 | 0.836 | 0.816 to 0.854 | +0.000 | 0/5 |
| B + lifestyle | 5 | 0.854 | 0.835 to 0.869 | +0.017 | 5/5 |
| C + blood count | 15 | 0.859 | 0.838 to 0.873 | +0.023 | 5/5 |
| D + chemistry | 16 | 0.859 | 0.838 to 0.872 | +0.023 | 5/5 |
| E everything | 26 | 0.861 | 0.839 to 0.874 | +0.025 | 5/5 |

Leave-one-cycle-out, full feature set. Cycles differ in assay method, field staff and population, so this approximates external validation within one survey.

| Held-out cycle | AUC |
|---|---|
| 1999-2000 | 0.873 |
| 2001-2002 | 0.819 |
| 2003-2004 | 0.879 |
| 2005-2006 | 0.905 |
| 2007-2008 | 0.845 |
| 2009-2010 | 0.873 |
| 2011-2012 | 0.864 |
| 2013-2014 | 0.866 |
| **mean** | **0.865** |

**Routine bloodwork carries prospective signal beyond age and sex.**

### 3.5 Does that gain survive a different decade?

Trained on NHANES 1999-2014 (33,834 adults, 339 deaths) and tested on NHANES III 1988-1994 (14,630 adults, 254 deaths). Identical features on both sides. Nothing from the test cohort touches fitting, calibration or imputation.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| age and sex only, logistic | 2 | 0.861 | 0.841 to 0.88 |
| age and sex only, ensemble | 2 | 0.852 | 0.831 to 0.872 |
| full blood work, logistic | 22 | 0.872 | 0.852 to 0.89 |
| full blood work, ensemble | 22 | 0.839 | 0.819 to 0.858 |

Gain over age and sex, transferred, blood work alone: **+0.011**. For reference, the full panel inside the training survey, which also includes BMI, smoking and alcohol: +0.025.

**The gain survives the transfer.** The better blood-work model scores 0.872 on NHANES III against 0.861 for the better age-and-sex model, a gain of +0.011 (95% CI +0.002 to +0.020).

An earlier version of this section reported the opposite. It fitted a tree ensemble on both arms, found the blood-work arm losing to age and sex by -0.013, and read that as the signal belonging to one survey. The signal did not belong to one survey; the ensemble did. On the same twenty-two features it scores 0.839 externally, where logistic regression transfers. **An external test validates a model, not a hypothesis**, and a failed transfer can belong to the model.

### 3.6 The same test on an organ-specific panel

The bowel panel was one of two here that claimed to screen for a named cancer from a routine lab report alone, so it carried more of the application's claim than the case-control panels did, and it has since been withdrawn for the reason this section records. NHANES III recorded both the site of any reported cancer and the age at which it was first told, which reconstructs the same eight-year window the training cohort uses.

Train: NHANES 2005-2014, 28,527 adults, 113 cases. Test: NHANES III, 14,499 adults, 56 cases. The two prevalences agree to within a hundredth of a percent, which is a check that the window was reconstructed the same way on both sides.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| age and sex only | 2 | 0.764 | 0.705 to 0.815 |
| full panel | 16 | 0.792 | 0.74 to 0.841 |

Gain over age and sex, transferred: **+0.028**, against +0.033 measured inside the training survey.

**That gain belongs to the baseline, not the bloodwork.** Both arms above were fitted with a calibrated tree ensemble, which given only age and a binary sex flag ranks people in coarse steps. Refitted with logistic regression, age and sex alone score 0.822 on NHANES III and the full sixteen-feature panel scores 0.822. Against the stronger baseline the transferred gain is -0.000, 95% CI -0.012 to +0.013, and the original interval, -0.013 to +0.072, never excluded zero either. **On this evidence routine bloodwork adds nothing to age and sex for bowel cancer.** An earlier draft of this paper read the ensemble comparison as a gain that survived transfer; it was a weak baseline.

### 3.7 Does triage on free bloodwork save money?

Per 100,000 people at real incidence, sending everyone for the confirmatory procedure against sending only those the panel flags.

| Panel | Procedure | Sent everyone | Sent if flagged | Cases missed | Apparent saving |
|---|---|---|---|---|---|
| Bowel | colonoscopy | 100,000 | 10,766 | 191.2 of 400 | $202,421,043 |
| Lung | low-dose chest CT | 100,000 | 11,980 | 235.0 of 470 | $12,306,138 |
| Liver | FibroScan (transient elastography) | 100,000 | 18,881 | 1664.5 of 4040 | $14,974,958 |

That apparent saving counts only treatment dollars. Charging a missed case what a life is conventionally worth changes the answer. Each panel is valued on its own endpoint: fifteen life-years for a cancer, five for liver disease, at $150,000 per QALY.

| Panel | Break-even per missed case | A case, valued | Verdict |
|---|---|---|---|
| Bowel | $1,125,687 | $2,250,000 | **stops saving** |
| Lung | $112,367 | $2,250,000 | **stops saving** |
| Liver | $24,368 | $750,000 | **stops saving** |

**The operating point, not the model, decides this.** Choosing the point on each panel's real ROC curve that maximises net benefit once a missed case is priced at a life:

| Panel | Sensitivity | Specificity | Procedures avoided per 100,000 | Cases missed | Net benefit |
|---|---|---|---|---|---|
| Bowel | 0.965 | 0.363 | **36,129** | 14.2 | $55,285,735 |
| Lung | 1.0 | 0.0 | **0** | 0.0 | $0 |
| Liver | 1.0 | 0.0 | **0** | 0.0 | $0 |

**The liver row is the interesting one.** That panel has the largest gain over age and sex of anything in this project, +0.114, and its best operating point is to send everyone: no triage threshold beats universal testing once a missed case is priced. Liver disease is common at 4% and a FibroScan is cheap at $500, so the scans a threshold saves are worth less than the cases it misses. **Discrimination did not decide this; prevalence and procedure cost did.** The panel that separates best is the one where triage helps least, which is the clearest available demonstration that AUC and decision value are different quantities.

An illustrative model, not a cost-effectiveness analysis: no discounting, no quality-adjusted life years beyond the per-panel figure above, and no price on the harm of an unnecessary procedure. The treatment costs are first-year figures and understate the late-stage penalty, which biases the model *towards* triage.
<!-- /AUTOGEN:paper_results -->

### 3.8 A survey cycle withheld from training

Section 4.2 argues that resampling inside one survey is not evidence of transfer. The strongest
answer to that is another source, and for most panels no such source exists in public data. The
always-available substitute is time: NHANES runs in two-year cycles under the same protocol but
with different laboratory runs, a different sample and a slowly moving population. Withholding the
most recent cycle gives a test set that shares the protocol and shares nothing else — no rows, no
calibration, no threshold. TRIPOD calls this temporal validation.

The 2017–2018 cycle is withheld from the liver and lung panels. The general panel, and the since-withdrawn
bowel panel, cannot join: CDC dropped the MCQ240 age-at-diagnosis series after 2015–2016, so their
screening-window target cannot be constructed on later data at all. Those two keep NHANES III.

| Panel | Withheld n | Events | AUC | Age and sex | Gain | 95% CI on the gain |
|---|---|---|---|---|---|---|
| Liver | 4,887 | 269 | 0.705 | 0.614 | **+0.091** | +0.057 to +0.125 |
| Lung | 2,050 | 13 | 0.824 | 0.846 | −0.022 | −0.126 to +0.056 |

**The two results say different things, and the difference is the point.** The liver panel keeps a
gain over age and sex that a bootstrap interval separates from zero, on patients it has never seen,
from a later period. That is the first confirmation in a US population that the panel is reading
chemistry rather than demographics. Its AUC still falls, 0.780 to 0.705, so the number on the card
is optimistic even where the effect is real.

The lung panel's gain does not reproduce. The withheld cycle holds thirteen events and the interval
runs from −0.126 to +0.056, which refutes nothing and confirms nothing. **Unconfirmed is the honest
word, and it is a different claim from the +0.047 that repeated resampling inside the training
cycles reports.** The interface says so on the card rather than quoting only the resampled figure,
because the resampled figure is the one that cannot fail.

<!-- AUTOGEN:lung_loco -->
Holding out every survey cycle in turn -- 10 cycles, 104 lung cancers, each scored by a model that never saw its cycle -- the lung panel scores 0.827 against 0.797 for the stronger age-and-sex model, a gain of +0.030 (95% CI -0.000 to +0.060). The estimate agrees with the in-survey gain, but its range still reaches zero, so the lung panel's advantage over age and sex is probably real and not yet shown.
<!-- /AUTOGEN:lung_loco -->

This is the weaker kind of external validation and is not presented as anything else. A panel that
passes here has survived a change of laboratory run and two years of drift. It has not survived a
change of country, and the liver panel is the standing proof that those are different questions: it
scores 0.442 on the German cohort, below chance.

One byproduct is worth recording. The liver panel previously trained on a multi-cycle file that
included 2017–2018 while `external_validation.py` listed that same cycle as an independent "USA"
cohort beside India and Germany — all 4,887 rows matched. No published number was wrong, because
that file refits models rather than loading the shipped one, but the framing invited a reader to
believe the shipped panel had been tested on a cohort it had been trained on. The holdout makes the
claim true rather than merely unstated, and an assertion now fails loudly if the holdout is removed.

### 3.9 A breast panel built on a screening population

The breast panel that shipped reads thirty nuclear measurements off a fine needle aspirate and
scores 0.997 on 569 Wisconsin biopsies. That number is real and it answers the wrong question for
this tool: the patient already has a biopsy, and 37% of the cohort is malignant. No sample size
turns a case-control set of post-biopsy lesions into a screening population. It stays, labelled
as interpretation.

The Breast Cancer Surveillance Consortium publishes a risk-estimation dataset that is the thing
the Wisconsin cohort lacks by construction: **2,392,998 screening mammograms, 11,638 followed by a
breast cancer within one year (0.49%)**, with risk factors recorded at the mammogram and the
cancer diagnosed after it, and a training/validation split specified by the consortium. A second
breast panel was fitted on a 400,000-mammogram sample of the training split, using what a woman
holding a mammogram report can answer — age, BMI, BI-RADS density, first-degree family history,
previous biopsy, previous mammogram result, age at first birth, menopause type and hormone
therapy. Race and ethnicity are recorded and are not features. Age and BMI are banded in the
source and mapped to band midpoints, so the feature is on the scale the service receives.

It was judged on the consortium's validation split, which it never saw — every statistic weighted
by count, intervals from a Poisson bootstrap over mammograms:

| | Result |
|---|---|
| Mammograms / cancers | 597,859 / 2,871 |
| AUC | **0.623** (0.613 to 0.633) |
| Age alone | 0.595 |
| Gain over age | **+0.028** (+0.021 to +0.035) |
| Rule-out cut, promised | catch 95.1%, exclude 11.0% |
| Rule-out cut, delivered | catch 95.3%, exclude 11.6% |
| Subgroups | White 0.625, Black 0.622, Asian or Pacific Islander 0.605; two groups under 50 cancers, not measured |

**0.623 is the lowest AUC on the site and the best-evidenced number on it.** Published BCSC and
Gail-type models land between 0.58 and 0.66; a screening model scoring 0.9 on this question would
be measuring something other than screening risk. The gain over age is small, its interval is the
tightest in the project, and it is measured on more held-out events than every other panel's test
sets combined. Its rule-out cut is the only one here whose delivered rate matched the promised rate
to within a point, which is what a cut tuned and tested on the same population at its real
prevalence should do and what the enriched cohorts cannot.

Two alternatives were tried first and failed, and both are worth recording. Rebuilding prostate on
NHANES PSA looked promising — 4,697 men with a measured PSA and 253 reported prostate cancers — until
the cases were checked: only 17 of the 253 had a PSA measured, because NHANES did not draw it in men
with a prostate cancer history, and those 17 had a *lower* median PSA than controls (0.85 against
1.0), because they had been treated. A model trained on that would have learned that low PSA means
cancer. Routine bloodwork on NHANES adds nothing for prostate at any diagnosis window either (gain
0.000 on 373 events). The ovarian and prostate screening arms of PLCO — CA-125 on 78,000 women,
PSA on the prostate arm — would answer both properly, and are available only through an NCI
project proposal with a named investigator.

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

**Where the question is undifferentiated, the answer depends on which question.** Asked "will this
person be diagnosed with any cancer", the general panel added about 0.002 over age and sex, its
rule-out call did no better than age and sex, and it was withdrawn. Serum cotinine, C-reactive
protein, the complete blood count, the full metabolic panel, waist circumference and physical
activity were each measured against that panel and each rejected. Asked prospectively, whether a
person will die of cancer within five years:

<!-- AUTOGEN:prospective_short -->
On the prospective cohort of 33,834 adults with NDI-confirmed outcomes, the full panel, which adds BMI, smoking and alcohol to routine blood work, gains +0.025 over the stronger age-and-sex model for cancer death within five years inside its survey; blood work alone gains +0.011 (95% CI +0.002 to +0.020) on NHANES III. That gain survives the transfer.
<!-- /AUTOGEN:prospective_short -->

That is a real signal and a small one. It sits on a death endpoint rather than a diagnosis, which the
limitations section treats at length, and at that size it does not make a screening instrument. But
it is the one undifferentiated result in this project that has survived a cohort from another
decade, and an earlier draft of this paper reported it as having reversed (section 4.2).

### 4.2 What an external cohort is actually testing

Two results in this project changed when they met a cohort from another decade, in opposite
directions, and neither change meant what it first appeared to.

The prospective blood-work panel looked like it worked inside NHANES: every paired repeat beat age
and sex, and holding out an entire survey cycle at a time gave a high mean AUC (section 3.4).
Resampling within one survey cannot settle that, because cycles share a protocol, a laboratory
contract, instrument calibration and an analytic pipeline; it varies the participants while holding
the measurement process fixed. So the panel was applied to NHANES III, and it appeared to fail: the
blood-work arm scored below age and sex alone.

That failure belonged to the model. Both arms had been fitted with a calibrated tree ensemble, and
the ensemble on twenty-two blood values had learned the training survey closely enough to lose to
age and sex on another one. Fitted as a logistic regression, the same features transfer, and the
gain over the better age-and-sex model holds on NHANES III with an interval above zero (section
3.5). **An external test validates a model, not a hypothesis.** A transfer that fails can mean the
model overfitted rather than that the signal was never there, and the only way to tell the two apart
is to try a model that cannot overfit in the same way.

The bowel panel went the other way. The identical procedure appeared to keep most of an
organ-specific gain, which an earlier draft read as proof that the external test is not simply a
harsher grader. That gain was an artefact of the baseline: given only age and a binary sex flag, a
tree ensemble ranks people in coarse steps, and against a logistic age-and-sex model the bowel
panel's transferred gain cannot be distinguished from zero (section 3.6). It was withdrawn.

Put together, the lesson is about what each number is compared with. An overfitted model made a real
signal look absent; an under-fitted baseline made an absent signal look real. **A baseline is a claim
too, and so is a model**, and every gain in this paper is now measured with both model kinds on both
sides of the comparison. The panels here that have no external cohort at all should still be read
with the first half of this section in mind.

### 4.3 Why that distinction matters more than any AUC here

The headline numbers in this project run from 0.73 to 0.997, and the highest are the least
informative. Breast at 0.997 and pancreatic at 0.969 are case-control designs: they separate known
cases from selected controls, on cohorts of 569 and 600. Read as screening performance they are
badly misleading, and the taxonomy exists to stop them being read that way.

The tool ships seven panels. One screens for a named cancer from a lab report alone, lung. One
estimates breast cancer risk from a mammogram report at real screening prevalence. Four require the
patient to already be inside the diagnostic pathway, and one detects liver disease rather than liver
cancer. A bowel panel and a general cancer-risk panel were withdrawn when neither could be shown to
add anything to age and sex. That sentence is a more honest summary of the work than any table of
discrimination statistics.

### 4.4 Discrimination is not usability

Panels with good discrimination can still have no usable operating point. Projected onto real
incidence, the pancreatic panel flags roughly 200 people for every true case and lung about 50, and
the withdrawn bowel panel flagged several hundred. A threshold sweep confirms that no operating point
brings them to a defensible ratio: raising specificity far enough to fix precision destroys
sensitivity first.

This is arithmetic, not a modelling deficiency. At an incidence of 13.9 per 100,000, no classifier
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

<!-- AUTOGEN:operating_point -->
The consequence is measurable. At Youden, the colorectal panel flags 10,766 people per 100,000 and misses 191 of 400 cancers. It appears to save $202M, and the appearance survives only while a missed cancer is priced at the $67,000 difference between early and late-stage treatment. Priced at fifteen life-years, the break-even is $1.13M against $2.25M and the saving evaporates.

Move along the same ROC curve — the same model, the same features, the same data — to the point that maximises net benefit once a missed cancer costs $2.25M, and the panel avoids 36,129 colonoscopies per 100,000 people while missing 14.2 of 400 cancers.
**But the lab values are not what pays.** Triage on age and sex alone, with no lab values at all, avoids 42,863 procedures, misses 10.6 cancers and nets $79.5M, against $64.5M for the best version of the panel on the same folds.
For lung and liver, no point on the curve beats sending everyone, which is why those panels offer no rule-out call.
<!-- /AUTOGEN:operating_point -->

Nothing about the model changed. What changed was the question asked of it. **A panel too weak to
screen with can be strong enough to rule out with**, and those are different claims with different
thresholds and different evidentiary burdens.

The converse also holds, and the liver panel demonstrates it. That panel has the largest gain over
age and sex in this project, far ahead of any other population panel (section 3.7). Its optimal
operating point is to send everyone: no threshold beats universal testing once a missed case is
priced. Liver disease is common in this cohort at 4%, and transient elastography costs $500, so the
scans a threshold saves are worth less than the cases it misses.

**Discrimination did not decide either result. Prevalence and procedure cost did.** The panel that
separates best is the one where triage helps least, and the one triage appeared to help was bowel, where the saving turned out to be the patient's age rather than the lab report. Any account of a model's usefulness that stops at AUC has no
way of telling those two apart. A literature that reports AUC and a single balanced
operating point is systematically failing to find this, and it is available for free in models that
already exist.

The corollary matters for how this application presents itself. "You are flagged" is a claim about
a person. "There is not enough here to exclude you" is a claim about the evidence, it is the one
these panels can actually support, and it is the one a patient deciding whether to push for an
expensive test needs.

### 4.6 A tool can be useful below the accuracy at which it is interesting

The mammogram-report breast panel reaches 0.623, the lowest AUC on the site. By the standards of a
modelling paper that is unremarkable, and it is the number that would be reported. On the
consortium's 597,859-mammogram validation split, its rule-out cut excluded 11.6% of women while
catching 95.3% of the cancers that followed, which is the rate it promised to within a point.

The two statements describe one model. The first is what gets published and the second is what
would matter to a health system, and the gap between them is not a modelling problem. It is a
reporting convention.

This section used to make the argument with the bowel panel, which at a rule-out threshold took a
large share of people out of the colonoscopy queue. That was true, and it was not the panel's doing:
the same search run on age and sex alone took out a similar share and missed half as many cancers.
So the argument needs a condition it did not have. **A tool can be useful below the accuracy at which
it is interesting only if it is useful beyond what the patient's age already says**, and that has to
be measured rather than assumed. For the breast panel's cut it has now been measured:

<!-- AUTOGEN:breast_vs_age -->
At the same share of cancers caught, 95.3%, on the 597,859-mammogram validation split, the panel's cut excluded 11.6% of women and a cut on age alone 1.8%: a difference of +9.8%, 95% CI +8.3% to +10.0%. **Unlike the bowel panel, this one earns its extra questions**: the density grading and history exclude materially more women than their age does, without catching fewer cancers.
<!-- /AUTOGEN:breast_vs_age -->

 No threshold repairs a signal that is not there.

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

**Four cohorts are case-control.** The biopsy breast panel, pancreatic, ovarian and prostate
assemble cases and match controls to them. Their discrimination does not transfer to a screening
population and is not claimed to. Breast is the one cancer where that is no longer the only
evidence: the mammogram-report panel in section 3.9 is fitted and validated on a screening
population at its real prevalence. Pancreatic has no screening population anywhere to build on,
because no screening programme for it exists. Ovarian and prostate do, in PLCO, behind an
application this project has not made.

**Small test sets on the highest-scoring panels.** Prostate is measured on 43 held-out patients,
ovarian on 70, breast on 114. Prostate's confidence interval runs from 0.705 to 0.952, which spans
"barely useful" to "excellent". These are not settled numbers.

**External validation is uneven.** The liver panel is tested across three countries and transfers
badly to one of them — 0.442 on the German cohort, below chance, because ALT and alkaline
phosphatase run in opposite directions between a mild-disease population and an advanced-disease
one. The general panel was tested on NHANES III, and liver and lung on a withheld 2017–2018 cycle; the
bowel panel's apparent transfer to NHANES III was an artefact of a weak baseline (section 3.6).
A withheld cycle of the same survey is a weaker test than another country and section 3.8
does not claim otherwise. The lung result is uninformative either way on thirteen events. No
external cohort exists in public data for any of the four case-control panels, and none is likely
to: they are assembled from cases and matched controls, and a screening population is the thing
they lack by construction.

**Fairness is measured where it can be and unmeasured where it cannot.** Subgroup performance by
race and ethnicity is reported for the NHANES panels. Reweighting was tested and did not close the
gaps. The Wisconsin breast cohort records neither race nor age nor sex, so for that panel the
question is unanswerable rather than answered acceptably.

**The cost model is illustrative, not a cost-effectiveness analysis.** It has no discounting, no
quality-adjusted life years beyond the single figure used to price a missed cancer, and no price on
the harm, anxiety or complication risk of an unnecessary procedure — only its invoice. Treatment
costs are first-year figures and understate the late-stage penalty, which biases it *towards*
triage. It compares triage against universal screening, not against current practice, which is
neither. Every input is swept and the sweeps are
reported, so a reader can see which inputs would move the sign.

**Pricing a life at fifteen years and $150,000 per QALY is a convention, not a fact.** It is the
figure conventionally used in US health economics, it is contested, and the break-even table is
presented so a reader who prefers a different number can read off their own answer rather than
accept ours.

**The rule-out thresholds are chosen on the training cohorts, and they degrade when moved.** They
are computed from out-of-fold predictions rather than from the fitted model's own scores, so they
are not fit to the data they are evaluated on. The two population-cohort cuts that were applied
unchanged to NHANES III both lost ground, and the bowel panel has since been withdrawn, so of these
only the general cut still ships:

| Cut | Promised | Delivered | Wrongly ruled out |
|---|---|---|---|
| Colorectal | catch 95.6%, exclude 38.6% | catch 94.6%, exclude 42.5% | 3 of 56 |
| General | catch 95.1%, exclude 22.2% | **catch 90.5%**, exclude 27.6% | **24 of 252** |

Both are inside the 5-point tolerance set beforehand, but the general cut clears it by four tenths
of a point, and it is the cut with the widest reach: it excludes roughly a fifth of everyone who
runs the panel, on five questions and no blood test. Both also excluded *more* people than
promised while catching fewer, which is the direction that matters — a cut that drifts this way
reassures more people on less evidence. The interface quotes each panel's own transferred figures
rather than the tuned ones, and rounds them as approximate.

<!-- AUTOGEN:general_vs_age -->
Inside its own survey, at 95.3% of cancers caught, the panel's cut excluded 22.7% of adults and a cut on age and sex alone 20.5% (+2.1%, 95% CI -3.0% to +4.3%); on NHANES III, at 95.3% of cancers caught, the panel's cut excluded 15.4% of adults and a cut on age and sex alone 13.4% (+2.1%, 95% CI -5.6% to +8.6%). **The general panel's rule-out call does not beat age and sex**, which is the rule that withdrew the bowel panel.
<!-- /AUTOGEN:general_vs_age -->

The mammogram-report breast cut is the exception, and the reason is design rather than luck: tuned
and tested at real screening prevalence on the consortium's own split, it promised to catch 95.1% and
exclude 11.0%, and delivered 95.3% and 11.6%. Section 4.6 tests whether it beats a cut on age alone, which is the question that
withdrew the bowel panel. The four case-control panels
(biopsy breast, ovarian, pancreatic, prostate) also ship cuts, and no public cohort exists to test
those against. Liver and lung ship no cut at all: the cost model says
everyone in those groups should have the confirmatory test regardless. Given section 4.2, the
untested case-control cuts should be read with the same suspicion the discrimination figures earn.

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
