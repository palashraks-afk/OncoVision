# Oncovision

A full stack platform that reads standard patient lab reports plus information about the patient
and returns multi-cancer risk assessments in real time.

Built as a mentored research project under the guidance of a clinical oncologist at UCI CHOC.

**[Live app](https://oncovisionai.vercel.app)** · **[Methodology](PROJECT.md)** · **[Full evaluation](EVALUATION.md)** · **[Validation protocol](PROTOCOL.md)**

> **This is a research prototype, not a medical device.** It has no regulatory clearance, no
> prospective validation, and no IRB approval. Read [the honest numbers](#honest-numbers) before
> reading anything else.

---

## What it does

Two inputs, read together:

1. **Your lab reports.** 64 values across body metrics, complete blood count and red cell
   indices, metabolic panel, liver panel, tumour markers, tobacco exposure and inflammation,
   the prostate work-up, and breast mass morphology. Uploaded as PDFs and parsed automatically,
   or typed in. The parser is tested on 28 of them at 100 percent across five different report
   layouts; see `test_parser.py`.
2. **Information about you.** 8 items no lab report contains: sex, smoking, pack-years, alcohol,
   hepatitis B and C, diabetes and menopausal status. The 14 sexual-history questions went out
   with the cervical panel, and the exercise question went when it was measured and found to add
   nothing, because asking for what nothing uses is only friction.

Eight calibrated models score the combination, and the interface reports what drove each score,
how accurate that model is, and what the score is worth at real prevalence.

### What "multi-cancer" means here, precisely

Eight panels is not eight cancers you can screen for, and the difference is the most important
thing on this page. Each panel carries its type on its own card, and the types are not
interchangeable:

| | Panels | What you need before running it |
|---|---|---|
| **Screening** | Bowel, lung, plus a general panel | Nothing but a lab report |
| **Triage** | Pancreatic, ovarian | A mass or a suspicion already found |
| **Interpretation** | Breast, prostate | A biopsy or MRI already performed |
| **Not a cancer panel** | Liver | It detects liver *disease*, not liver cancer |

So: **six named cancers, of which two can be screened for from a lab report alone.** The other
four are triage and interpretation tools for someone already inside the diagnostic pathway,
which is a genuinely useful thing to be and a different thing from screening.

Both screening panels then run into the precision problem below, which is measured rather than
argued about. That is the honest headline, and everything underneath it is the evidence.

Every panel answers with whatever it is given. A blank field is filled with that feature's
training median, and the panel says how many of its inputs were actually yours, so a score
built on three values out of twenty-seven is labelled low confidence rather than refused or
presented as if it were complete.

---

## What ships

Eight panels ship. One was withdrawn because the evidence did not support serving it.

<!-- AUTOGEN:shipped -->
| Panel | Trained on | Test AUC | 95% CI | Threshold | Sens | Spec | Flagged per true case |
|---|---|---|---|---|---|---|---|
| Breast malignancy | 569 Wisconsin biopsies | 0.997 | 0.99 to 1.0 | 29.8% | 0.976 | 0.986 | 1.0 |
| Pancreatic cancer | 600 samples, 3 tissue banks | 0.969 | 0.938 to 0.991 | 73.0% | 0.731 | 0.979 | 210.4 |
| Ovarian malignancy | 349 operated ovarian masses | 0.949 | 0.886 to 0.994 | 58.1% | 0.853 | 0.944 | 1.3 |
| Prostate cancer | 212 biopsied men | 0.840 | 0.705 to 0.952 | 66.9% | 0.8 | 0.778 | 1.4 |
| Lung cancer | 21,916 adults with tobacco exposure | 0.829 | 0.73 to 0.902 | 1.0% | 0.571 | 0.852 | 55.1 |
| Bowel cancer | 23,794 NHANES adults | 0.793 | 0.708 to 0.867 | 1.0% | 0.526 | 0.853 | 768.4 |
| Liver disease | 35,511 NHANES adults | 0.760 | 0.73 to 0.789 | 4.0% | 0.61 | 0.77 | 10.0 |
| General cancer | 23,923 NHANES adults | 0.732 | 0.692 to 0.77 | 2.9% | 0.68 | 0.65 | 16.9 |
| ~~Cervical~~ | 858 Caracas referrals | 0.725 | withdrawn, a lucky split | | | | |
<!-- /AUTOGEN:shipped -->

### Does each panel beat the obvious baseline?

<!-- AUTOGEN:baselines -->
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
<!-- /AUTOGEN:baselines -->

Bowel is quoted from 20 paired repeats rather than one split, because its single
split disagreed with its cross-validation. Prostate is the one panel where plain
logistic regression beats the ensemble on the held-out split, 0.876 against
0.840, which is stated rather than buried; on repeated splits the two are within
noise of each other.

Bowel is quoted from 20 paired repeats rather than a single split, for the
reason given under "Colorectal had the same conflict" below. Its single-split
figures are 0.793 against an 0.817 baseline, which is why one split was not
allowed to decide it.

The general panel is the weak one, and the interface says so rather than hiding it.

### What each panel actually is, now visible to the user

The single biggest gap between this project and its own documentation was that
the documentation knew four of the eight panels are not screening tests and the
interface did not. Someone opening the app saw eight cancer risks in one list
and had no way to tell that two of them require a biopsy or an MRI they will
never have, and that three of the remaining ones would flag dozens or hundreds
of healthy people per cancer found.

Every result card now carries its type.

| Panel | Type | What you must already have |
|---|---|---|
| General | Screening | Nothing. Routine bloodwork and your history. |
| Liver | Screening | Nothing. Routine bloodwork and your history. |
| Bowel | Screening | Nothing. A routine blood count. |
| Lung | Screening | Nothing, but it is offered to people with tobacco exposure. |
| Pancreatic | **Triage** | A CA 19-9 result, which is ordered when pancreatic or biliary cancer is already suspected. |
| Ovarian | **Triage** | An ovarian mass already found on imaging. |
| Prostate | **Interpretation** | A PI-RADS score from a prostate MRI. |
| Breast | **Interpretation** | Nuclear measurements from a biopsy already taken. |

A triage panel asks whether something already found is malignant. An
interpretation panel reads a diagnostic test that has already been performed.
Neither can be run from a lab report, and pretending otherwise is the main way
this application could mislead someone.

The pancreatic row was missing from this table for a while, and the reason is
worth recording. `PANEL_KIND` had no entry for it, and the lookup that reads
that table defaulted to `("screening", "")`. So the panel silently claimed the
strongest label available, with no note explaining itself, purely because nobody
had written a line for it. It reads CA 19-9, a tumour marker ordered when cancer
is already suspected, and its controls were people with benign disease of the
same organs, so it was always triage. The default is now gone: a panel with no
declared type raises at training time rather than assuming one, and
`tools/audit.py` checks for the omission separately.

### Three panels are labelled "not a screening test" on their own card

Separately from what a panel needs, there is the question of whether acting on
it would be defensible. A panel flagging hundreds of healthy people per true
case is not a screening instrument whatever its AUC says, and
`experiments/operating_point.py` showed that for three of them no threshold
setting fixes it.

| Panel | Flagged per true case | Card says |
|---|---|---|
| Bowel | 768 | Not a screening test |
| Pancreatic | 210 | Not a screening test |
| Lung | 55 | Not a screening test |
| General | 17 | (no warning) |
| Liver | 10 | (no warning) |
| Prostate | 1.4 | (no warning) |
| Ovarian | 1.3 | (no warning) |
| Breast | 1.2 | (no warning) |

The threshold for that warning is 50 flagged per case, set in `train_models.py`
as `NOT_SCREENING_ABOVE`, and it is applied automatically from the measured
precision rather than hand-assigned, so a panel cannot quietly drift past it.

### The precision problem, attacked properly

The worst number in this project was never an AUC. It was that the pancreatic
panel flagged roughly 843 people for every true case and the bowel panel 768.
Two causes were tangled together, and separating them fixed one of them.

**Cause one: the threshold ignored prevalence.** Every panel picked its
operating point by Youden's J, which maximises sensitivity plus specificity.
That weights a false positive and a false negative equally and never looks at
how rare the disease is, so on a cancer at 0.01 percent prevalence it happily
chooses a cut that drowns the true cases. The chooser now takes the prevalence
the panel will actually meet, and when Youden lands somewhere with poor
precision it re-picks to maximise precision subject to keeping sensitivity at or
above 70 percent. Pancreatic went from 843 flagged per case to **210**, giving
up sensitivity from 1.00 to 0.73.

**Cause two: two panels were being projected onto the wrong population.** The
breast panel reads a fine needle aspirate that has already been taken because a
lesion was found. Projecting it onto SEER population incidence asks what would
happen if it were run on every woman in the country, which is not what it does
and never could be. That is the identical error caught earlier on cervical.
Corrected to the roughly 25 percent malignancy rate among breast lesions taken
to biopsy, breast goes from 52.7 flagged per case to **1.2**, with sensitivity
unchanged at 0.81. Nothing about the model changed; the denominator was wrong.

| Panel | Before | After | What changed |
|---|---|---|---|
| Breast | 52.7 | **1.2** | wrong prior, corrected to post-biopsy |
| Pancreatic | 842.8 | **210.4** | precision-aware threshold |
| Bowel | 768.4 | 768.4 | nothing helps, see below |
| Lung | 55.1 | 55.1 | nothing helps, see below |
| Ovarian | 1.3 | 1.3 | already fine |
| Prostate | 1.4 | 1.4 | already fine |
| Liver | 9.9 | 9.9 | already fine |
| General | 16.9 | 16.9 | already fine |

### Three panels have no usable operating point, at any threshold

This is the part that does not get fixed, and it is worth stating as a finding
rather than a disclaimer. Sweeping the entire threshold range and asking whether
any point reaches 20 flagged per true case while keeping useful sensitivity:

| Panel | Youden | At 99% specificity | Reaches 20 per case? |
|---|---|---|---|
| Pancreatic | 826 per case | 130 per case, sens 0.59 | **no** |
| Bowel | 696 per case | 658 per case, sens 0.04 | **no** |
| Lung | 52 per case | 31 per case, sens 0.08 | **no** |

Bowel is the starkest: tightening all the way to 99 percent specificity moves it
from 696 to 658 while sensitivity collapses to 4 percent. There is no setting of
this dial that makes a population bowel screen out of a complete blood count.
That is a property of a cancer at 0.037 percent prevalence, not of the model,
and no better model changes it.

So those three panels are honest risk indicators and are not population
screening instruments, whatever their AUCs say. Reproduce with
`python experiments/operating_point.py`.

### Fairness: I was wrong that this could not be measured

The documentation said, in several places and for a long time, "no race or
ethnicity in any cohort, so accuracy across those groups is unmeasured rather
than acceptable". The second half was the right instinct. The first half was
false, and false in the direction that let the project off the hook.

NHANES records race and ethnicity for every participant, and four of the eight
shipped panels are built on NHANES. Half the product was measurable the whole
time and simply had not been measured.

Race is carried as a **stratifier and never as a feature**. Using it as a
predictor bakes population averages in as though they were biology, which is
what medicine has spent recent years removing from estimates like eGFR. But not
recording it means fairness cannot be checked, which is worse. So it rides in
the data, is excluded from every feature list, and exists to answer this
question.

| Panel | Overall | Weakest measurable group | | Gap |
|---|---|---|---|---|
| Lung | 0.823 | Non-Hispanic Black | 0.756 | **-0.066** |
| Bowel | 0.804 | Non-Hispanic Black | 0.752 | **-0.052** |
| General | 0.757 | Other Hispanic | 0.672 | **-0.085** |
| Liver | 0.743 | Other Hispanic | 0.713 | -0.030 |

Three panels have a gap wide enough to matter, and the direction is the bad one.
Black Americans have higher mortality from both colorectal and lung cancer, so a
panel that ranks them less well is not a neutral gap, it compounds a disparity
that already exists. That now appears on the panel itself rather than in a
footnote, next to the score, phrased as a reason to weigh the result less.

The liver panel is the one clean result: Non-Hispanic Black is its **best**
group at 0.798 against 0.743 overall.

Several groups cannot be scored at all because the cancers are rare enough that
a subgroup carries under ten events. Those are reported as unmeasurable rather
than given a number, because a subgroup AUC on two cases is noise and printing
it would be worse than admitting the gap.

**Still genuinely unmeasurable:** breast, ovarian, pancreatic and prostate. The
Wisconsin, Soochow, tissue-bank and transperineal-biopsy cohorts do not record
race, and no analysis of them can change that.

Reproduce with `python experiments/fairness.py`.

### Does each panel beat just reading one line of the lab report?

The prostate panel was held to this bar: it had to beat PSA on its own, because
every man in that cohort already had a PSA drawn and a model that only matches it
is worth nothing. That was a fair test applied to exactly one panel, which is not
a standard. So it was applied to all of them, on identical folds, ten repeats.

| Panel | Full panel | Best single marker | Gain | Wins |
|---|---|---|---|---|
| Pancreatic | 0.966 | CA 19-9 alone 0.936 | +0.030 | 10/10 |
| Breast | 0.956 | Nuclear area alone 0.923 | +0.032 | 10/10 |
| Ovarian | 0.931 | CA 125 alone 0.808 | +0.124 | 10/10 |
| Ovarian | 0.931 | CA 125 + HE4, the ROMA pair 0.889 | +0.043 | 10/10 |
| Prostate | 0.828 | PSA alone 0.670 | +0.158 | 10/10 |
| Prostate | 0.828 | PI-RADS alone 0.809 | +0.019 | 10/10 |
| Lung | 0.821 | Serum cotinine alone 0.625 | +0.196 | 10/10 |
| Lung | 0.821 | Age, sex and smoking 0.792 | +0.029 | 10/10 |
| Bowel | 0.810 | Haemoglobin alone 0.518 | +0.291 | 10/10 |
| Liver | 0.741 | ALT + AST alone 0.653 | +0.088 | 10/10 |

Every panel beats its own strongest single input, and every one wins all ten
repeats. Two of these are worth pulling out.

The pancreatic panel is **not** CA 19-9 wearing a hat, which was the specific
worry that prompted the check: it adds 0.030 over reading that number alone.

The ovarian panel beats **the ROMA pair**, CA 125 with HE4, by 0.043. That is the
combination used clinically to triage an adnexal mass, so this is the one place
in the project where a panel measurably improves on the standard-of-care index
rather than merely reproducing it.

The narrowest margins are prostate over PI-RADS (+0.019) and lung over the
smoking question (+0.029). Both are consistent across every repeat, but both are
small, and neither panel would be worth much without the input it is only
modestly improving on.

Reproduce with `python experiments/single_marker_check.py`.

### The seven open problems, worked through

Each of these was carried in the documentation as an open defect. Three turned
out to be real and fixable, two were misdiagnosed by me and the real cause was
different, one is genuinely unfixable from the available data, and one cannot be
fixed by writing code at all.

**1. Colorectal quoted an unrepresentative split. Fixed.**
Its held-out split gives 0.793 against an 0.817 age-and-sex baseline, so on that
one draw the panel loses to its own baseline, while over twenty paired repeats it
beats it by 0.038. Every panel now carries the mean across repeated splits
alongside the single split, and the interface shows the stable number first.
A single split is one draw; that is how the cervical panel came to publish 0.725
when its real mean was 0.594.

**2. Prostate: isotonic calibration was NOT the problem.**
The suspicion was that isotonic calibration fitted inside 169 rows was flattening
the ranking and costing AUC against plain logistic regression. Measured across
three calibration methods on every small panel, that is wrong:

<!-- AUTOGEN:calibration -->
| Panel | n | AUC none | AUC isotonic | AUC sigmoid | Brier none | Brier isotonic |
|---|---|---|---|---|---|---|
| Breast | 569 | 0.993 | 0.992 | 0.993 | 0.0276 | 0.0267 |
| Pancreatic | 600 | 0.969 | 0.966 | 0.968 | 0.0585 | 0.0533 |
| Ovarian | 349 | 0.935 | 0.933 | 0.935 | 0.0985 | 0.0889 |
| Prostate | 212 | 0.829 | 0.83 | 0.831 | 0.1661 | 0.1664 |
| Lung | 21,916 | 0.825 | 0.819 | 0.823 | 0.0273 | 0.0047 |
| Bowel | 23,794 | 0.808 | 0.805 | 0.814 | 0.0288 | 0.004 |
| General | 23,923 | 0.756 | 0.756 | 0.758 | 0.1522 | 0.0294 |
| Liver | 35,511 | 0.747 | 0.749 | 0.751 | 0.1641 | 0.036 |
<!-- /AUTOGEN:calibration -->

Prostate spans 0.002 across all three methods, and no panel anywhere moves by
enough to justify a change. The first version of this test only covered the
small panels, on the argument that isotonic overfitting is a small-sample
problem; that argument is probably right and was still an assertion, so the large
NHANES panels were measured too.

The Brier columns are the real result. Calibration barely touches ranking, plus
or minus 0.003, and improves the probabilities by four to six times. That is
exactly its job: the interface prints a percentage and the percentage has to mean
something. Isotonic stays everywhere.

The 0.840 against 0.876 gap on prostate is therefore split noise on a 43-row test
set whose repeated-split spread is 0.732 to 0.909. Not a defect.
Reproduce with `python experiments/calibration_method.py`.

**3. Liver in Germany: the "units bug" guess was wrong. It is case mix.**
This was carried for a long time as "0.442, below chance, probably a units or
encoding mismatch, unresolved". Both halves were wrong.

The raw CSV was downloaded from archive.ics.uci.edu and compared value by value
against what `ucimlrepo` returns. They match exactly, so there is no column
misalignment. The unit conversions were checked and are correct.

The real cause is that two of the eight shared markers point in **opposite
directions** in the two cohorts:

| Marker | NHANES | Germany | |
|---|---|---|---|
| AST | 0.657 | 0.938 | agree |
| Bilirubin | 0.516 | 0.836 | agree |
| Alkaline phosphatase | 0.591 | **0.306** | inverted |
| ALT | 0.654 | **0.218** | inverted |

The model learned from NHANES that raised ALT means liver disease, which is
correct there. In the German cohort raised ALT means the patient is healthy,
because NHANES captures mild self-reported liver disease where ALT rises, while
the German cohort is blood donors against biopsy-confirmed hepatitis, fibrosis
and cirrhosis, and in advanced cirrhosis ALT falls as hepatocyte mass is lost.
German median ALT is 23.1 in donors, 15.2 in hepatitis and 5.6 in cirrhosis.

Dropping ALT and alkaline phosphatase lifts the transfer to 0.764. That number
is **not** reported as validation anywhere, because those two features were
chosen by looking at the test result, which is fitting the test set. The 0.442
stands, now with a mechanism attached, and it bounds the claim: this panel
describes mild liver disease in a US population and should not be expected to
rank advanced cirrhosis. Reproduce with `python experiments/liver_germany.py`.

**4. Ovarian and prostate still have no external cohort.**
Searched again across every host this environment can actually download from:
Mendeley, figshare, Harvard Dataverse, OpenML, and DataCite across all
repositories. Nothing exists pairing an ovarian mass or a prostate biopsy with
the required markers. This stays an open limitation, now with the search on
record rather than as an assumption.

**5. Lung's target is a lifetime diagnosis.** Unchanged and unfixable from
NHANES: age at diagnosis exists only in five of the ten cycles, and restricting
to it leaves 34 to 54 events against a floor of roughly 96. Pooling all ten
cycles is what made the panel possible at all, and the survivor bias is the
price. Stated on the panel.

**6. Breast had no subgroup measurement. Now it has one, and it found something.**
The Wisconsin cohort records no age, sex or race, so the demographic breakdown
every other panel carries is impossible. But lesion size is available, and it is
the axis that matters clinically, because small lesions are the ones where an
earlier answer changes anything.

| Subgroup | n | Malignant | AUC |
|---|---|---|---|
| Smallest third by nuclear area | 190 | 6 | **0.974** |
| Middle third | 189 | 39 | 0.976 |
| Largest third | 190 | 167 | 0.991 |

This weakness has since been fixed rather than only disclosed. The panel used to
score 0.954 overall and 0.739 on the smallest lesions, a spread of 0.216, and it
was weakest exactly where an earlier answer would change something.

The cause was that it read four of the thirty Wisconsin measurements. That came
from this project's rule that a model may only use what the application can
collect, and at the time the form asked for four numbers. But breast is an
INTERPRETATION panel: whoever runs it is holding a pathology report that already
carries all thirty. Restricting to four was not a constraint of the setting, it
was a stale assumption about the form.

Measured on the small-lesion subgroup alone, four features score 0.680 and all
thirty score 0.952. Shipped, the subgroup now reads 0.974 and the spread is
0.019. Race and age remain unmeasurable from this cohort and that limitation
stands unchanged. Reproduce with `python experiments/breast_small_lesions.py`
and `python experiments/breast_subgroups.py`.

**7. No prospective validation and no IRB.** Not fixable by writing code. An
ethics approval is granted by an institution to a named investigator for a
specific protocol, and no amount of analysis substitutes for it. The executable
protocol and the analysis harness exist in `PROTOCOL.md` and
`prospective_analysis.py` so that a real study could be run, but until one is,
this remains a research prototype and the interface says so.

### One split is not an estimate, and cervical proved it

Every panel here published an AUC from a single 80/20 split with seed 42. That
is the standard protocol and it hides a real problem: a split is one draw from
a distribution, and on a small cohort that distribution is wide.

So the whole protocol was repeated with different seeds, refitting from
scratch each time, and the shipped split was located inside its own
distribution. Reproduce with `python experiments/split_stability.py`.

<!-- AUTOGEN:stability -->
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
<!-- /AUTOGEN:stability -->

All eight live panels sit between the 20th and 80th percentile of their own
split distribution, which is what a representative number looks like. Cervical sat at
the 97th. Its published 0.725 described one favourable shuffle, not the model.

### Cervical is withdrawn

At a mean of 0.594 with a spread running 0.421 to 0.789, the interval includes
chance. That is the same standard prostate was withdrawn under, and applying it
to one panel and not the other would make the standard meaningless.

The cause is arithmetic rather than bad luck: 55 positive biopsies against the
roughly 96 events needed, so a single test split carries about 11 cases and
cannot estimate anything stably.

The external cohort that was built for it survives the withdrawal. 11,100 women
from NHANES 2005 to 2014 carrying the same risk history, with 184 cervical
cancers, in `fetch_nhanes_cervical.py`. If the panel is ever rebuilt on a cohort
with enough events, the transfer test is already there.

### Colorectal had the same conflict, and won it

Colorectal showed the mirror image and had to be arbitrated the same way rather
than by picking the flattering number.

    cross-validation, all 96 events    model 0.804, age and sex 0.774   +0.030
    single held-out split, ~19 events  model 0.793, age and sex 0.817   -0.024

Nineteen events cannot settle that. Twenty paired repeats on identical folds
can: the model won **20 out of 20**, mean difference **+0.038**, with a 95 percent
range of +0.011 to +0.066 that excludes zero. Reproduce with
`python experiments/colorectal_vs_age.py`.

So colorectal ships and cervical does not, and both decisions come from the same
test rather than from which answer was more convenient.

### Lung: the confound had to be removed before the panel meant anything

Lung was reported as unbuildable in an earlier pass, on 57 events. Pooling all
ten NHANES cycles from 1999 to 2018 raised that to 117, and adding the lab
values that pass had missed changed the picture: serum cotinine, the nicotine
metabolite, is measured in every cycle and is an objective replacement for the
smoking question.

The first version still flattered itself. On the full 45,396-adult cohort the
questionnaire alone reaches 0.836, because almost every case smoked and most
controls did not, so "do you smoke" separates the groups nearly on its own.

Restricting controls to people with tobacco exposure, by self-report or serum
cotinine at or above 3 ng/mL, asks the real question and is also the population
that actually gets offered lung screening:

| Feature set | AUC | vs questionnaire | Wins |
|---|---|---|---|
| Questionnaire alone | 0.792 | | |
| Plus self-reported pack-years | 0.793 | +0.001 | 5/10 |
| Plus **serum cotinine** | 0.808 | **+0.016** | 10/10 |
| Plus the whole lab report | 0.821 | **+0.029** | 10/10 |

The middle two rows are the point. A measured lab value beat the self-reported
dose it replaces, which is the claim this whole application makes, tested and
upheld. Reproduce with `python experiments/lung_vs_smokers.py`.

Spirometry adds far more, 0.807 to 0.914, but only 13 lung cases have FEV1 and
FVC recorded, so it is nowhere near the event floor and is left out.

The limitation travels with the panel: pooling ten cycles means giving up age at
diagnosis, so the target is a lifetime diagnosis, and lung cancer survival is
poor enough that the cases who live to be interviewed are survivor-biased.

### Prostate: reinstated on a cohort that actually has PSA

The old prostate panel was withdrawn on 97 Stanford records with an interval
sitting on chance. NHANES then confirmed the negative from the other direction:
with 738 lifetime prostate cases and no PSA available, bloodwork added -0.000
over age, because NHANES excludes men with a prostate cancer history from the
PSA subsample by design.

The replacement is 212 men with suspected prostate cancer who all went to
transperineal biopsy, 121 with adenocarcinoma, controls being men whose biopsy
came back benign. Measured against PSA alone, the only baseline that counts
because every one of these men already had a PSA drawn:

| Feature set | AUC | vs PSA alone | Wins |
|---|---|---|---|
| Age alone | 0.568 | | |
| PSA alone | 0.670 | | |
| Age, PSA, volume, PSA density, BMI | 0.668 | -0.003 | 10/20 |
| **Plus PI-RADS from MRI** | **0.826** | **+0.156** | 20/20 |

Blood and ultrasound alone do not beat reading the PSA number. They tie it. What
carries this panel is the radiologist's PI-RADS score, exactly as nuclear
morphology carries the breast panel, so it ships as an **interpretation** panel
that asks for an MRI score and says so on its face rather than pretending a lab
report is enough.

The dataset came from Zenodo, which blocks this environment on every route, so
it was recovered through another path and then verified rather than trusted:
PSAD is a derived column equal to PSA divided by prostate volume, and that
identity holds across all 212 rows to within 0.005, which jointly verifies three
columns row by row.

### Lung, colorectal and prostate were all requested. One survived measurement.

All three were tested identically against NHANES, sweeping the diagnosis window
from four years out to lifetime, because a lifetime target counts someone cured
thirty years ago as positive and mostly predicts age. Reproduce with
`python experiments/site_window_sweep.py`.

| Site | Events at best window | Age and sex baseline | With bloodwork | Gain | Outcome |
|---|---|---|---|---|---|
| Colorectal | 96 at 8 years | 0.774 | **0.804** | **+0.030** | ships |
| Prostate | 373 lifetime | 0.874 | 0.876 | +0.002 | NHANES route dead, see below |
| Lung | 57 lifetime | n/a | n/a | n/a | superseded, see below |

**Colorectal ships.** This is the ColonFlag idea rebuilt from open data: age,
sex and a complete blood count. ColonFlag is the best known colorectal
early-detection model, uses exactly that feature set, reports AUCs in the low
0.80s, and is proprietary with no public data behind it. The eight year window
is the tightest one clearing the roughly 96 event floor, which makes this the
smallest shipped panel by event count and gives it the widest interval of any
NHANES panel.

**Prostate could not be built from NHANES, and that part is structural.**
Bloodwork adds between 0.000 and 0.002 over age alone at every single window
tested, from 153 events at four years to 373 at lifetime. It is an age model
wearing a lab coat. Prostate risk lives in PSA, and NHANES excludes men with a
prostate cancer history from the PSA subsample by design, which leaves 17 cases
paired with a PSA value. The older Stanford cohort has 97 records and a
confidence interval whose lower bound sits on chance. Shipping any of these
would mean printing a cancer risk score driven entirely by the patient's
birthday. A cohort that does carry PSA was found later and the panel now ships
on that instead; see the prostate section above.

**Lung failed here for lack of events, and was rescued later.** 34 cases at four years, rising only
to 57 at lifetime, against a floor of roughly 96. Widening the window does not
rescue it because the ceiling is how many people NHANES sampled who had lung
cancer and survived to be interviewed, which is itself a survival bias. No
public tabular cohort pairs lung cancer with a lab panel either: the open lung
datasets are CT imaging, and the large screening trials that would answer this,
NLST and PLCO, require an application and a data use agreement rather than a
download. Pooling all ten NHANES cycles and adding serum cotinine eventually
got there; see the lung section above.

### Two panels are triage, not population screening

The ovarian and cervical panels run *after* something has already been found. Ovarian separates
a malignant ovarian mass from a benign one in women going to surgery; cervical prioritises women
already being assessed for colposcopy. Their precision is therefore projected onto prevalence in
the referred group, not onto SEER population incidence.

That distinction is not cosmetic. Projected onto population incidence the cervical panel would
flag roughly **7,383 women per true case**, which would be indefensible to ship. Projected onto
the 6.4 percent prevalence among referred women it flags 9.4 per case at 96.2 percent NPV.
Same model, same numbers, completely different meaning, so the prior is stated on the panel.

### The ovarian panel is the one where the bloodwork does the work

Cases in that cohort are seventeen years older than controls, 53 against 36, which is exactly the
trap the general panel fell into. So age was made a baseline and had to be beaten:

| Feature set | Inputs | AUC |
|---|---|---|
| Age and menopausal status | 2 | 0.792 |
| Routine bloodwork only, no age, no tumour markers | 20 | **0.938** |
| Tumour markers only | 5 | 0.931 |
| Everything | 27 | **0.941** |

Routine chemistry and a blood count, with no age and no tumour markers, reach 0.938. This is the
one panel where the lab report itself carries the signal, and it is worth contrasting with the
general panel below, where it does not.

### What the cervical panel looked like before it was withdrawn

This section is kept as a record of a mistake, not as a description of something that ships. The
cervical panel is **withdrawn**; the reasoning is in "Cervical is withdrawn" above.

At the time it shipped, the argument for it went like this. Cross-validated AUC was 0.587 while
the held-out split gave 0.725. With 55 positive biopsies the interval ran 0.547 to 0.881, which
excludes chance, and it beat age alone by a wide margin. The gap between 0.587 and 0.725 was noted
and read as ordinary optimism in a held-out number, so the panel shipped labelled as the weakest
evidence in the project.

That reading was wrong. The gap was not optimism to be discounted, it was the signal that the
held-out number was a lucky draw, and running the split thirty times showed the shipped figure
sitting at the 97th percentile of its own distribution. The lesson is that a large gap between
cross-validation and a held-out split is not a number to average over. It is a reason to resample
before believing either.

Cutting its 15 input fields had also been tried and failed: eight fields scored 0.665 with an
interval containing chance, and six scored 0.552. In hindsight that too was evidence the panel had
nothing stable to find.

### The general panel now answers a screening question

It used to predict "have you ever been told you had cancer". Someone cured thirty years ago counted
as positive, so the model was largely predicting age: 0.781 against 0.777 for age and sex alone, a
gain of 0.004.

NHANES 2005 to 2014 records **age at diagnosis**, so the cohort can be cut properly. Positives are
people diagnosed **within four years of the blood draw**, and long-ago survivors are excluded rather
than relabelled. That is a screening question, and it fixed the target.

It did not make the panel strong. On a single held-out cycle the gain over age and sex rose from
0.004 to 0.019, and that 0.019 was quoted here for some time. Measured properly, on repeated
paired folds rather than one split, the gain is **0.006**. The single split was flattering by a
factor of three on the panel with the least to spare, which is the same error the cervical panel
was withdrawn for.

### Routine bloodwork does not detect general cancer

Measured, not assumed, on a held-out NHANES cycle against the recent-diagnosis target:

| Feature set | Inputs | AUC |
|---|---|---|
| Age and sex only | 2 | 0.729 |
| Age, sex, lifestyle (shipped) | 5 | **0.748** |
| All 14 routine blood values | 14 | **0.663** |
| Everything combined | 19 | 0.737 |

**Fourteen blood values score worse than knowing someone's age.** A complete blood count and
metabolic panel are not cancer detection tests. That is exactly why the specific panels here use
disease-specific markers, and why commercial multi-cancer blood tests use cell-free DNA rather than
routine chemistry. Reproduce with `python experiments/screening_vs_age.py`.

The general panel therefore reads risk factors, not the lab report, and that is a measured decision
rather than an oversight.

### Choosing the operating point

A fixed 0.5 threshold drove measured sensitivity to **0.008** once prevalence was realistic. That is
not a broken model: a calibrated model on a 3 percent condition is correctly reporting that almost
nobody is more likely than not to have it.

Each panel now selects its threshold by Youden's J on out-of-fold predictions inside the training
data and freezes it in the bundle, so the interface, the evaluation and the prospective analysis all
use one number.

### External validation

| Panel | Test | Internal | External |
|---|---|---|---|
| General | NHANES 2015-2018, unseen cycles | 0.789 | **0.804** |
| Liver | NHANES 2015-2018, unseen cycles | 0.734 | **0.716** |
| Liver | India, 583 patients | 0.734 | 0.640 |
| Liver | Germany, 589 patients | 0.734 | 0.442 |
| Pancreatic | Leave-one-tissue-bank-out, mean | 0.969 | **0.962** |
| Breast | WPBC, 198 patients, sensitivity only | 0.915 | **0.894** |

Germany remains the hardest transfer, which is honest and worth keeping visible.

### Pancreatic: withdrawn, then reinstated on evidence

This panel was withdrawn for having no external test. That was wrong. Its cohort is drawn from
**three independent tissue banks** and the site column had been discarded as an identifier during
preprocessing.

| Held out site | Train n | Test n | Cases | AUC | 95% CI |
|---|---|---|---|---|---|
| BPTB | 235 | 365 | 74 | 0.978 | 0.962 to 0.990 |
| CPTB | 459 | 141 | 34 | 0.959 | 0.911 to 0.991 |
| UPTB | 506 | 94 | 22 | 0.950 | 0.895 to 0.990 |

Mean 0.962, every interval excludes chance, drop from the internal split 0.007. It transfers
between institutions, so it was reinstated. It has still never met a screening population, where
it would flag roughly 843 people per true case, and the interface says so on the panel.

### Prostate: withdrawn, and the search is documented

Test AUC 0.786, 95% CI **0.505 to 0.99**. The lower bound sits on chance.

External validation was searched for and does not exist. The Stanford cohort has no site column.
NHANES measured serum PSA on 4,697 men across 2005 to 2010, but holds only **17 prostate cancer
cases**, because men already diagnosed are excluded from the PSA subsample by design. That is a
structural exclusion, not a gap that more searching would close.

### A negative result, kept

The breast panel needs a biopsy, which contradicted this project's own rule that a model may only
use features the app can collect. The obvious fix was to rebuild it on blood markers using Breast
Cancer Coimbra, then validate on NHANES women.

It was tried and it failed. Internal 0.804 on Coimbra, external **0.495** on NHANES with a 95% CI of
0.377 to 0.607, which contains chance. Logistic did no better at 0.518.

So the panel was not built, and the rule was amended instead to acknowledge two kinds of panel:
**screening** panels that read routine bloodwork, and one **interpretation** panel that reads a
diagnostic test already performed. The experiment is committed at
`experiments/blood_breast_panel.py` so the negative result is reproducible rather than merely
asserted. Running it before building the feature is what stopped a non-working panel from shipping.

### The one design that can actually test the premise

Every cohort above records the blood and the answer at the same visit, or assembles cases after
the fact. That design cannot separate "the bloodwork predicts the cancer" from "the cancer has
already changed the bloodwork", and no modelling choice repairs it.

NCHS links every NHANES participant to the National Death Index and publishes the linkage, so the
sample is drawn, years pass, and the death certificate arrives later from a different agency.

    33,834 adults, NHANES 1999-2014, no cancer diagnosis at baseline
    339 deaths from malignant neoplasm within 60 months of the blood draw
    1.00% event rate, stable across all eight cycles

| Feature set | Features | AUC | Gain over age and sex | Wins |
|---|---|---|---|---|
| Age and sex | 2 | 0.816 | — | |
| + BMI, smoking, alcohol | 5 | 0.827 | +0.011 | 5/5 |
| **+ complete blood count** | 15 | **0.832** | **+0.016** | 5/5 |
| + metabolic and liver panel | 16 | 0.825 | +0.010 | 5/5 |
| Everything | 26 | 0.829 | +0.013 | 5/5 |

Leave-one-cycle-out, training on seven NHANES cycles and testing on the eighth, gives a mean of
0.837, so the model transfers between survey waves rather than fitting one of them.

**The gain is real and it is too small to act on.** Every arm beats age and sex on 5 of 5 paired
repeats, and the largest gain is 0.016. Set against the liver panel's +0.106 and the pancreatic
panel's +0.498, this is the clearest statement in the project of where routine bloodwork carries
signal and where it does not: it carries a great deal about organ-specific disease when the
organ's chemistry is on the panel, and very little about undifferentiated cancer risk.

The outcome here is death from cancer, not detection of it. People who developed cancer and
survived count as non-cases, because they did, and the endpoint is confounded by everything that
determines whether a cancer is survivable. It is a different question, honestly labelled, on a
design the other cohorts cannot offer. Reproduce with
`python experiments/prospective_mortality.py`.

### The gain did not survive a different decade, and that changed the conclusion

The prospective panel was then tested properly. NHANES III ran from 1988 to 1994 and is linked to
the National Death Index by the same agency using the same method, so it gives a real external
cohort: different decade, different analysers, different field staff, higher smoking prevalence.

Identical features on both sides. The test cohort is imputed with the TRAINING medians, never its
own, so its distribution cannot leak into the fit.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| Age and sex only | 2 | **0.852** | 0.831 to 0.873 |
| Full blood work | 22 | 0.839 | 0.819 to 0.858 |

**Transferred gain: -0.013. The internal gain was +0.013.** Adding twenty blood values makes the
prediction *worse* on people measured in a different decade than using age and sex alone. Whatever
those values contributed inside NHANES 1999-2014 belonged to that survey rather than to human
physiology.

The uncomfortable part is what this says about the leave-one-cycle-out result directly above it.
Holding out a whole NHANES cycle and training on the other seven gave a mean of 0.837 with no fold
below 0.808, which reads like solid evidence of generalisation. It was not. Cycles of one survey
share a protocol, a laboratory contract, instrument calibration and an analytic pipeline;
resampling across them varies the participants while holding the measurement process fixed.

**Every internal validation strategy in this project, including the repeated paired
cross-validation that is its arbiter everywhere else, would have reported this panel as working.**
Only a genuinely external cohort caught it. The panels here that have no external cohort should be
read with that in mind, and that caution is now the honest headline of this section rather than a
footnote. Reproduce with `python experiments/prospective_external.py`.

### The bowel panel was tested the same way, and it passed

The section above should not be read as "external validation always fails". It is a test, not a
penalty, and the same test applied to an organ-specific panel gave the opposite answer.

NHANES III recorded both the site of any reported cancer and the age at which the person was first
told, which reconstructs exactly the eight-year window the bowel panel's training cohort uses.

    train   NHANES 2005-2014   23,794 adults, 96 colorectal cancers (0.40%)
    test    NHANES III         14,499 adults, 56 colorectal cancers (0.39%)

The two prevalences agree to within a hundredth of a percent, which is a check that the window was
reconstructed the same way on both sides rather than a coincidence.

| Feature set | Features | External AUC | 95% CI |
|---|---|---|---|
| Age and sex only | 2 | 0.761 | 0.699 to 0.814 |
| **Full panel** | 16 | **0.790** | 0.739 to 0.838 |

**Transferred gain +0.029, against +0.039 measured inside the training survey.** Roughly three
quarters of the effect is still there on people measured fifteen years earlier, on different
analysers, by different field staff.

Put beside the section above, this is the sharpest form of the whole project's result. Under an
identical procedure, the undifferentiated panel's gain reversed sign and the organ-specific panel's
gain largely survived. The two questions do not merely differ in effect size. They differ in
whether the effect is real at all. Reproduce with
`python experiments/colorectal_external.py`.

### A good mechanism is not evidence: the bowel panel and iron deficiency

The classic way a right-sided colon cancer announces itself is iron-deficiency anaemia from slow
occult bleeding. The bowel panel read haemoglobin and red cell count and nothing else from the
blood count, so it could tell that someone was anaemic but not what kind — and "anaemic" on its own
is nearly useless, because most anaemia is not cancer. What separates iron deficiency is the shape
of the red cells: a low haemoglobin with a **low MCV** and a **high RDW**, where anaemia of chronic
disease leaves both normal and B12 deficiency pushes MCV up.

Those indices are printed on the same blood count, on the same sample, at no extra cost. It looked
like an obvious omission with a textbook mechanism behind it.

| Arm | Features | AUC |
|---|---|---|
| Shipped today | 16 | 0.810 |
| + MCV and RDW | 18 | 0.812 |
| + all six CBC indices | 22 | 0.807 |

MCV and RDW gain 0.003 and win 6 of 10 paired repeats. All six indices together *lose* 0.003 and
win 3 of 10. The bar was set at 0.005 with 8 of 10 wins before the experiment ran, and neither arm
clears it. **Nothing was added to the panel.**

One plausible reason, offered as a hypothesis rather than a conclusion: this cohort's target is a
diagnosis within eight years of the blood draw, and a tumour bleeding enough to shift the red cell
indices is usually not eight years away from being found. The signature may be real and simply
absent at the moment these samples were taken.

The reason this is written down at all is that the mechanism was persuasive, the omission was real,
and the fix still did not work. A project that only records the experiments that succeeded cannot
be checked. Reproduce with `python experiments/colorectal_iron.py`.

### The app asked for exercise and nothing read the answer

`tools/audit.py` checks that every question the form asks is consumed by some
live panel. It found one that was not: hours of exercise per week. The form
asked, and no model had read it since the general and liver panels moved to
NHANES.

There were two honest responses: wire it up, or stop asking. Which one depended
on whether it carried information, so it was measured rather than guessed.
NHANES has recorded physical activity under the Global Physical Activity
Questionnaire since 2007, so recreational exercise hours were pulled and offered
to the general panel. Work activity was deliberately excluded, because a
warehouse shift and a run are not the same exposure and the form's question says
"exercise".

It did not help. The general panel's gain over age and sex went from +0.005 to
+0.003 on the cycles that actually asked, and the same on the full pooled cohort:
a small loss, consistently, not a small win. Physical inactivity is a real cancer
risk factor at the population level, but that is not the same claim as adding
information once you already know a person's age, sex, BMI, smoking and drinking.
It is largely spoken for by those.

So the question was removed from the form. Waist circumference was tested in the
same run, on the reasoning that BMI cannot separate a heavy-set person from a
centrally obese one. It came out at +0.003, in the right direction and consistent
across both cohorts, but under the 0.005 bar set before the experiment ran. It
was not adopted, and the bar was not moved afterwards to let it in. The column is
still collected, because it is free and a different cohort or target may make
better use of it.

Both results are in `experiments/general_body_activity.py`.

### The liver panel was reading ALP without the value that disambiguates it

Alkaline phosphatase rises in liver disease. It also rises in bone disease. The value that separates
those two is GGT, and it is the first thing a hepatologist asks for after a raised ALP.

The liver panel did not have it. Not because it was rejected, and not because a patient would have
to pay for another test: GGT sits on the same comprehensive metabolic panel NHANES already draws,
on the same blood sample, printed on the same page. It had simply never been pulled.

Three other analytes on that same panel were in the same position, so all four were tested together
against what shipped:

| Arm | Features | AUC |
|---|---|---|
| Shipped | 11 | 0.741 |
| **+ GGT** | 12 | **0.750** |
| + GGT, globulin | 13 | 0.749 |
| + GGT, globulin, LDH, uric acid | 15 | 0.749 |

GGT is worth +0.009, winning 5 of 5 paired repeats on identical folds, and the two ranges do not
overlap: 0.738 to 0.744 without it, 0.747 to 0.753 with it. Globulin, LDH and uric acid add nothing
once GGT is in, and two of the three make it very slightly worse, so only GGT was taken. Adding all
four because all four were free would have been the easy call and the wrong one.

The German external cohort cannot check this, because it does not record any of the four. That is
stated rather than worked around. Reproduce with `python experiments/liver_extra_analytes.py`.

### PDF parser accuracy

The parser sits upstream of every model and was previously unmeasured. `test_parser.py` renders lab
reports in five deliberately different layouts from known ground truth and compares field by field.

| | Before | After |
|---|---|---|
| Overall field accuracy | 68.9% | **100%** (90/90) |
| Range-printed-before-result layout | 5.6% | 100% |

Four real bugs: `ast` matched inside the word "fasting", "Platelet Count" never matched a
plural-only pattern, "Alpha-Fetoprotein" never matched the unhyphenated pattern, and reference
ranges were read as results. Root cause was flattening the document into one string; reports are
line-oriented, so the parser now works line by line and blanks reference intervals before looking
for a value.

### The prospective study is executable, not just described

`PROTOCOL.md` requires the analysis script to be committed before enrolment opens.
`prospective_analysis.py` is that script, written before any participant exists so the endpoint
cannot be chosen after seeing the data.

- `--dictionary` emits the data dictionary and a blank capture template, so the instrument and the
  analysis cannot drift apart
- `--simulate` runs the whole pipeline on synthetic data in the study schema, which already caught
  one real bug before any patient data exists
- Primary endpoint is PPV at the frozen shipped threshold. Retraining or recalibrating on
  prospective data is explicitly forbidden.

## Known limitations

- **Four of the eight cohorts are case-control, not screening series.** Breast, pancreatic,
  ovarian and prostate assemble cases and match controls to them, which is why their AUCs are the
  highest here and the least transferable. The NHANES cohorts behind general, liver, bowel and lung
  are population-based instead. See the precision table above.
- **The breast panel is an interpretation panel, not a screening one.** Its inputs need a biopsy
  that already happened. Rebuilding it on blood was tried and failed at chance, so the rule was
  amended to name the two classes rather than pretend there is one.
- **The liver panel detects liver disease, not liver cancer.** Liver disease is roughly 300 times
  more common and is the dominant precursor to hepatocellular carcinoma, so this is a useful thing
  to detect, but it is not a cancer claim.
- **One panel is withdrawn: cervical.** Repeated-split testing put its shipped number at the
  97th percentile of its own distribution, with a mean of 0.594 and a spread from 0.421 to 0.789.
  It was a lucky split, not a working model. Prostate and pancreatic were each withdrawn at one
  point and reinstated only after a cohort or a validation design was found that supported them.
  Both decisions and the evidence behind them are above and on the methodology page.
- **No prospective test and no IRB.** No real patient report has been followed to an outcome. A
  submittable study protocol with pre-specified statistics is drafted in [PROTOCOL.md](PROTOCOL.md).
  Its sample-size table shows the pancreatic panel would need roughly 690,000 participants to
  validate at real incidence, which is why that panel carries the strongest caution here.
- **The ensemble is within noise of logistic regression** on breast and pancreatic.
- **Race and ethnicity exist only in the NHANES cohorts.** Subgroup AUC is measured for the general
  panel across six groups and runs 0.555 to 0.667. For breast and pancreatic it remains unmeasured.

---

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 App Router, React 19, TypeScript, Tailwind v4, Recharts, Vercel |
| Backend | FastAPI, Uvicorn, Pydantic, pdfplumber, pandas, Render |
| Models | XGBoost + Extra Trees ensembles and logistic regression, whichever wins on CV AUC. Isotonic calibrated. SHAP for attribution |

### Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service identity and loaded models. Wakes a sleeping instance. |
| `GET` | `/health` | Liveness and model count. |
| `GET` | `/models` | Registry with held-out metrics and cohort caveats. |
| `POST` | `/predict` | Flat object of lab values and history, every field optional. |
| `POST` | `/parse-pdf` | Up to five PDFs, returns extracted biomarker values. |

### Design decisions worth knowing

**Schema alignment.** A model may only train on features the application can actually collect.
Columns the interface never asks for are dropped before fitting, because a model trained on inputs
it will never receive reports an accuracy that does not describe how it performs in use.

**Median imputation, not zero.** Each bundle stores the training median of every feature. A zero BMI
is not a missing BMI, it is an impossible patient, and tree models will split on it.

**Clinical thresholds never overwrite the model.** PSA at 4.0, CA 19-9 at 37, AFP at 10 and the rest
are evaluated separately and surfaced as flags. The number shown is the model's own output.

**Per-patient SHAP.** Each ensemble member is explained separately and normalised before averaging,
because XGBoost reports in log odds and Extra Trees in probability. Features the patient left blank
are excluded, since a contribution from an imputed median describes the training set, not the person.

---

## Running it

```bash
# install
pip install -r requirements.txt

# download the real external cohorts first
python fetch_external.py

# structural checks: duplicate keys, panels training on features the form never
# asks for, model features with no range check, panels with no declared type.
# Exits non-zero, so it can gate a commit.
python tools/audit.py

# evaluate, this produces evaluation.json which training embeds
python evaluate.py

# cross-country external validation and leave-one-cohort-out
python external_validation.py

# PDF parser accuracy, needs the API running
python test_parser.py

# train, calibrate, and write model bundles
python train_models.py

# regenerate the sample case pool
python make_cases.py

# serve the API
cd backend && pip install -r requirements.txt
uvicorn api:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL` and falls back to the deployed backend.

### Repository

```
oncovision/
├── backend/
│   ├── api.py              FastAPI service, PDF parsing, scoring, gating
│   ├── models/             calibrated ensemble bundles, one per live panel
│   └── model_metrics.json
├── data/                   source cohorts
├── frontend/src/app/
│   ├── page.tsx            assessment, guide, methodology, developer
│   ├── fields.ts           input schema and glossary
│   └── cases.ts            generated sample case pool
├── tools/
│   └── audit.py            structural checks, exits non-zero so it can gate a commit
├── experiments/            every question that was measured, including the ones
│                           that came back negative and changed nothing
├── tests/lab_reports/      5 synthetic report layouts for the parser
├── fetch_external.py       downloads the real cohorts, harmonises units
├── fetch_nhanes_*.py       builds the NHANES cohorts, one per panel
├── evaluate.py             held-out evaluation, CIs, calibration, PPV, baselines
├── external_validation.py  trains on one country, tests on another
├── train_models.py         training and calibration pipeline, panel config
├── make_cases.py           sample case generator
├── test_parser.py          renders the 5 layouts, measures parser accuracy
├── prospective_analysis.py pre-committed study analysis, data dictionary, simulator
├── EVALUATION.md           full evaluation report, regenerated by evaluate.py
├── PROTOCOL.md             the prospective study protocol
└── PROJECT.md              methodology write-up
```

---

## Privacy

Uploaded PDFs are read into memory, parsed, and discarded inside the request. Nothing is written to
disk and no database is attached. Values live in browser state for the session only.

The service also does not accept what it cannot use. When the cervical panel was withdrawn, its
questions were taken out of the form, but the API went on accepting them: number of sexual
partners, age at first intercourse, pregnancies, contraceptive and IUD history, and STD and HPV
status. Nothing scored any of it. That is worse than dead code, because an endpoint that accepts a
sexual history is one that can receive and log one, and there was no reason for it to exist.

`tools/audit.py` now fails if the API accepts any field that no live panel reads and no form sends,
and separately if a withdrawn panel is still named as the consumer of an input. Seventeen fields
were removed when that check was first run.

---

## License

MIT, see [LICENSE](LICENSE). The additional notice there is not decorative: this is not a medical
device and must not be used to make or defer a medical decision.

**Palash Rakshit** · [palash.raks@gmail.com](mailto:palash.raks@gmail.com) · [LinkedIn](https://www.linkedin.com/in/Palash-Rakshit10)
