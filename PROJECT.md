# Oncovision

A full stack platform that reads standard patient lab reports and returns multi-cancer risk
assessments in real time.

Built as a formal mentored research project under the direct guidance of a clinical oncologist at
UCI CHOC, ensuring the clinical data and testing parameters met professional medical standards.

---

## 1. The goal

Make multi-cancer screening cheap enough to be routine, and make the results readable to someone
without a medical background.

Most patients get lab work every year and never learn what is in it beyond a flag or two. A
comprehensive metabolic panel and a complete blood count together produce roughly thirty numbers.
A physician scans them for values outside the reference range and moves on, which is the correct
use of a nine minute appointment but leaves most of the information on the table. The signal that
matters in early detection is frequently not one number being wrong. It is a particular
combination of numbers being slightly unusual together, which is exactly the kind of pattern a
person reading a printout is poorly equipped to notice and a trained model is well equipped to
find.

Oncovision closes that gap by reading two things together.

The first is **your lab reports**, and not only a blood test. It covers whatever panels have been
run: the complete blood count, metabolic and liver chemistry, tumour markers, and measurements
taken from biopsy imaging.

The second is **information about you**, meaning what a lab report does not contain. Age, sex,
smoking, alcohol, exercise, hepatitis B and C, and diabetes. Several of the models draw more from
this half than from the chemistry, so neither side is sufficient alone.

Both are scored against patterns learned from anonymised patient records, and the system returns a
per-cancer probability with a plain English explanation of what drove it.

### The problem this addresses

Cancer screening in the United States has two gaps. It covers very few cancers, and the tests that
do exist cost enough that plenty of people never take them.

**Almost nothing is screened for.** Only four cancer types have a screening test recommended by the
US Preventive Services Task Force: breast, cervical, colorectal and lung. Those four account for 29
percent of cancer cases. 57 percent of diagnosed cancers have no recommended screening test at all,
and that group causes 70 percent of cancer deaths.

**Screening catches a small share of what it covers.** Only 14 percent of cancers in the United
States are diagnosed after the patient had a recommended screening test. A 2025 analysis estimated
that current screening, as it is actually used, leaves as much as 87 percent of cancer deaths
unaddressed.

**Timing decides the outcome.** Across all cancers combined, five year relative survival is roughly
92 percent when the cancer is still localised and roughly 15 percent once it has spread to distant
sites. The gap between those two numbers is largely a question of when it was found.

**Cost keeps people out.** Multi-cancer blood tests already exist. Galleri, the best known, lists at
$949 and is not covered by Medicare or most insurance. A screening colonoscopy averages about
$2,750 without insurance and still runs several hundred dollars out of pocket for many who have it.

Oncovision adds no new test and no new cost. It works from a lab report already paid for, plus
information the patient can supply in a minute, and checks that combination against four cancer
models at once rather than the one or two a given age and sex qualifies for. The marginal cost of
running it is nothing, which is the point. A test that costs $949 will not become routine. A test
that reads a PDF already sitting in a patient portal can.

What it cannot do is stand in for the screening above. A colonoscopy looks at a colon and a
mammogram looks at breast tissue. Oncovision looks at numbers, and numbers can be normal in someone
who has cancer. It is built to raise a question early enough to be worth asking, not to answer one.

Sources: NORC at the University of Chicago, analysis of cancers detected by screening. Ofman et
al., *Cancer Biomarkers*, 2025, on cancer deaths not addressed by current screening. SEER five year
relative survival by stage at diagnosis. Published list price for Galleri and reported average
colonoscopy cost, both as of 2025.

It is a screening support tool. It does not diagnose, and it is not a substitute for a physician.

---

## 2. What the user does

Three inputs, all optional, all feeding the same models.

**Upload a lab report.** The user drops in up to five PDFs straight from a patient portal. An
automated parsing system pulls the specific medical variables out of them and fills the form.

**Correct or type the values.** Twenty three lab values across six panels: body metrics, complete
blood count, metabolic panel, liver panel, tumour markers, and breast mass morphology. Anything the
parser misread can be typed over, and anything the patient does not have is left blank.

**Answer the history questions.** Seven items that are not on a lab report and cannot be parsed
from one: sex, smoking, alcohol, exercise, hepatitis B, hepatitis C and diabetes. Fields no shipped
panel reads were removed rather than collected for nothing.

Pressing **Generate case** loads a real record drawn at random from the training data so the system
can be tried without entering anything, and states up front which result to expect.

The output is five cards sorted highest first: four cancer panels and a healthy baseline. Each card
expands to show which inputs the model relied on, how each value compares to its clinical reference
limit, which established clinical thresholds were crossed, and how accurate that particular model
is.

---

## 3. How the system works

### Stage 1: Ingestion

PDFs are read with `pdfplumber`, which extracts the text layer. Lab reports are laid out very
differently between Quest, Labcorp, and hospital systems, and the same analyte is printed under
several names, so a regex synonym map locates each of the twenty three biomarkers under its
variants. WBC appears as `WBC` or `white blood`. Alpha fetoprotein appears as `AFP` or
`alpha fetoprotein`. CA 19-9 appears as `CA 19-9`, `CA19-9`, or `CA19_9`. Unit strings are stripped
before matching so that `mg/dL` and `U/L` do not interfere with number capture.

The approach is text based, not image based, so a scanned report will not parse. The interface says
so directly and tells the user to check the parsed values against the source before running, since
a report with an unusual layout can cause a reference range to be captured instead of a result.

### Stage 2: Sanitisation

Every value is checked against a table of physiologically possible ranges. A BMI of 400 or a
creatinine of 90 is not a patient, it is a typo or a misparsed line. Values outside the bounds are
dropped rather than clamped, and the list of what was dropped and why is returned to the interface
and shown to the user. Clamping would silently change a score. Dropping and reporting will not.

### Stage 3: Schema alignment

This is the design decision the accuracy of the whole system rests on.

The source datasets were collected independently and share no common schema. The Wisconsin
breast set has thirty morphology columns. The pancreatic cohort has urinary protein assays. The
prostate cohort has surgical findings. If each model is trained on everything its own dataset
happens to contain, then at prediction time it receives a handful of real values and nothing for
the rest, and its reported accuracy describes a situation that never occurs in use.

So the rule is that a model may only train on features the application can actually collect. Every
dataset column is either mapped onto the canonical input schema or dropped before fitting. The
breast model uses all thirty of the available columns, because the panel requires a biopsy that
has already been taken and reported, and such a report carries all thirty; an earlier version asked
for four and paid 0.272 of AUC on small lesions for the difference. The pancreatic model drops the urinary panel and drops `stage`, which is only known after
diagnosis and would leak the answer. The prostate model drops tumour volume and capsular
penetration because those are surgical findings, not screening inputs, and converts the dataset's
log PSA back into the ng/mL a patient reads off a report.

Categorical columns are translated into one shared encoding, so a smoking answer means the same
thing to every model that consumes it.

Each saved model carries its own feature list and the training median of every one of its features.
Values the patient does not supply are filled with that median, not with zero. A zero BMI is not a
missing BMI, it is an impossible patient, and tree models will happily split on it.

### Stage 4: Inference

Four soft voting ensembles are served, one per domain, each combining an XGBoost gradient boosted forest with
an Extra Trees classifier.

The two learners are paired because they fail differently. XGBoost fits trees sequentially, each
one correcting the residual error of the ensemble so far, which makes it sharp but prone to chasing
noise in small cohorts. Extra Trees fits many deep trees independently and randomises the split
thresholds rather than optimising them, which makes it high variance individually but well
decorrelated in aggregate. Averaging their predicted probabilities is steadier than either alone,
particularly on the smaller cohorts here.

Class imbalance is handled inside each learner rather than by resampling: positive class weighting
in XGBoost, balanced subsampling in Extra Trees.

### Stage 5: Clinical thresholds

Established clinical decision limits are evaluated separately from the models. PSA at 4.0 ng/mL,
CA 19-9 at 37 U/mL, AFP at 10 ng/mL, total bilirubin at 1.2 mg/dL, calcium at 10.3 mg/dL, and the
rest.

Crossing one raises a flag on the relevant panel. It does not modify the model's probability. The
number shown to the user is the model's own output, and the clinical rules sit beside it as
separate, labelled information. Mixing the two would produce a figure that is neither a probability
nor a rule, and would make the reported AUC meaningless as a description of what the user sees.

### Stage 6: Attribution

Feature importance is averaged across both members of each ensemble, then returned with every
prediction, ranked and paired with the patient's own reading and the clinical limit for that
marker. The interface renders this as a weighted list plus two charts: a marker profile and a
comparison of each value against its reference limit.

This is a ranking of what the model relies on across all patients. It is not a per patient
decomposition of an individual score, and the documentation says so. A Shapley decomposition would
be the correct tool for that claim.

---

## 4. Training data

Every model was trained on de-identified, publicly released clinical research data.

| Panel | Source | Records | What counts as positive |
|---|---|---|---|
| Liver | NHANES 2005-2018, US adults | 35,511 | A clinical liver disease diagnosis, not liver cancer |
| General | NHANES 2005-2014, US adults | 23,923 | A cancer diagnosis within four years of the blood draw |
| Cervical | Hospital Universitario de Caracas | 858 | A positive cervical biopsy |
| Pancreatic | Pancreatic biomarker cohort, 3 tissue banks | 600 | Confirmed adenocarcinoma |
| Breast | Wisconsin Diagnostic Breast Cancer | 569 | A malignant fine needle aspirate |
| Ovarian | Third Affiliated Hospital of Soochow University | 349 | Ovarian cancer on histology, against a benign ovarian tumour |
| ~~Prostate~~ | Stanford prostate cohort | 97 | Gleason score of 7 or above, withdrawn |

Two of the newer cohorts deserve a note on how their controls were chosen, because it changes
what the number means.

The **ovarian** controls are not healthy volunteers. They are 178 women whose ovarian mass turned
out to be benign, against 171 whose mass turned out to be cancer, all of them operated on and all
labelled from the resected specimen. Healthy versus cancer would be an easy problem and a useless
one, since nobody needs a model to tell a woman with a large pelvic mass apart from a woman
without one. Malignant versus benign is the decision a clinician actually faces.

Its source values are SI and are converted to conventional US units at load time in
`fetch_ovarian.py`, because the rest of this project reads American lab reports. Albumin at 42 is
4.2 g/dL, not a critical value. Silently mixing unit systems is what drove the German liver
transfer below chance, so every conversion is explicit and listed in that file.

The **cervical** cohort ships with four columns deliberately deleted. Dx, Dx:Cancer, Dx:CIN and
Dx:HPV record a diagnosis the patient already carries, so training on them would predict a biopsy
result from the fact that the disease is already known. That single decision is the difference
between the 0.725 reported here and the near-perfect numbers usually published on this dataset.

The pancreatic target deserves a note. Its diagnosis column has three classes: healthy controls,
benign hepatobiliary disease, and pancreatic ductal adenocarcinoma. Treating it as binary and
reading the second class returns the probability of benign disease, not cancer, which is a subtle
and completely wrong answer. The positive class is defined explicitly as adenocarcinoma, separated
from both of the other two.

---

## 5. Measured performance

Held-out split cut before any model was fitted. Bootstrap intervals, 2,000 resamples.
Reproduce with `python evaluate.py`.

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

The general panel is the weak one and the page says so. It adds very little over knowing
someone's age and sex, which is a real limitation rather than a rounding error.

### Cross-validated against held-out, which is where cervical shows its weakness

| Panel | CV AUC | Held-out AUC | Gap |
|---|---|---|---|
| Pancreatic | 0.972 | 0.969 | -0.003 |
| Ovarian | 0.942 | 0.949 | +0.007 |
| Breast | 0.954 | 0.972 | +0.018 |
| Liver | 0.734 | 0.753 | +0.019 |
| General | 0.761 | 0.732 | -0.029 |
| Cervical | 0.587 | 0.725 | **+0.138** |

Five panels agree with themselves to within 0.03. Cervical does not. A 0.138 gap on 55 positive
biopsies means the held-out split was a favourable one and the honest estimate sits nearer 0.6
than 0.725. It still excludes chance and still beats age alone by 0.267, which is why it ships,
but it is the least certain thing in this project and both numbers are published rather than only
the flattering one.

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

| Panel | n | AUC none | AUC isotonic | AUC sigmoid | Brier none | Brier isotonic |
|---|---|---|---|---|---|---|
| Pancreatic | 600 | 0.969 | 0.966 | 0.968 | 0.0585 | 0.0533 |
| Breast | 569 | 0.957 | 0.954 | 0.955 | 0.0727 | 0.0712 |
| Ovarian | 349 | 0.935 | 0.933 | 0.935 | 0.0985 | 0.0889 |
| Prostate | 212 | 0.829 | 0.830 | 0.831 | 0.1661 | 0.1664 |
| Lung | 21,916 | 0.825 | 0.822 | 0.825 | 0.0274 | **0.0047** |
| Bowel | 23,794 | 0.808 | 0.805 | 0.814 | 0.0288 | **0.0040** |
| General | 23,923 | 0.756 | 0.756 | 0.758 | 0.1524 | **0.0294** |
| Liver | 35,511 | 0.738 | 0.742 | 0.743 | 0.1675 | **0.0360** |

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

### Triage panels are judged against referral prevalence, not SEER

Ovarian and cervical run after something has already been found, so the prior that matters is
prevalence in the referred group. Ovarian uses the roughly 20 percent malignancy rate among women
taken to surgery for an adnexal mass. Cervical uses the 6.4 percent positive-biopsy rate in its
own referral cohort.

The alternative is not a rounding difference. At SEER population incidence the cervical panel
would flag about **7,383 women per true case**. At referral prevalence it flags 9.4, at 96.2
percent NPV. The model is identical and only the prior changed. That is exactly why neither panel
is offered as a population screen, and why the prior is printed on the panel itself.

### Where the bloodwork earns its place, and where it does not

The ovarian panel is the one that vindicates the premise of this application. Its cases are
seventeen years older than its controls, 53 against 36, which is the same trap the general panel
fell into, so age was made a baseline and had to be beaten:

| Feature set | Inputs | AUC |
|---|---|---|
| Age and menopausal status | 2 | 0.792 |
| Routine bloodwork only, no age, no tumour markers | 20 | **0.938** |
| Tumour markers only | 5 | 0.931 |
| Everything | 27 | **0.941** |

A blood count and a metabolic panel, carrying no age and no tumour markers, reach 0.938.

The general panel is the mirror image, measured the same way on a held-out NHANES cycle:

| Feature set | Inputs | AUC |
|---|---|---|
| Age and sex only | 2 | 0.729 |
| Age, sex, lifestyle | 5 | **0.748** |
| All 14 routine blood values | 14 | **0.663** |
| Everything combined | 19 | 0.737 |

Fourteen blood values score worse than knowing someone's age. Routine chemistry is not a general
cancer detection test, which is why the specific panels here use disease specific markers and why
commercial multi-cancer blood tests use cell-free DNA rather than routine chemistry.

Taken together, those two tables are the honest summary of this whole project. Reading a lab
report works when there is a specific disease and a specific marker to read, and it does not work
as an undirected sweep for any cancer at all.

### The general panel was rebuilt around a screening question

It used to predict "have you ever been told you had cancer". A person cured thirty years ago
counted as positive, so the model was largely predicting age.

NHANES 2005 to 2014 records age at diagnosis, so the cohort can be cut properly: positives are
people diagnosed **within four years of the blood draw**, and long-ago survivors are excluded
rather than relabelled. That is a screening question. On a held-out cycle the gain over age and
sex rose from 0.004 to 0.019.

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

`tools/audit.py` checks that every question the form asks is consumed by some live panel. It found
one that was not: hours of exercise per week. The form asked, and no model had read it since the
general and liver panels moved to NHANES.

There were two honest responses: wire it up, or stop asking. Which one depended on whether it
carried information, so it was measured rather than guessed. NHANES has recorded physical activity
under the Global Physical Activity Questionnaire since 2007, so recreational exercise hours were
pulled and offered to the general panel. Work activity was deliberately excluded, because a
warehouse shift and a run are not the same exposure and the form's question says "exercise".

Because the instrument only exists from 2007, the answer is missing by cycle rather than at random.
Median imputation across that gap would hand 4,147 people a made-up value and let the model learn
"no exercise recorded" as a cycle marker, so every arm was run twice: once pooled, once restricted
to the four cycles that actually asked.

It did not help either way. The general panel's gain over age and sex went from +0.005 to +0.003 on
the cycles that asked, and the same on the pooled cohort: a small loss, consistently, not a small
win. Physical inactivity is a real cancer risk factor at the population level, but that is not the
same claim as adding information once you already know a person's age, sex, BMI, smoking and
drinking. It is largely spoken for by those.

So the question was removed from the form. Waist circumference was tested in the same run, on the
reasoning that BMI cannot separate a heavy-set person from a centrally obese one and that it is
central adiposity the literature ties to risk. It came out at +0.003, in the right direction and
consistent across both cohorts, but under the 0.005 bar set before the experiment ran. It was not
adopted, and the bar was not moved afterwards to let it in. The column is still collected, because
it is free and a different cohort or target may make better use of it.

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

### Choosing the operating point

A fixed 0.5 threshold drove measured sensitivity to 0.008 once prevalence was realistic. Each
panel now selects its threshold by Youden's J on out-of-fold predictions inside the training data
and freezes it in the bundle, so the interface, the evaluation and the prospective analysis all
use one number.

Probabilities are clipped to the 0.1 to 99.9 range before display. Isotonic calibration maps its
top bin to exactly 1.0, and no panel here has earned the right to print 100 percent.

## 6. The sample case system

Pressing Generate case draws one of fifty records at random and states which panel should come out
on top before the analysis runs.

The cases are real records from the training data, expressed in the application's input schema,
with panels a given record does not cover left at a normal adult reading. Nothing is invented.

Two properties matter. First, positives are sampled evenly across each model's probability range
rather than taking the most confident record, so scores land anywhere between the low fifties and
ninety nine instead of always reading the ceiling. Second, fifteen of the fifty are negative
records, so a meaningful share of generated cases come back clean, which is what a screening tool
does most of the time.

Every case is scored before it is written into the pool and is only kept if the panel that comes
out on top matches the expectation shown to the user. The generator is committed as
`make_cases.py`, so the pool can be regenerated whenever the models are retrained.

---

## 7. Architecture

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 App Router, React 19, TypeScript, Tailwind CSS v4, Recharts, deployed on Vercel |
| Backend | FastAPI, Uvicorn, Pydantic, pdfplumber, pandas, deployed on Render |
| Machine learning | XGBoost, scikit-learn Extra Trees, soft voting ensembles, joblib serialisation |

### Endpoints

| Method | Route | Purpose |
|---|---|---|
| POST | `/predict` | A flat object of lab values and history, every field optional. Returns a ranked per panel assessment with attribution and model metrics |
| POST | `/parse-pdf` | Up to five PDFs in, extracted biomarker values out |
| GET | `/models` | The model registry with measured AUC, sensitivity, and specificity |

Models are loaded once at API startup and held in memory. Bundles are compressed at serialisation
time, which takes the five of them from 119 MB to 33 MB and keeps them small enough to commit and
deploy.

### Repository

```
oncovision/
├── backend/
│   ├── api.py              FastAPI service, PDF parsing, scoring, gating
│   ├── models/             calibrated ensemble bundles, one per live panel
│   ├── model_metrics.json  measured performance
│   └── requirements.txt
├── data/                   source training datasets
├── frontend/
│   └── src/app/
│       ├── page.tsx        assessment, guide, methodology, developer
│       ├── fields.ts       input schema and glossary
│       ├── cases.ts        generated sample case pool
│       └── layout.tsx
├── tools/audit.py          structural checks, exits non-zero to gate a commit
├── experiments/            every question that was measured, negative ones included
├── evaluate.py             held-out evaluation, CIs, calibration, PPV, baselines
├── external_validation.py  trains on one country, tests on another
├── train_models.py         training pipeline and panel config
├── make_cases.py           sample case generator
└── PROJECT.md
```

### Running it

```bash
# train
pip install -r requirements.txt

# structural checks first: duplicate keys, panels training on features the form
# never asks for, model features with no range check, panels with no declared
# type. Exits non-zero, so it can gate a commit.
python tools/audit.py

python train_models.py

# regenerate the sample cases
python make_cases.py

# serve
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL` and falls back to the deployed backend when it is not set.

---

## 8. Privacy

Uploaded PDFs are read into memory, parsed, and discarded inside the request. Nothing is written to
disk and no database is attached to the service. Lab values live in browser state for the length of
a session and are gone when the tab closes.

The service also does not accept what it cannot use. When the cervical panel was withdrawn, its
questions were taken out of the form, but the API went on accepting them: number of sexual
partners, age at first intercourse, pregnancies, contraceptive and IUD history, and STD and HPV
status. Nothing scored any of it. That is worse than dead code, because an endpoint that accepts a
sexual history is one that can receive and log one, and there was no reason for it to exist.

`tools/audit.py` now fails if the API accepts any field that no live panel reads and no form sends,
and separately if a withdrawn panel is still named as the consumer of an input. Seventeen fields
were removed when that check was first run.

---

## 9. Limitations

Stated in the application itself, not only here.

**Most cohorts are case-control, not screening series.** These records come from people who
already had a reason to be tested, so several cohorts run 6 to 49 percent positive against real
incidence measured in hundredths of a percent. That gap is why the precision table matters more
than the AUC table. The two NHANES panels are the exceptions: they are nationally representative.

**Two panels are triage, not screening, and must not be read as screening.** The ovarian panel
assumes a mass has already been found; the cervical panel assumes a woman is already being
assessed. Their precision is projected onto referral prevalence. Applied to an unselected
population the cervical panel would flag roughly 7,383 women per true case. The prior is printed
on each panel for exactly this reason.

**The cervical panel is the weakest evidence here.** Cross-validated AUC 0.587 against a held-out
0.725, on 55 positive biopsies. The gap means the held-out split was favourable and the honest
estimate is nearer 0.6. It ships because its interval excludes chance and it beats age alone by
0.267, but it is the one panel whose headline number should be treated with suspicion.

**The ovarian panel has no external cohort.** Single centre, single country, 349 patients, one
eight-year window. It is the strongest new panel by internal measurement and the least tested
across institutions. A second ovarian cohort is the highest-value thing that could be added next.

**The breast panel contradicts the schema rule.** Its four inputs are nuclear morphology from a
fine needle aspirate, which requires a biopsy that already happened. It interprets a biopsy rather
than screening for one. It also carries no age, sex or race, so it has no subgroup measurement at
all.

**The liver panel detects liver disease, not liver cancer.** It trains on 35,511 real NHANES
adults and is externally tested against India and Germany. Germany remains the hardest transfer
at 0.442, below chance. That was long guessed to be a unit or encoding bug; it is not.
ALT runs in opposite directions in the two cohorts because they capture mild and advanced
liver disease respectively. See the worked section below.

**The general panel barely beats age and sex**, 0.732 against 0.727. Routine bloodwork was
measured and made it worse. It is shipped with that number stated because removing it would hide
a real result about what lab reports can and cannot do.

**External validation is uneven.** Pancreatic is validated leave-one-tissue-bank-out, liver
against India, Germany and later NHANES cycles, general against a held-out cycle, breast against
WPBC for sensitivity only. Ovarian and cervical have none. Prostate was withdrawn partly because
none could be found.

**No prospective test and no IRB.** No real patient report has been run through this and followed
to an outcome. There is no ethics approval, registration, or clinical validation of any kind.

**The ensemble is within noise of logistic regression** on breast and pancreatic.

**Race and ethnicity are measured on the four NHANES panels, and three show a real gap.**
Lung and bowel are both weakest for Non-Hispanic Black patients, who have the higher mortality
from those cancers, and the general panel is weakest for Other Hispanic patients. Breast,
ovarian, pancreatic and prostate remain unmeasurable because their cohorts do not record it.
AUC by sex and age band is in `EVALUATION.md`, and by race in `experiments/fairness.py`.

**A low score is not a clear result.** Plenty of cancers produce entirely normal lab results early.
The tool can raise a question. It cannot answer one.

## 10. Provenance

Feature selection, biomarker weighting, and the clinical threshold table were developed under the
direct guidance of a clinical oncologist at UCI CHOC. That mentorship is what moved the design away
from reporting raw model probability alone and toward a system that reports the model output, the
clinical thresholds, and the model's own measured reliability side by side, so that a reader can
tell which part of the answer came from where.

---

**Palash Rakshit**
Founder and Lead Developer
[palash.raks@gmail.com](mailto:palash.raks@gmail.com) · [medtechcentral@gmail.com](mailto:medtechcentral@gmail.com) · [linkedin.com/in/Palash-Rakshit10](https://www.linkedin.com/in/Palash-Rakshit10)

*Oncovision is a diagnostic support prototype built for educational and research purposes. It is not
a substitute for professional medical advice, diagnosis, or treatment.*
