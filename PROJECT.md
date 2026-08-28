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
breast model uses four of the thirty available columns because those four are what a user can
supply. The pancreatic model drops the urinary panel and drops `stage`, which is only known after
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

| Panel | Trained on | Test AUC | 95% CI | Threshold | Sens | Spec | Flagged per true case |
|---|---|---|---|---|---|---|---|
| Breast malignancy | 569 Wisconsin biopsies | 0.972 | 0.942 to 0.994 | 38.8% | 0.81 | 0.94 | 52.7 |
| Pancreatic cancer | 600 samples, 3 tissue banks | 0.969 | 0.938 to 0.991 | 16.6% | 1.00 | 0.88 | 842.8 |
| Ovarian malignancy | 349 operated ovarian masses | 0.949 | 0.886 to 0.994 | 58.1% | 0.85 | 0.94 | 1.3 |
| Prostate cancer | 212 biopsied men | 0.840 | 0.705 to 0.952 | 66.9% | 0.80 | 0.78 | 1.4 |
| Lung cancer | 21,916 adults with tobacco exposure | 0.829 | 0.730 to 0.902 | 1.0% | 0.57 | 0.85 | 55.1 |
| Bowel cancer | 23,794 NHANES adults | 0.793 | 0.708 to 0.867 | 1.0% | 0.53 | 0.85 | 768.4 |
| Liver disease | 35,511 NHANES adults | 0.753 | 0.723 to 0.784 | 4.2% | 0.55 | 0.80 | 9.9 |
| General cancer | 23,923 NHANES adults | 0.732 | 0.692 to 0.770 | 2.9% | 0.68 | 0.65 | 16.9 |
| ~~Cervical~~ | 858 Caracas referrals | 0.725 | withdrawn, a lucky split | | | | |

### Does each panel beat the obvious baseline?

| Panel | Model | Logistic | Age and sex alone | Gain |
|---|---|---|---|---|
| Pancreatic | 0.969 | 0.968 | 0.500 | +0.469 |
| Liver | 0.753 | 0.731 | 0.602 | +0.151 |
| Prostate | 0.840 | 0.876 | 0.661 | +0.179 |
| Ovarian | 0.949 | 0.911 | 0.813 | +0.136 |
| Lung | 0.829 | 0.785 | 0.778 | +0.051 |
| Bowel | 0.809 | 0.800 | 0.771 | +0.038 |
| General | 0.732 | 0.731 | 0.727 | **+0.005** |

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
| Smallest third by nuclear area | 190 | 6 | **0.739** |
| Middle third | 189 | 39 | 0.806 |
| Largest third | 190 | 167 | 0.923 |

The panel scores 0.954 overall and 0.739 on the smallest lesions, a spread of
0.216. It is materially weaker exactly where it would be most useful, and that
now appears on the panel instead of being invisible. Race and age remain
unmeasurable from this cohort and that limitation stands unchanged.
Reproduce with `python experiments/breast_subgroups.py`.

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

| Panel | Rows | Events | Mean AUC | Spread across splits | Shipped split | Percentile |
|---|---|---|---|---|---|---|
| Pancreatic | 600 | 130 | 0.969 | 0.939 to 0.995 | 0.969 | 50th |
| Breast | 569 | 212 | 0.959 | 0.901 to 0.991 | 0.972 | 73rd |
| Ovarian | 349 | 171 | 0.928 | 0.852 to 0.969 | 0.949 | 70th |
| Liver | 35,511 | 1,436 | 0.750 | 0.744 to 0.756 | 0.753 | 60th |
| General | 23,923 | 750 | 0.743 | 0.692 to 0.772 | 0.732 | 20th |
| Cervical | 858 | 55 | **0.594** | **0.421 to 0.789** | 0.725 | **97th** |

Five panels sit between the 20th and 73rd percentile of their own split
distribution, which is what a representative number looks like. Cervical sat at
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
│   ├── api.py              FastAPI service
│   ├── models/             5 serialised ensemble bundles
│   ├── model_metrics.json  measured performance
│   └── requirements.txt
├── data/                   source training datasets
├── frontend/
│   └── src/app/
│       ├── page.tsx        assessment, guide, methodology, developer
│       ├── fields.ts       input schema and glossary
│       ├── cases.ts        generated sample case pool
│       └── layout.tsx
├── train_models.py         training pipeline
├── make_cases.py           sample case generator
└── PROJECT.md
```

### Running it

```bash
# train
pip install -r requirements.txt
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

**No race or ethnicity in any cohort**, so accuracy across those groups is unmeasured rather than
acceptable. AUC by sex and age band is in `EVALUATION.md`.

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
