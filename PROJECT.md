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
| Liver | Hepatocellular cohort | 5,000 | A liver cancer diagnosis |
| General | Cancer risk cohort | 1,500 | A recorded cancer diagnosis |
| Pancreatic | Pancreatic biomarker cohort | 600 | Confirmed adenocarcinoma |
| Breast | Wisconsin Diagnostic Breast Cancer | 569 | A malignant fine needle aspirate |
| Prostate | Stanford prostate cohort | 97 | Gleason score of 7 or above |

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
| Liver disease | 35,511 NHANES adults | 0.753 | 0.723 to 0.784 | 4.2% | 0.55 | 0.80 | 9.9 |
| General cancer | 23,923 NHANES adults | 0.732 | 0.692 to 0.770 | 2.9% | 0.68 | 0.65 | 16.9 |
| ~~Prostate~~ | 97 Stanford records | 0.786 | 0.505 to 0.99 | withdrawn | | | |

### Does each panel beat the obvious baseline?

| Panel | Model | Logistic | Age and sex alone | Gain |
|---|---|---|---|---|
| Pancreatic | 0.969 | 0.968 | 0.500 | +0.469 |
| Liver | 0.753 | 0.731 | 0.602 | +0.151 |
| General | 0.732 | 0.731 | 0.727 | **+0.005** |

The general panel is the weak one and the page says so. It adds very little over
knowing someone's age and sex, which is a real limitation rather than a rounding
error.

### The general panel was rebuilt around a screening question

It used to predict "have you ever been told you had cancer". A person cured
thirty years ago counted as positive, so the model was largely predicting age.

NHANES 2005 to 2014 records age at diagnosis, so the cohort can be cut properly:
positives are people diagnosed **within four years of the blood draw**, and
long-ago survivors are excluded rather than relabelled. That is a screening
question. On a held-out cycle the gain over age and sex rose from 0.004 to 0.019.

### Routine bloodwork does not detect general cancer

Measured rather than assumed, on a held-out NHANES cycle:

| Feature set | Inputs | AUC |
|---|---|---|
| Age and sex only | 2 | 0.729 |
| Age, sex, lifestyle | 5 | **0.748** |
| All 14 routine blood values | 14 | **0.663** |
| Everything combined | 19 | 0.737 |

Fourteen blood values score worse than knowing someone's age. A complete blood
count and metabolic panel are not cancer detection tests, which is exactly why
the specific panels here use disease specific markers, and why commercial
multi-cancer blood tests use cell-free DNA rather than routine chemistry.

The general panel therefore reads risk factors, not the lab report, and that is
a measured decision rather than an oversight.

### Choosing the operating point

A fixed 0.5 threshold drove measured sensitivity to 0.008 once prevalence was
realistic. Each panel now selects its threshold by Youden's J on out-of-fold
predictions inside the training data and freezes it in the bundle, so the
interface, the evaluation and the prospective analysis use one number.

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

**Every cohort is case-control, not a screening series.** These records come from people who
already had a reason to be tested, so the cohorts run 21 to 37 percent positive against real
incidence measured in hundredths of a percent. That gap is why the precision table matters more
than the AUC table.

**The breast panel contradicts the schema rule.** Its four inputs are nuclear morphology from a
fine needle aspirate, which requires a biopsy that already happened. It interprets a biopsy rather
than screening for one.

**The liver panel detects liver disease, not liver cancer.** It trains on 35,511 real NHANES
adults and is externally tested against India and Germany. Germany remains the hardest transfer
at 0.442, which is kept visible.

**No external validation.** Every number comes from a held-out split of the same cohort the model
trained on. This is the single largest remaining gap.

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
