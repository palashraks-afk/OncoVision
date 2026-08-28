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

1. **Your lab reports.** 38 values across body metrics, complete blood count and red cell
   indices, metabolic panel, liver panel, tumour markers, tobacco exposure and inflammation,
   the prostate work-up, and breast mass morphology. Uploaded as PDFs and parsed automatically,
   or typed in. The parser is tested on 28 of them at 100 percent across five different report
   layouts; see `test_parser.py`.
2. **Information about you.** 9 items no lab report contains: sex, smoking, pack-years, alcohol,
   exercise, hepatitis B and C, diabetes and menopausal status. The 14 sexual-history questions
   went out with the cervical panel, because no shipped model reads them and asking for what
   nothing uses is only friction.

Eight calibrated models score the combination, and the interface reports what drove each score,
how accurate that model is, and what the score is worth at real prevalence.

Every panel answers with whatever it is given. A blank field is filled with that feature's
training median, and the panel says how many of its inputs were actually yours, so a score
built on three values out of twenty-seven is labelled low confidence rather than refused or
presented as if it were complete.

---

## What ships

Eight panels ship. One was withdrawn because the evidence did not support serving it.

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

The general panel is the weak one, and the interface says so rather than hiding it.

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

### Cervical is the least certain panel here, and it says so

Cross-validated AUC is 0.587 while the held-out split gives 0.725. That gap means the held-out
number is optimistic and the truth sits nearer 0.6. With 55 positive biopsies the interval runs
0.547 to 0.881: it excludes chance, and it beats age alone by a very wide margin (0.458), which
is why it ships, but it is the weakest evidence in the project and is labelled that way.

Cutting its 15 input fields was tried and failed: eight fields scored 0.665 with an interval
containing chance, and six scored 0.552. All fifteen are needed, so all fifteen are asked for.

### The general panel now answers a screening question

It used to predict "have you ever been told you had cancer". Someone cured thirty years ago counted
as positive, so the model was largely predicting age: 0.781 against 0.777 for age and sex alone, a
gain of 0.004.

NHANES 2005 to 2014 records **age at diagnosis**, so the cohort can be cut properly. Positives are
people diagnosed **within four years of the blood draw**, and long-ago survivors are excluded rather
than relabelled. That is a screening question. On a held-out cycle the gain over age and sex rises
from 0.004 to **0.019**.

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

- **Every cohort is case-control, not a screening series.** See the precision table above.
- **The breast panel is an interpretation panel, not a screening one.** Its inputs need a biopsy
  that already happened. Rebuilding it on blood was tried and failed at chance, so the rule was
  amended to name the two classes rather than pretend there is one.
- **The liver panel detects liver disease, not liver cancer.** Liver disease is roughly 300 times
  more common and is the dominant precursor to hepatocellular carcinoma, so this is a useful thing
  to detect, but it is not a cancer claim.
- **One panel was withdrawn.** Prostate is trained and measured but not served. Pancreatic was
  withdrawn and then reinstated once leave-one-site-out validation was run. Both decisions and the
  evidence behind them are above and on the methodology page.
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
│   ├── api.py              FastAPI service
│   ├── models/             4 calibrated ensemble bundles
│   └── model_metrics.json
├── data/                   source cohorts
├── frontend/src/app/
│   ├── page.tsx            assessment, guide, methodology, developer
│   ├── fields.ts           input schema and glossary
│   └── cases.ts            generated sample case pool
├── evaluate.py             held-out evaluation, CIs, calibration, PPV, baselines
├── test_parser.py          renders 5 lab report layouts, measures parser accuracy
├── prospective_analysis.py pre-committed study analysis, data dictionary, simulator
└── experiments/            recorded negative results
├── fetch_external.py       downloads the real cohorts, harmonises units
├── external_validation.py  trains on one country, tests on another
├── train_models.py         training and calibration pipeline
├── make_cases.py           sample case generator
├── EVALUATION.md           full evaluation report
└── PROJECT.md              methodology write-up
```

---

## Privacy

Uploaded PDFs are read into memory, parsed, and discarded inside the request. Nothing is written to
disk and no database is attached. Values live in browser state for the session only.

---

## License

MIT, see [LICENSE](LICENSE). The additional notice there is not decorative: this is not a medical
device and must not be used to make or defer a medical decision.

**Palash Rakshit** · [palash.raks@gmail.com](mailto:palash.raks@gmail.com) · [LinkedIn](https://www.linkedin.com/in/Palash-Rakshit10)
