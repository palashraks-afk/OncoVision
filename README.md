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

1. **Your lab reports.** 23 values across body metrics, complete blood count, metabolic panel,
   liver panel, tumour markers, and breast mass morphology. Uploaded as PDFs and parsed
   automatically, or typed in.
2. **Information about you.** 7 items no lab report contains: sex, smoking, alcohol, exercise,
   hepatitis B and C, diabetes. Inherited risk, prior diagnosis, family history and cirrhosis were
   removed once no shipped panel read them, because asking for what nothing uses is only friction.

Four calibrated models score the combination, and the interface reports what drove each score,
how accurate that model is, and what the score is worth at real population prevalence.

---

## What ships

Four panels ship. One was withdrawn because the evidence did not support serving it.

| Panel | Trained on | Test AUC | 95% CI | Sens | Spec | Flagged per true case |
|---|---|---|---|---|---|---|
| General cancer | 37,564 US adults, NHANES 2005-2018 | 0.781 | 0.764 to 0.797 | 0.80 | 0.62 | 5.6 |
| Liver disease | 35,511 US adults, NHANES 2005-2018 | 0.753 | 0.721 to 0.782 | 0.55 | 0.80 | 9.9 |
| Breast malignancy | 569 biopsies, Wisconsin | 0.972 | 0.941 to 0.994 | 0.81 | 0.94 | 52.7 |
| Pancreatic cancer | 600 samples, 3 tissue banks | 0.969 | 0.937 to 0.991 | 1.00 | 0.88 | 842.8 |
| ~~Prostate~~ | 97 records, Stanford | 0.786 | **0.505** to 0.99 | | | withdrawn |

All measured on a 20% split cut before any model was fitted. Reproduce with `python evaluate.py`.

### The general panel was rebuilt, and its AUC fell on purpose

It used to report 0.966. That number was worthless: trained on a 1,500-record risk-factor cohort,
it scored **0.574** on a representative sample of US adults. It was measuring its cohort, not
cancer risk.

Retrained on **37,564 NHANES adults** and tested on cycles it never saw, it reports 0.781, and
holds at **0.804** under temporal validation. A lower number that is true is worth more than a
higher one that is not.

The liver panel moved the same way: 35,511 NHANES adults, chemistry plus diabetes and hepatitis
serology, with India and Germany kept as independent external cohorts instead of training data.

### Choosing the operating point instead of assuming 0.5

Once the panels trained on real prevalence, a fixed 0.5 threshold drove measured sensitivity to
**0.008**. That was not a broken model. A calibrated model on a 4 percent condition is correctly
reporting that almost nobody is more likely than not to have it, and 0.5 is simply the wrong cut.

Each panel now picks its threshold by Youden's J on out-of-fold predictions inside the training
data, and freezes it in the model bundle so the interface, the evaluation and the prospective
analysis all use the same number.

| Panel | Threshold | Sensitivity |
|---|---|---|
| Liver | 3.6% | 0.55 |
| General | 9.3% | 0.80 |
| Pancreatic | 25.8% | 1.00 |
| Breast | 40.5% | 0.81 |

The interface shows the calibrated probability against that panel's reference band, the way a lab
report shows a value against its reference interval, rather than colouring everything by a
universal 50 percent line.

One thing this fixed by accident: the threshold was first chosen on uncalibrated out-of-fold scores
and applied to calibrated probabilities. Two different scales, and the symptom was a
reasonable-looking threshold that caught nobody.

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
