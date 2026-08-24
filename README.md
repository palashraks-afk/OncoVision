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
2. **Information about you.** 11 items no lab report contains: sex, smoking, alcohol, exercise,
   inherited risk, prior diagnosis, family history, hepatitis B and C, cirrhosis, diabetes.

Four calibrated models score the combination, and the interface reports what drove each score,
how accurate that model is, and what the score is worth at real population prevalence.

---

## What ships, and what does not

Four panels ship. One was built, measured, and withdrawn because the evidence did not support
serving it.

| Panel | Test AUC | 95% CI | PPV at real prevalence | Flagged per true case | External test |
|---|---|---|---|---|---|
| **Liver disease** | 0.892 | 0.857 to 0.924 | **54.4%** | **1.8** | 3 countries, leave-one-out |
| General cancer | 0.966 | 0.937 to 0.989 | 20.6% | 4.9 | NHANES, 5,173 adults |
| Breast malignancy | 0.972 | 0.940 to 0.994 | 2.44% | 41 | WPBC, 198 patients, partial |
| Pancreatic cancer | 0.969 | 0.938 to 0.991 | 0.19% | 525 | 3 tissue banks, leave-one-site-out |
| ~~Prostate~~ | 0.786 | **0.505** to 0.99 | 0.22% | 453 | **none exists** |

All measured on a 20% split cut before any model was fitted. Intervals are bootstrap percentile
intervals over 2,000 resamples. Reproduce with `python evaluate.py`.

### Pancreatic: withdrawn, then reinstated on evidence

This panel was withdrawn for having no external test. That was wrong, and finding out why is the
most instructive thing in this repository.

Its own cohort is drawn from **three independent tissue banks**, and the site column had been
discarded as an identifier during preprocessing. Training on two and testing on the third is a real
test between institutions.

| Held out site | Train n | Test n | Cases | AUC | 95% CI | Logistic |
|---|---|---|---|---|---|---|
| BPTB | 235 | 365 | 74 | 0.978 | 0.962 to 0.990 | 0.982 |
| CPTB | 459 | 141 | 34 | 0.959 | 0.911 to 0.991 | 0.973 |
| UPTB | 506 | 94 | 22 | 0.950 | 0.895 to 0.990 | 0.963 |

Mean leave-one-site-out AUC **0.962**, every interval excludes chance, and the drop from the
internal random split is **0.007**. The panel transfers between institutions, so it was reinstated.

What that does not show is transfer to a screening population. All three sites are pancreatic
tissue banks running 20 to 24 percent cases, and at real incidence this panel still flags roughly
**525 people per true case**. Both facts are true at once and the interface reports both.
Reproduce with `python experiments/pancreatic_multisite.py`.

### Prostate: withdrawn, and the search is documented

Test AUC 0.786 with a 95% CI of **0.505 to 0.99**. The lower bound sits on chance. Specificity
0.571 with an interval of 0.167 to 1.0, which carries no information on 20 test records.

External validation was searched for and does not exist:

- The Stanford cohort has **no site column**, so unlike the pancreatic cohort it cannot be split by
  institution.
- NHANES measured serum PSA on **4,697 men** across 2005 to 2010, which looked like the answer. It
  contains only **17 prostate cancer cases**, because men already diagnosed are excluded from the
  PSA subsample. That is far below the roughly 96 events needed.

97 records and two usable features, with no route to an external test, cannot support a clinical
claim.

## External validation

**Liver, three independent real cohorts:** ILPD India (583), HCV Germany (589), NHANES USA (4,887).
Every ordered pair, so no direction is cherry-picked. The German cohort reports in SI units, so
albumin, total protein and bilirubin had to be converted before any comparison was valid.

| Direction | Internal | External | Drop |
|---|---|---|---|
| Germany to India | 0.995 | 0.698 | 0.297 |
| USA to India | 0.700 | 0.640 | 0.060 |
| India to Germany | 0.785 | 0.623 | 0.162 |
| India to USA | 0.785 | 0.575 | 0.210 |
| Germany to USA | 0.995 | **0.531** | **0.464** |
| USA to Germany | 0.700 | 0.442 | 0.258 |

**A model trained on German patients scores 0.995 on its own held-out data and 0.531 on Americans.**
Same model, same task, 0.464 lost purely to the change of population.

**General to NHANES 2017-2018**, 5,173 US adults: internal 0.966, external **0.596**, a drop of
0.370. NHANES matters because it is not case-control.

**Breast to WPBC**, 198 independent confirmed cancer patients: internal sensitivity 0.915, external
**0.894**. No AUC, because that cohort has no benign class. Breast is the panel that transfers.

### Leave one cohort out, and what it changed

| Held out | Trained on | Train n | AUC | 95% CI | Logistic |
|---|---|---|---|---|---|
| India | Germany + USA | 5,476 | 0.710 | 0.664 to 0.753 | 0.753 |
| Germany | India + USA | 5,470 | 0.641 | 0.564 to 0.724 | 0.654 |
| USA | India + Germany | 1,172 | 0.580 | 0.547 to 0.612 | 0.655 |

Mean external AUC is **0.585 training on one cohort and 0.644 training on two**. Cohort diversity is
worth about 0.06 on an unseen population, and logistic regression beat the ensemble in all three
folds. Both findings were acted on: the liver panel now trains on all three pooled, 6,059 patients
across three continents. Its honest expected performance on a genuinely new population is nearer
**0.69** than the 0.892 measured internally.

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
