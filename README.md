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

## Honest numbers

All measured on a 20% test split cut **before** any model was fitted, never used for training,
model selection, or calibration. Intervals are bootstrap percentile intervals over 2,000 resamples.
Reproduce with `python evaluate.py`.

| Panel | Test AUC | 95% CI | Sens. | Spec. | Logistic baseline | Test n |
|---|---|---|---|---|---|---|
| Breast | 0.972 | 0.940 to 0.994 | 0.786 | 0.958 | 0.964 | 114 |
| Pancreatic | 0.969 | 0.938 to 0.991 | 0.731 | 0.947 | 0.968 | 120 |
| General | 0.966 | 0.937 to 0.989 | 0.910 | 0.984 | 0.917 | 300 |
| Liver | 0.892 | 0.857 to 0.924 | 0.595 | 0.984 | 0.892 | 1,212 |

The pipeline picks whichever model wins on cross-validated AUC inside the training data, rather
than assuming the ensemble does. Two panels ship plain logistic regression because it won.

### Precision at real prevalence, which is the number that decides usability

Cohort AUC is measured on disease-enriched data. Projecting measured sensitivity and specificity
onto real population prevalence gives what a user would actually experience.

| Panel | Cohort positive | Population prevalence | PPV there | People flagged per true case |
|---|---|---|---|---|
| **Liver** | 12% | 3.1% | **54.4%** | **1.8** |
| General | 37% | 0.4507% | 20.6% | 4.9 |
| Breast | 37% | 0.1325% | 2.44% | 41 |
| Pancreatic | 22% | 0.0139% | **0.19%** | **525** |

Read the bottom row. As a population screen the pancreatic panel would flag roughly 525 people for
every one who has the disease. The liver panel flags fewer than two, because it is trained on real
pooled data, its target is common, and it was built after the generalisation problem was measured
rather than before.

### External validation

Every held-out number above still shares a hospital, assay machines and population with its
training data. Three of four panels now have a genuine outside test.

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
Same model, same task, 0.464 lost purely to the change of population. Every internal number in this
README should be read with that in mind.

**General panel to NHANES 2017-2018**, 5,173 US adults: internal 0.966, external **0.596**, a drop
of 0.370. NHANES matters because it is not case-control. It samples the public at 10.3% prevalence.

**Breast to WPBC**, 198 independent confirmed cancer patients: internal sensitivity 0.915, external
**0.894**. No AUC, because that cohort has no benign class, so this is a partial test and is
labelled as one. Breast is the panel that transfers.

**Pancreatic**: no public cohort shares its feature set. Untested.

### Leave one cohort out, and what it changed

Train on two countries, test on the third.

| Held out | Trained on | Train n | AUC | 95% CI | Logistic |
|---|---|---|---|---|---|
| India | Germany + USA | 5,476 | 0.710 | 0.664 to 0.753 | 0.753 |
| Germany | India + USA | 5,470 | 0.641 | 0.564 to 0.724 | 0.654 |
| USA | India + Germany | 1,172 | 0.580 | 0.547 to 0.612 | 0.655 |

Mean external AUC is **0.585 training on one cohort and 0.644 training on two**, so cohort
diversity is worth about 0.06 on a population the model has never seen. Logistic regression also
beat the ensemble in all three folds.

Both findings were acted on: the liver panel now trains on all three pooled, 6,059 patients across
three continents. Its honest expected performance on a genuinely new population is nearer **0.69**
than the 0.892 measured internally.

### PDF parser accuracy

The parser sits upstream of every model, and was previously unmeasured. `test_parser.py` renders
lab reports in five deliberately different layouts from known ground truth and compares field by
field.

| | Before | After |
|---|---|---|
| Overall field accuracy | 68.9% | **100%** (90/90) |
| Range-printed-before-result layout | 5.6% | 100% |

Four real bugs were found and fixed: `ast` was matching inside the word "fasting", "Platelet Count"
never matched the plural-only pattern, "Alpha-Fetoprotein" never matched the unhyphenated pattern,
and reference ranges were being read as results. The root cause was flattening the whole document
into one string; reports are line-oriented, so the parser now works line by line and blanks
reference intervals before looking for a value.

### A panel that was withdrawn

**Prostate is trained, measured, and not served.** Test AUC 0.786, 95% CI 0.505 to 0.99, so the
lower bound sits on chance. Specificity 0.571 with a CI of 0.167 to 1.0, an interval carrying no
information on 20 test records. It does not beat logistic regression. Reported rather than deleted,
because a panel that failed its evaluation is evidence about the method.

## Known limitations

- **Every cohort is case-control, not a screening series.** See the precision table above.
- **The breast panel contradicts the project's own schema rule.** Its inputs are nuclear morphology
  from a fine needle aspirate, which requires a biopsy that already happened. It interprets a
  biopsy, it does not screen for one.
- **The liver panel detects liver disease, not liver cancer.** Liver disease is roughly 300 times
  more common and is the dominant precursor to hepatocellular carcinoma, so this is a useful thing
  to detect, but it is not a cancer claim.
- **The pancreatic panel has no external test and is the weakest claim here.** No public cohort
  shares its feature set. Its 0.969 should be discounted by something like the 0.06 to 0.46
  generalisation loss measured on liver, and at real incidence it flags 525 people per true case.
  PROTOCOL.md shows validating it prospectively would need roughly 690,000 participants. There is a
  reasonable argument it should be withdrawn on the same grounds as prostate.
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
