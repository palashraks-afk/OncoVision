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

| Panel | Test AUC | 95% CI | Sens. | Spec. | Logistic baseline | Shipped model | Test n |
|---|---|---|---|---|---|---|---|
| General | 0.966 | 0.937 to 0.989 | 0.910 | 0.984 | 0.917 | ensemble | 300 |
| Breast | 0.972 | 0.940 to 0.994 | 0.786 | 0.958 | 0.964 | ensemble | 114 |
| Pancreatic | 0.969 | 0.940 to 0.992 | 0.731 | 0.947 | 0.968 | **logistic** | 120 |
| Liver | 0.785 | 0.697 to 0.865 | 0.928 | 0.206 | 0.831 | **logistic** | 117 |

The pipeline picks whichever model wins on cross-validated AUC inside the training data. Two panels
ship plain logistic regression because it beat the ensemble. Assuming the fancy model wins is
exactly the kind of thing that goes unchecked.

### External validation

Every held-out number above still shares a hospital, assay machines and population
with its training data. Three of the four panels now have a genuine outside test.

| Panel | External cohort | Internal | External | Drop |
|---|---|---|---|---|
| General | NHANES 2017-2018, 5,173 US adults | 0.966 | **0.596** | **0.370** |
| Liver | Germany, 589 patients (trained on India) | 0.785 | **0.623** | 0.162 |
| Liver | India, 583 patients (trained on Germany) | 0.995 | **0.698** | **0.297** |
| Breast | WPBC, 198 independent cancer patients | 0.915 sens | **0.894 sens** | 0.021 |
| Pancreatic | none exists publicly | 0.969 | not tested | unknown |

**The general panel is the cautionary one.** It looked like the best model in the
project at 0.966 and scores 0.596 on a nationally representative US sample, which
is barely better than a coin flip. NHANES matters because it is not case-control:
10.3% prevalence, sampled from the public rather than from people already being
investigated.

**The breast panel is the one that holds.** 0.894 external sensitivity against
0.915 internal, on 198 independent confirmed cancer patients from a separate
Wisconsin cohort. No AUC is available because that cohort has no benign class,
so this is a partial external test and is labelled as one.

**Liver was tested both directions.** A model trained on German patients scores
0.995 on its own held-out data and 0.698 on Indian patients. Same model, same
task, 0.297 lost purely to the change of population. The German cohort reports in
SI units, so albumin, total protein and bilirubin had to be converted before the
two were comparable at all.

Logistic regression scored 0.736 going India to Germany against the ensemble's
0.623, and 0.618 on NHANES against the ensemble's 0.596. The simpler model
transferred better in both cases.

### Subgroup accuracy, measured rather than disclaimed

NHANES is the only cohort here carrying race and ethnicity.

| Group | n | Prevalence | AUC | 95% CI |
|---|---|---|---|---|
| Non-Hispanic Asian | 750 | 4.1% | 0.667 | 0.560 to 0.767 |
| Mexican American | 685 | 6.7% | 0.625 | 0.549 to 0.698 |
| Non-Hispanic White | 1,777 | 17.4% | 0.578 | 0.542 to 0.614 |
| Non-Hispanic Black | 1,219 | 7.4% | 0.571 | 0.511 to 0.631 |
| Other Hispanic | 483 | 6.8% | 0.555 | 0.452 to 0.661 |
| Other or multiracial | 259 | 10.0% | 0.563 | 0.453 to 0.668 |

Every group sits between 0.55 and 0.67. The model is weak across all of them
rather than unfair between them, which is a different problem and a real one.

### The number that actually matters

Cohorts here are enriched for disease. Projecting measured sensitivity and specificity onto real
population prevalence gives the precision a user would experience.

| Panel | Cohort positive | Population prevalence | PPV there | People flagged per true case |
|---|---|---|---|---|
| General | 37% | 0.4507% | **20.6%** | 4.9 |
| Liver | 71% | 3.1% | **3.6%** | 27.8 |
| Breast | 37% | 0.1325% | **2.44%** | 41 |
| Pancreatic | 22% | 0.0139% | **0.19%** | 525 |

Used as a population screen today, the pancreatic panel would flag about 525 people for every one
who has the disease. No AUC figure changes that.

### A panel that was withdrawn

**Prostate is trained, measured, and not served.** Test AUC 0.786 with a 95% CI of 0.505 to 0.99, so
the lower bound sits on chance. Specificity 0.571 with a CI of 0.167 to 1.0, an interval that
carries no information because the test split is 20 records. It does not meaningfully beat logistic
regression (0.769). 97 records and two usable features cannot support a clinical claim. It is
reported rather than deleted, because a panel that failed its evaluation is evidence about the
method.

## Known limitations

- **Every cohort is case-control, not a screening series.** See the precision table above.
- **The breast panel contradicts the project's own schema rule.** Its inputs are nuclear morphology
  from a fine needle aspirate, which requires a biopsy that already happened. It interprets a
  biopsy, it does not screen for one.
- **The liver panel detects liver disease, not liver cancer.** It replaced a synthetic cohort with
  583 real patients. Its specificity is 0.206, meaning it flags most people, so it is useful for
  ruling out rather than ruling in.
- **The pancreatic panel has no external test.** No public cohort shares its feature set. Its 0.969
  should be discounted by something like the 0.16 to 0.37 generalisation loss measured elsewhere.
- **No prospective test and no IRB.** No real patient report has been followed to an outcome. A
  submittable study protocol with pre-specified statistics is drafted in [PROTOCOL.md](PROTOCOL.md).
  Its sample-size table shows the pancreatic panel would need roughly 690,000 participants to
  validate at real incidence, which is why that panel carries the strongest caution here.
- **The ensemble is within noise of logistic regression** on breast and pancreatic.
- **No race or ethnicity in any cohort**, so accuracy across those groups is unmeasured rather than
  acceptable. AUC by sex and age band is in `EVALUATION.md`.

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

# cross-country external validation of the liver panel
python external_validation.py

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
