# Oncovision

A full stack platform that reads standard patient lab reports plus information about the patient
and returns multi-cancer risk assessments in real time.

Built as a mentored research project under the guidance of a clinical oncologist at UCI CHOC.

**[Live app](https://oncovisionai.vercel.app)** · **[Methodology](PROJECT.md)** · **[Full evaluation](EVALUATION.md)**

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

Four calibrated ensembles score the combination, and the interface reports what drove each score,
how accurate that model is, and what the score is worth at real population prevalence.

---

## Honest numbers

All measured on a 20% test split cut **before** any model was fitted, never used for training,
model selection, or calibration. Intervals are bootstrap percentile intervals over 2,000 resamples.
Reproduce with `python evaluate.py`.

| Panel | Test AUC | 95% CI | Sens. | Spec. | Logistic baseline | Test n |
|---|---|---|---|---|---|---|
| General | 0.966 | 0.937 to 0.989 | 0.910 | 0.984 | 0.917 | 300 |
| Breast | 0.972 | 0.940 to 0.994 | 0.786 | 0.958 | 0.964 | 114 |
| Liver | 0.970 | 0.958 to 0.979 | 0.798 | 0.986 | 0.942 | 1000 |
| Pancreatic | 0.969 | 0.939 to 0.991 | 0.731 | 0.947 | 0.968 | 120 |

### The number that actually matters

Every cohort here is case-control and enriched for disease. Projecting the measured sensitivity
and specificity onto real SEER incidence gives the precision a screening user would experience.

| Panel | Cohort positive | Real incidence | PPV there | People flagged per true case |
|---|---|---|---|---|
| General | 37% | 0.4507% | **20.6%** | 4.9 |
| Breast | 37% | 0.1325% | **2.44%** | 41 |
| Liver | 22% | 0.0095% | **0.54%** | 187 |
| Pancreatic | 22% | 0.0139% | **0.19%** | 525 |

Used as a population screen today, the pancreatic panel would flag about 525 people for every one
who has the disease. No AUC figure changes that. What these models can reasonably do is rank and
explain values for someone who already has a reason to be asking.

### A panel that was withdrawn

**Prostate is trained, measured, and not served.** Test AUC 0.786 with a 95% CI of 0.505 to 0.99,
so the lower bound sits on chance. Specificity 0.571 with a CI of 0.167 to 1.0, an interval that
carries no information because the test split is 20 records. It does not meaningfully beat logistic
regression (0.769). 97 records and two usable features cannot support a clinical claim. It is
reported rather than deleted, because a panel that failed its evaluation is evidence about
the method.

---

## Known limitations

- **Every cohort is case-control, not a screening series.** See the precision table above.
- **The breast panel contradicts the project's own schema rule.** Its inputs are nuclear morphology
  from a fine needle aspirate, which requires a biopsy that already happened. It interprets a
  biopsy, it does not screen for one.
- **The liver cohort is synthetic.** Largest dataset, joint-highest AUC, generated rather than
  observed. That AUC describes a generator.
- **No external validation.** Every number comes from a held-out split of the same cohort the model
  trained on. This is the single largest gap.
- **No prospective test and no IRB.** No real patient report has been followed to an outcome.
- **The ensemble is within noise of logistic regression** on breast and pancreatic.
- **No race or ethnicity in any cohort**, so accuracy across those groups is unmeasured rather than
  acceptable. AUC by sex and age band is in `EVALUATION.md`.

---

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 App Router, React 19, TypeScript, Tailwind v4, Recharts, Vercel |
| Backend | FastAPI, Uvicorn, Pydantic, pdfplumber, pandas, Render |
| Models | XGBoost + Extra Trees soft-voting ensembles, isotonic calibrated, SHAP for attribution |

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

# evaluate first, it produces evaluation.json which training embeds
python evaluate.py

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
