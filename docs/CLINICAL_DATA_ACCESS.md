# Getting data with real symptoms and cancer diagnoses

This is the main thing a survey cannot give: people's symptoms and labs recorded **before** a
cancer diagnosis, with the diagnosis afterwards. It is where a stronger model could be built,
and where the navigator's rules could be tested on what they were written for.

I cannot get access for you. Each of these needs a person, an institution, or both. The
details below are my understanding and **have not been checked against each programme's current
rules**; confirm on the official site before applying.

| Source | What it has | Who can apply (to confirm) | Effort |
|---|---|---|---|
| **MIMIC-IV** (PhysioNet) | ICU and hospital records with labs and diagnosis codes, including cancer. Not a symptom-first population. | Credentialed researchers who finish CITI training and sign the data use agreement | Low (days to weeks). Already planned in `docs/NEXT_STEPS_FOR_RAHUL.md` |
| **PLCO** (NCI CDAS) | Large screening trial with follow-up and cancer outcomes, blood tests in a subset, and questionnaires. | Researchers, through a project proposal | Medium. A draft is in `docs/PLCO_CDAS_PROPOSAL.md` |
| **All of Us** (NIH) | Electronic health records, surveys and labs for many people. | Researchers at registered institutions | Medium to high, needs an institution |
| **UK Biobank** | Linked primary-care and cancer-registry records, blood tests. | Approved researchers, with a fee | High |
| **CPRD / QResearch** | UK primary-care records, the data NG12 and QCancer were built on. | Approved researchers, with a fee | High |
| **SEER-Medicare** | Cancer registry linked to claims. No labs or symptoms. | Researchers | Medium. Probably the wrong data |

## Recommended order

1. **Now, free, possible alone:** MIMIC-IV, to test the pipeline on real hospital data with
   diagnoses. Limit: it is hospital data, so a different population from the people the
   navigator is for. Start with the CITI course.
2. **With Dr. Chavan or a collaborator at an institution:** PLCO or All of Us, with a short
   proposal that states the question in advance.
3. **Later, with a group:** UK data, to test the rules on the population NG12 was written for.

## What to write in any application

Use the pre-registrations in `docs/` as the model: the question, the bars, the baseline you
must beat (age, sex, smoking, BMI), the out-of-era test, and what a missed bar will mean.
Say plainly that earlier survey work found no gain over the baseline, because a reviewer who
knows the field will ask.

## What not to do

Do not buy or scrape health data. Do not use patient records from a hospital without that
hospital's written approval. Do not rely on a site that offers "patient data" without a data
use agreement.
