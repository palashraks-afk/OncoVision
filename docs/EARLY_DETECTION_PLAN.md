# Picking cancer out of a healthy person with no prompt: what is known, and what a student project can do

The honest answer to "how do we achieve this" is that nobody has fully done it. This
records what the best attempts reached, why this project's routine-lab approach fell
short, and the route that is reachable. Figures below come from search summaries of the
published sources and must be checked against the papers before they are cited.

## What the best attempts have reached

| Approach | Best result found | What it needed | Limit |
|---|---|---|---|
| **Special blood test, cell-free DNA methylation** (Galleri, NHS-Galleri randomised trial, 2026) | Sensitivity 54.7% for 12 prespecified cancers and 30.7% across all cancers; specificity 99.55%; positive predictive value about 52%. By cancer: lung 63.8%, colorectal 53.3%, breast 19.1%, prostate 10.7% | A purpose-built assay and a trial of about 140,000 people | The pre-specified primary endpoint, fewer stage III and IV diagnoses, was not statistically significant; the company reports a reduction in stage IV. Whether it saves lives is unproven. About half of positive results are false. |
| **The whole health record over time** (Placido et al., Nature Medicine 2023, pancreatic cancer from disease-code sequences) | Danish records AUROC 0.88 for cancer within 36 months, 0.83 after excluding the last 3 months; applied unchanged to US veterans 0.71, retrained 0.78; the 1,000 highest-risk patients over 50 had about 59 times the average risk | Records on 6 million Danish and 3 million US patients | Falls sharply on a new population; it ranks risk, it does not diagnose |
| **Routine blood plus clinical factors** (UK Biobank, colorectal) | C-index 0.67, AUC 0.76 with boosted trees | A large biobank with pre-diagnostic blood | Modest; routine labs add little |
| **Clinical and polygenic risk, eight cancers** (UK Biobank) | AUC 0.68 to 0.83 | Genotyping plus clinical data | Targets who to screen, not who has cancer |

**What this says about the idea.** The overall-picture idea is the right one, and the
signal that exists lives in the whole clinical history across time and in purpose-built
assays, not in a single set of routine labs. This project's own tests agree: routine labs,
and even 337 survey variables, found nothing that survives, while the report-based panels
for breast and prostate did.

## What a useful early-detection tool would have to show

1. **A pre-diagnostic cohort.** Samples and records from before anyone knew, with a later
   diagnosis date. Every dataset this project reached lacks that.
2. **Specificity near 99.5%.** Cancer is rare, so a 5% false-positive rate buries the
   true cases. This is the arithmetic that sinks every routine-lab approach.
3. **A stronger baseline than chance.** Age and sex, then smoking and family history,
   then a published model such as QCancer, so a gain is a real gain.
4. **External validation.** The pancreatic model fell from 0.88 to 0.71 on a new
   population. A result inside one cohort is not evidence.
5. **Proof of benefit,** which only a trial gives.

Items 1 to 4 are within reach of a pre-registered study on existing biobank data. Item 5
is not.

## The route that fits

**Risk stratification from the overall picture, tested on the right data, claiming only
what it shows.** The goal is not "this person has cancer". It is "this person's overall
record places them in a group with several times the average risk, so the screening they
are already eligible for, or a conversation with their doctor, matters more for them".

Success in that sense can be measured: how much more cancer is found in the top 1% and 5%
of predicted risk, and how many people would have to be followed per cancer found.

### Pre-registered protocol (to be fixed before any data is opened)

- **Data.** UK Biobank (about 500,000 adults, baseline blood chemistry and blood count,
  questionnaires, lifestyle, family history, and linked cancer registry) as the training
  and internal set; All of Us or HRS or CHARLS as the external set. All need an application
  and a named investigator.
- **Outcome.** First incident cancer, by site and any site, diagnosed at least 12 months
  after baseline, so that cancer already present at baseline is not mistaken for early
  signal. Anyone with prior cancer is removed. Everybody else stays in, with deaths from
  other causes handled as a competing risk and not dropped.
- **Inputs, nested.** Age and sex; plus smoking, alcohol, BMI and family history; plus
  routine blood chemistry and blood count; plus medications, conditions and
  symptoms from the record; plus change from the person's own earlier values where
  repeat measures exist.
- **Comparators.** Age and sex, then a published risk model for each cancer (QCancer or the
  site-specific equivalent), each fitted the same way, with the better of regression and an
  ensemble taken for every comparator.
- **Primary result.** For each cancer, the gain in discrimination over the strongest
  comparator with a 95% interval, and the enrichment of cases in the top 1% and 5% of
  risk with the number needed to follow per cancer found.
- **Bars, set now.** A tool is called useful for a cancer only if its gain over the best
  comparator has an interval excluding zero in the internal set, is repeated in the
  external set, and the top 5% captures materially more cancers than the same share
  chosen by the best comparator.
- **Reported whatever the answer.** Including every cancer that fails. With a dozen sites
  tested, a lone positive is a lead and not a finding.

### What this would produce

An honest, externally validated answer to the question this project set out to ask, on
data capable of answering it. The likely result, from the published numbers above, is
modest gains for a few sites and none for most. That is a publishable result and it is
the real state of the field.

## What a student project cannot do

It cannot build a cell-free DNA assay, run a screening trial, or prove that detection
saves lives. It can build the interpretation layer for the people who take such tests: a
positive result there is wrong about as often as it is right, which is exactly the kind of
result a patient needs explained honestly.

## What needs people

1. A named investigator and an institution, for UK Biobank and All of Us applications.
2. Dr. Chavan, or a collaborator in adult oncology or epidemiology, to advise on the
   clinical questions and sign the applications.
3. Registration for CHARLS, which is free and is the quickest first step.
