# Pre-registration: a hypothesis-free discovery screen for free cancer-risk information

Written and committed on 18 October 2026, before the full NHIS adult questionnaire has been downloaded or
screened. Known going in: age, sex and smoking carry nearly all the free signal (earlier studies here); BMI,
schooling, marital status, self-rated health and activity add about 0.001 AUC. The question is whether
**anything else the survey asks** (several hundred items) adds to cancer-death risk beyond age, sex, smoking and
BMI, replicably, and specifically for cancer.

## Data

NHIS Sample Adult files 1997-2009 (the same cohort and linked mortality as `docs/CANCER_AGE_PREREG.md`): adults
35-84, no earlier cancer, known smoking and BMI. Outcome: death from cancer within ten years. Every variable that
appears in the NCHS SAS layout of the file is a candidate, except those excluded below.

Excluded in advance: identifiers, weights, dates, interviewer and survey-design fields; anything about cancer
diagnosis, cancer treatment or cancer screening results (leakage or reverse causation); mortality fields.
Variables are used in the years where they exist with the same name and label; a variable needs at least two
interview years in each of the discovery and replication sets.

## Design

- Discovery set: interview years 1997-2002 (only years in which the variable exists). Replication set: 2005-2009.
  Nothing from the replication set is looked at before the discovery hit list is frozen and written to disk.
- Baseline model: logistic regression with age (cubic spline), sex, smoking status, cigarettes a day, years since
  quitting and BMI. Each candidate variable is added on its own. Codes with few distinct values (12 or fewer) are
  treated as categories, missing and "unknown" codes included as their own category (so no coding guess is
  needed); other variables are cut into five quantile groups plus a missing group.
- Test: likelihood-ratio test of the variable block. Discovery hit: false discovery rate q < 0.05 across all
  variables screened.
- Replication: a hit replicates if, in the replication set, the same block has p < 0.05 after Bonferroni over the
  hits, with the same direction of effect for its main contrast (the discovery's largest log-odds contrast).
- Specificity (is it cancer, or just frailty?): for every replicated hit, the AUC gain over the baseline for cancer
  death must be at least half its AUC gain for death from other causes.
- Usefulness: a hit counts as a **discovery** only if, added to the baseline, it raises replication-set AUC for cancer
  death by at least 0.005 (95% bootstrap interval above zero) **and** raises coverage of cancer deaths at 20%
  invited by at least 1 point (interval above zero).

## Second screen: who the free model fails

The free model under-predicted cancer death in Black adults and over-predicted it in Hispanic adults, and ranked them
worse (`docs/cancer_age_paper/`). In each of non-Hispanic Black, Hispanic and other adults, the same scan is run
with the baseline refitted in that group, using the same discovery/replication split and the same rules. A hit here
that raises AUC in that group by at least 0.01 (interval above zero) in replication counts as an equity discovery.

## What would be reported, and what would not

All candidates screened, the number passing each stage, and every miss. If nothing passes all stages, the finding
is that the free-information ceiling is firm, which is itself the result: a negative with 300+ variables screened and
a replication rule fixed in advance. No variable is rescued after the fact, and the thresholds above do not move.

## Limits stated in advance

Self-reported variables only. Death from any cancer, not site. Multiple variables that measure the same thing will
all pass together. Associations are not causes and the screen finds predictors, not mechanisms. Variables present
only in some years have less power. This is one survey; an NHANES check is added where the same item exists.
