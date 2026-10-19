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

## Addendum, 18 October 2026 (before any screening result exists)

Two decisions made while writing the code, recorded here so they are not post hoc:

- Interview-year indicators are added to the baseline model, so an item asked in only some years is not confounded
  with period trends in cancer death.
- Candidate items must have the same label in at least two discovery and two replication years; years where the label
  differs are dropped for that item. Labels are read from the NCHS SAS programs. Categories with fewer than 50
  respondents are pooled. Models use a very light ridge penalty (C = 10,000) so that sparse categories cannot produce
  infinite coefficients; likelihood-ratio tests use the penalised fits.


## Addendum 2, 18 October 2026 (before any screening result exists)

Written after the full questionnaire was downloaded and the code was run on real data for the first time. What had been
looked at before this addendum: cohort sizes and cancer-death counts by year and by race group (to match the age paper);
the screen run on the first three items by name with only size and convergence printed, no test result; and the screen run
on **shuffled copies** of items, which have no real link to cancer, to check the test is calibrated. No test result for a
real item has been seen. Everything below fixes the code so it does what this document says, or records a decision that
was forced by the data. **None of it changes the baseline model, the discovery and replication years, the false discovery
rate, the Bonferroni step, the same-direction rule, the cancer-specificity rule or the usefulness thresholds.**

1. **Equity bar.** The text above says an equity hit must raise AUC in its group by at least 0.01 with the interval above
   zero. The code applied the all-adults rule (0.005 AUC and 1 point of coverage). It now uses 0.01 AUC with the interval
   above zero for the equity screens. Replication and cancer-specificity still apply. Coverage at 20% invited is reported but
   not required, because the registered text names only the AUC criterion.
2. **No silent drops at replication.** If the replication years had no one in a category that had been pooled in discovery, the
   item was skipped with no record. The replication block now has the discovery layout, the test counts only columns that
   vary in the replication years, and a hit that cannot be tested is listed with the reason.
3. **Race and ethnicity were missing in 1997-2002.** The code chose one race variable name for every year, but NCHS uses a
   different name in different years (RACEREC, RACER_P, RACERP_I, RACERPI2; HISPAN_P, HISPANCR, HISPAN_I). The discovery years
   came out with no White or Black adults. The code now takes the first variable that has a value for each person. 2005-2009 is
   unchanged and matches the age paper exactly (White 45,815 adults and 1,384 cancer deaths; Black 11,431 and 397; Hispanic
   11,379 and 194; other 4,543 and 97). As in the age paper, adults whose Hispanic subgroup code is 0 are left unclassified
   (135 adults in 2005-2009).
4. **Race and ethnicity are not candidate items.** Race is a stratifier, never a feature (project rule), so every race and
   Hispanic-origin item is excluded by name and by label. Items such as birthplace or language, if any, are kept; a hit that
   could be an ethnicity proxy will be flagged in the report.
5. **Solver.** On shuffled items the likelihood-ratio test was miscalibrated: null p-values averaged 0.61 (should be 0.50), 2.8%
   fell below 0.05 (should be 5%) and 14% of tests gave an impossible likelihood-ratio of zero, because the default lbfgs fits
   stopped before converging. Fits now use the exact Newton-Cholesky solver at a tight tolerance. Same shuffles, new solver:
   mean p 0.47, 5.7% below 0.05, no zeros, uniformity test p = 0.10 (316 null tests); a tight-tolerance lbfgs gave identical
   numbers, so this is the optimum. Same model, same ridge (C = 10,000).
6. **Size floors (not registered, only reduce power).** The code skips an item that has fewer than 5,000 people or fewer than
   100 cancer deaths in its discovery years, or fewer than two categories with 50 people. Skipped items are listed with their
   reasons. Counts, from sizes only: all adults, 79 of 79 candidates clear the floors; non-Hispanic Black, 69 of 79; Hispanic,
   69 of 79; **other, 0 of 79** (3,840 adults and 107 cancer deaths in the discovery years). The "other" equity screen will
   therefore be reported as not screenable rather than run.
7. **How many items.** The text above expected "several hundred" and "300+" items. The registered eligibility rule (a stable
   label in at least two discovery and two replication years) leaves **79**. Of 2,800 columns: 288 excluded by name, 508 by
   label (cancer, screening, baseline items, race), 178 with no label, 1,636 asked in fewer than two discovery or two
   replication years, 111 asked in enough years but with changed wording (for example "had" became "have"), and 79 candidates.
   The 111 are listed in the excluded file and are **not** screened. The result will be stated for 79 items, not for 300.
8. **Power of the equity screens, stated in advance.** Replication has 397 cancer deaths in Black adults and 194 in Hispanic
   adults. An AUC gain of 0.01 with an interval above zero is hard to show at that size, so a miss there means "not shown", not
   "absent".
9. **Freeze.** The run is split into `discover` and `replicate`. `discover` writes the hit list and its SHA-256 and stops. The
   hit list and hash are committed before `replicate` is run, and `replicate` refuses to run if the file no longer matches.
10. **Reporting.** The output now has the count at every stage, the list of excluded columns with reasons, and the list of
    skipped items with reasons. Commands are `python experiments/discovery_screen.py [equity 1|2|3] discover|replicate`.
11. **Speed only.** Item columns are passed to workers one at a time and eight workers are used. The screen takes minutes.
