# Eight problems, the candidate solutions, and what happened to each

The request was a hundred solutions per problem, screened. A list that long would
be padded with ideas nobody could defend, so this is every candidate that could be
defended, with a verdict from evidence and not from opinion. Each has one of:

- **WORKED**: tested against a bar written before the run, and it passed
- **FAILED**: tested the same way, and it did not pass. Kept, because a failure is a result
- **READY**: built and tested on what is available, waiting on data only a person can request
- **NEEDS YOU**: only a human with a name, a login or an institution can do it
- **REJECTED**: not worth running, with the reason

Nothing in the NEEDS YOU column has been submitted.

---

## 0. The round's biggest finding, because it changes the picture

The five-year cancer-death panel that shipped last round **was not a cancer panel**
and has been withdrawn.

Its cohort kept only people who died of cancer within five years or lived past
five, and dropped the 1,510 who died of something else in the window. So every
early death left in the file was a cancer death, and a model could score well by
recognising who looked close to dying. The external cohort (NHANES III) was built
the same way, so it could not catch this. A test that shares a flaw with the
thing it tests proves nothing about the flaw.

With those people put back, on NHANES III:

| The shipped model scored against | AUC |
|---|---|
| cancer deaths | 0.875 |
| deaths from other causes | 0.873 |
| death from any cause | 0.874 |

Among people who died, where age and sex cannot separate the causes, it reached
0.574 against 0.553 for age and sex (95% CI on the gain −0.023 to +0.068). The bar,
written before the cohorts finished downloading, needed both that and the honest
cancer outcome. The first held and the second did not, so it is withdrawn. What
the blood work does predict is **death from any cause**, +0.025 over age and sex.
That is real, and it is not a cancer finding.

This is the same lesson as the earlier reversals, one level down: not "which model"
or "which baseline" but **which people the cohort lets in**. Dropping anyone based
on what happens after the index date selects on the future.

---

## 1. We don't have the right training data

Needed: blood drawn BEFORE anyone knew, with a later diagnosis date.

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | NHANES linked to the National Death Index | **FAILED**, as a cancer source | Built the panel; see section 0. Death endpoint, and the file's selection flaw. |
| 2 | Retrospective NHANES "ever had cancer" | **FAILED** | The three withdrawn panels. Blood after diagnosis. |
| 3 | The rest of the checkup in NHANES (HbA1c, lipids, urine, BP, waist) | **FAILED** | Bowel −0.031; general +0.016 inside the survey, −0.026 on NHANES III. |
| 4 | **CHARLS** (China, 45+) | **NEEDS YOU, and the best lead** | Blood from the same people in 2011 and 2015 (confirmed), cancer asked every later wave. Two draws four years apart is the "your own change" test. Registration only. |
| 5 | **MIMIC-IV** | **READY** | Repeat counts with timestamps and diagnosis dates. Pipeline built and tested on the open demo. Needs CITI plus PhysioNet. |
| 6 | HRS 2016 Venous Blood Study | **NEEDS YOU** | Sensitive Health Data order form. Blood 2016, cancers 2018–2022. |
| 7 | ELSA (England, 50+) | **NEEDS YOU** | Nurse-visit blood every four years; free after UK Data Service registration. Cancer variable not confirmed. |
| 8 | NHANES–Medicare linkage, NCHS Research Data Center | **NEEDS YOU** | Would give NHANES a diagnosis endpoint. Restricted-use; needs an RDC proposal. |
| 9 | PLCO via NCI CDAS | **NEEDS DR. CHAVAN** | Only screening population with CA-125 and PSA. Draft written. |
| 10 | UK Biobank | **NEEDS YOU** | Application and fee. Would make liver cancer possible. |
| 11 | All of Us | **NEEDS AN INSTITUTION** | Registered-tier access through a data use agreement. |
| 12 | CHOC registry linked to labs | **NEEDS DR. CHAVAN + IRB** | The strongest design and the slowest. |
| 13 | Healthy people pulled from the web | **REJECTED** | Cases from one source and controls from another teach the model which source it came from. 33,000 healthy controls already exist; controls were never the bottleneck. |
| 14 | Kaggle / Mendeley "cancer blood test" files | **REJECTED** | No provenance, mostly case-control or synthetic. Zenodo, Dryad, Figshare and Dataverse searched: no pre-diagnostic labs with outcomes. |
| 15 | Synthetic data | **REJECTED** | Cannot show real signal. |
| 16 | TCGA / GEO | **REJECTED** | Tumour tissue, not routine labs. |
| 17 | German hospital CBC file (523,844 samples) | **REJECTED** for outcomes | Checked: no cancer labels. Useful only for lab-variation work. |
| 18 | eICU | **REJECTED** | ICU patients, same CITI requirement, no screening population. |
| 19 | Literature extraction of case reports | **REJECTED** | No controls. |

---

## 2. One blood draw carries very little signal

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | More routine values | **FAILED** | See 1.3 above. |
| 2 | Richer models (tree ensembles) | **FAILED** | Logistic regression won every model selection. |
| 3 | Pattern matching: nearest neighbours, centroid, unusualness | **FAILED** | Best 0.844 against 0.875. Unusualness as an extra input +0.0008, below the margin. |
| 4 | Keep the people the cohort dropped (fix the target) | **WORKED**, as a diagnosis | It is what exposed the panel. |
| 5 | Search for cancer-specific signal among decedents, 5/10/15 years | **A LEAD, NOT A FINDING** | 6 pairs; positive inside the survey in all 6; on NHANES III positive in 5 of 6, interval excluding zero in 1 (10 years vs other deaths, +0.035, CI +0.007 to +0.062). One of six is a lead to confirm. |
| 6 | What that lead is made of | **FAILED**, as a cancer signal | Stable, replicated drivers point AWAY from cancer: BUN and glucose, i.e. kidney disease and diabetes. Haemoglobin flips sign between cohorts. It is "headed for a kidney or metabolic death", with cancer left over. |
| 7 | Add BMI, smoking, alcohol | **PARTIAL** | Larger internal gains, but NHANES III lacks them, so it cannot be confirmed. |
| 8 | Site-specific outcomes | **NOT POSSIBLE** with public files | The public linked-mortality file has one "malignant neoplasms" code. Site needs restricted files. |
| 9 | Trajectory features | **READY** | Section 3. |
| 10 | Tumour markers where they are ordered | **WORKED**, narrowly | Ovarian (CA-125, HE4), pancreatic (CA 19-9), prostate (PSA). Case-control, so not screening. |
| 11 | Mammogram report plus history | **WORKED** | +0.014 over age plus the density line already on the report. |
| 12 | Derived ratios (neutrophil/lymphocyte, AST/ALT) | **NOT TESTED** | Cheap. Not pulled because the cause-specific result says there is little left to find in single-draw labs. |
| 13 | Cell-free DNA, methylation, proteomics | **REJECTED** | Not routine labs; not on the reports people already have. |
| 14 | Polygenic risk | **REJECTED** | Not on a lab report. |

---

## 3. The most promising fix is untested: your own change over time

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | Trajectory pipeline | **READY** | `fetch_mimic_trajectory.py` and `experiments/trajectory_vs_snapshot.py`. Drops survivors, refuses labs within 90 days of diagnosis, refuses to report below the event floor. Runs unchanged on the full data. |
| 2 | CHARLS two-draw change | **NEEDS YOU** | Free, registration only. Not built, because the file layout should be read from the real download rather than guessed. |
| 3 | ELSA, every four years | **NEEDS YOU** | Same. |
| 4 | Commercial precedent | **EVIDENCE** | ColonFlag reports AUC 0.74–0.82, validated in Israel, the UK and the US. Those papers do not report the age-and-sex baseline, and this project's own bowel work found age and sex alone at 0.83. Whether the trajectory beats age is the open question. |
| 5 | Normalise each value by the report's own printed reference range | **READY, untested** | MIMIC `labevents` carries reference ranges per result, which is the standard way clinicians compare across labs. It would also address problem 5. |

---

## 4. The one working blood panel may not measure cancer

**Answered, and it did not.** Section 0. Withdrawn, with the bar written first.

| # | Candidate | Verdict |
|---|---|---|
| 1 | Test against all-cause death | **DONE**: 0.874, no worse than cancer. |
| 2 | Test among decedents only | **DONE**: not shown. |
| 3 | Rebuild keeping other deaths | **DONE**: fixes the cohort, removes the signal. |
| 4 | Relabel as an all-cause mortality score | **REJECTED**: real signal (+0.025), but it is not the product and existing mortality scores already do it. |
| 5 | Competing-risk survival model | **COVERED**: the decedent analysis answers the same question more directly. |

---

## 5. Real labs differ from research labs

| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| 1 | Stress-test the shipped models | **DONE** | Liver loses about 0.02 AUC at 10% bias and 0.03 at 20%; AST is the weakest value (−0.026 at ±20%). The mortality panel barely moved, which was another sign it mostly read age. |
| 2 | Train across simulated labs (random per-batch bias) | **WORKED on liver, NOT YET SHIPPED** | On NHANES 2021–2023: worst case at 10% bias +0.021, clean AUC +0.011. On the mortality cohort: no gain, and that panel is now withdrawn, so the original two-cohort rule can no longer be met. |
| 3 | Confirm on liver's other unseen cycle, same thresholds | **WORKED** | NHANES 2017–2018: worst case +0.012 (need ≥ 0.010), clean AUC +0.007. A control trained on four plain copies is WORSE (0.697 against 0.703), so the gain is the bias and not the duplication. |
| 3a | Ship it | **NEXT, a deliberate follow-up** | Augmenting before cross-validation would put copies of one patient in both training and calibration folds and inflate every internal number. It needs a fold-internal wrapper that survives pickling to the backend, then a re-run of liver's evidence chain (temporal validation, fresh cycle, fairness, stability, demographic gain). About +0.01 AUC and about 0.02 under lab bias, so worth doing, not worth rushing. |
| 4 | Clip impossible values | **ALREADY DONE** | `feature_ranges`; refuses what a living patient cannot have. |
| 5 | Unit-mixup guard | **ALREADY DONE** | Found on the German cohort. |
| 6 | Reference-range normalisation | **READY** | See 3.5. |

---

## 6. Cancer is rare, so a good test still flags mostly healthy people

| # | Candidate | Verdict |
|---|---|---|
| 1 | Show flagged-people-per-case on every card | **ALREADY DONE** |
| 2 | Restrict to an enriched population (older, smokers) | **Moot** for the withdrawn panel; arithmetic helps only as much as the base rate rises |
| 3 | Two-stage testing | **Needs a second test that exists** |
| 4 | Lower the threshold | **FAILED**: sensitivity collapses before precision recovers |

This one is arithmetic, not a modelling flaw. The fix is to claim less, and the cards do.

---

## 7. No patient has used it and been followed

**NEEDS DR. CHAVAN.** `docs/PROSPECTIVE_STUDY_PROTOCOL.md` is drafted. The nearest
real step is a retrospective chart review through CHOC's IRB, which gives labs
before diagnosis in real patients without recruiting anyone.

## 8. Saving money is unproven

| # | Candidate | Verdict |
|---|---|---|
| 1 | Triage cost models for every panel | **DONE**: only supplemental MRI for dense breasts pays, and modestly at Medicare prices |
| 2 | Prostate biopsy triage | **FAILED**: biopsying everyone beats both the panel and PI-RADS ≥ 3 in all 20 price combinations |
| 3 | Price "which screening to do first" | **Cannot be priced** without a confirmatory procedure and a diagnosis endpoint |

---

## What this round actually changed

1. Found and fixed a selection flaw in the one bloodwork cohort, and withdrew the
   panel it had produced.
2. Measured how fragile the live lab-based panel is to lab-to-lab bias, and found a
   training fix that worked on both of liver's unseen cycles and is not yet shipped,
   for the reason in 5.3a.
3. Searched for cancer-specific signal in routine labs and found a lead whose
   makeup argues against it.
4. Found CHARLS, the cheapest route to the trajectory test.

The next step with the largest payoff is the same as before and still belongs to a
person: **register for CHARLS and take the CITI course.**
