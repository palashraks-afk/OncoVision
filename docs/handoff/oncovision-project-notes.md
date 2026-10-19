---
name: oncovision-project
description: "Rahul's Oncovision cancer-risk platform — its evidence rules, panel taxonomy, and the standards a change has to meet"
metadata: 
  node_type: memory
  type: project
  originSessionId: c1d0fd06-17f6-4390-b854-7fe39d1b7da1
  modified: 2026-10-13T03:15:25.853Z
---

Oncovision (`C:\Users\Rahul\Desktop\oncovision`, GitHub `palashraks-afk/OncoVision`): a
full-stack platform that reads standard patient lab reports and returns multi-cancer risk
assessments. A formal mentored research project under a clinical oncologist at UCI CHOC,
so its claims have to survive scrutiny, not just run.

Rules the project holds itself to. These are not style preferences — breaking one has
shipped a real bug at least once:

- **Measure, don't assume.** Every claim gets an experiment in `experiments/`, including
  the ones that come back negative and change nothing. Set the adoption bar BEFORE running,
  and do not move it afterwards to let a favourite through.
- **Repeated paired CV on identical folds is the arbiter** *for internal questions only.*
  One train/test split is not an estimate — that is what withdrew the cervical panel.
- **Internal validation cannot detect a survey-specific effect, and neither can
  leave-one-cycle-out.** Proven here: the checkup-extended general panel gained +0.016
  inside NHANES and REVERSED to -0.026 on NHANES III (a different decade). Cycles of one
  survey share protocol, lab contract and pipeline, so resampling across them holds the
  measurement process fixed. Treat any panel without an external cohort as unvalidated.
  (An earlier version of this note blamed the prospective panel; that reversal belonged to
  the tree ensemble, and the model was later found to be the wrong kind of finding anyway.)
- **An external cohort built by the same rule shares the cohort's flaw and cannot catch
  it.** The 5-year cancer-death panel passed NHANES III (0.875 vs 0.861) and was still not a
  cancer panel: both cohorts dropped everyone who died of another cause inside the window,
  so every early death was a cancer death and "looks close to dying" scored well. Never drop
  people by what happens AFTER the index date; keep them with their cause. The clean test is
  among decedents only, where age and sex cannot separate cancer from other deaths
  (`experiments/mortality_cause_specificity.py`). Also compare a model's score on OTHER
  outcomes: a cancer panel that predicts non-cancer death equally well is measuring frailty.
- **Schema alignment**: a model may only train on features the app can actually collect.
  A panel training on something the form never asks scores an imputed median for every
  real patient and looks fine in CV.
- **Panel taxonomy**, stated on every card: *screening* (reads routine bloodwork),
  *triage* (something already found — is it malignant), *interpretation* (reads a
  diagnostic test already performed). Never let a panel default to "screening".
- **Race is a stratifier, never a feature** (eGFR precedent).
- Precision is judged as **people flagged per true case** at realistic prevalence, not AUC.

`tools/audit.py` encodes the structural checks; each exists because that bug shipped.
Run it before committing — it exits non-zero.

Live panels (as of 2026-10-04): six -- liver, pancreatic, ovarian, prostate, breast
(biopsy), breast_screening (BCSC mammogram). Withdrawn, listed in `train_models.WITHDRAWN`,
and not to be quietly revived: cervical (lucky split), colorectal/bowel and general (no gain
over the stronger age-and-sex baseline), lung (no gain on unseen NHANES 2017-18 plus
2021-23), cancer_mortality (predicts non-cancer death as well as cancer death). Withdrawing
a panel also removes form/API fields no remaining panel reads (the alcohol, smoking,
pack-years, cotinine and CRP precedents), and the trainer deletes its stale model files.
No lab-report panel screens for a named cancer; the honest remaining routes are data
needing a person (CHARLS two blood draws 2011/2015 -- registration only; MIMIC-IV via CITI;
HRS; PLCO via Dr. Chavan). See docs/PROBLEM_SOLUTIONS.md and docs/NEXT_STEPS_FOR_RAHUL.md.

- **Weakest-baseline trap**: gains must be measured against the better of logistic and
  ensemble on age+sex; the tree ensemble on 2 features ranks age coarsely and invents gains.
- **Firecrawl MCP token is invalid** and PMC blocks WebFetch with a CAPTCHA; use WebSearch
  or the Europe PMC REST API for discovery. NHANES 2021-2023 (`_L` files) exists and is a
  clean unseen cycle.
- **Look for a public clinical spreadsheet before assuming a gap needs an application.**
  PI-CAI's `clinical_information/marksheet.csv` (github.com/DIAGNijmegen/picai_labels,
  1,500 men, age/PSA/PSAD/volume/PI-RADS/ISUP) gave the project its first external
  validation of a case-control panel, free, where a consortium request was assumed.
  Watch for verification bias in such cohorts: 468 PI-CAI negatives were never biopsied,
  so every claim is repeated on the biopsied-only subset.
- **Price a data request before making it.** Re-banding cohorts that carry exact values
  showed BCSC's banded age/BMI costs <=0.0035 AUC, which removed a months-long request
  from the plan (`experiments/banding_cost.py`).
- **The right comparator is what the clinic already holds**, not chance and not always
  age: the breast panel is judged against age plus the BI-RADS density on the report
  (+0.014), and prostate against reading the PSA number alone.

- **The "overall picture" idea has been tested and does not rescue it.** 337 inputs (every NHANES
  lab, exam, symptom, habit, history item; `fetch_nhanes_wide.py`, `experiments/whole_picture.py`):
  among decedents 0.685 vs labs 0.647 vs age/sex 0.604, but for cancer death vs everyone it adds
  +0.003. The drivers toward cancer are ABSENCES of heart/diabetes/stroke; unintended weight loss
  adds nothing. Whatever beats age among decedents is cause-of-death elimination, not a tumour
  pattern. Always check what the stable drivers actually are before calling a gain "cancer".
  SEQN must be excluded from features (it is a hidden cycle marker).
- **Mechanics that cost real time.** `PAPER.md`'s `paper_results` AUTOGEN block regenerates
  sections 3.1-3.7 wholesale, so anything hand-inserted inside that span is silently deleted
  by `tools/sync_docs.py`; add new results sections after 3.9, and grep for your marker AFTER
  running the generators. `python evaluate.py <panel>` re-evaluates one panel and merges
  (a full pass takes over an hour). Commit with `git commit -F <file>`: PowerShell
  here-strings and heredocs mangled multi-line messages repeatedly. Write patch scripts with
  the Write tool, not shell heredocs (escape sequences get eaten).
- **Pre-register, then obey it, even against yourself.** The lab-bias augmentation passed on
  liver and failed on the mortality cohort; the rule needed both, so it was not adopted, and
  the follow-up on liver's other unseen cycle was run with the same thresholds instead of
  deciding post hoc.

See [[rahul-working-style]] — he wants fast iteration and no ceremony, which coexists with
the above: move fast, but never assert a number you did not measure.

## Navigator (added 2026-10-05, commit 3252f34, live at /navigator)
Symptom-and-lab rule engine (backend/navigator.py + navigator_rules.json), rules from NICE NG12 **2015**, tri-state logic, NOT clinician-reviewed (stated in file and UI). Burden measured on NHANES (platelet rule gated behind a symptom after it alerted 3.7% of symptom-free adults); readability grade 6.9 vs pre-set bar. Next human steps: Dr. Chavan sign-off + vignette kappa study (docs/NAVIGATOR_VALIDATION_PROTOCOL.md, docs/DR_CHAVAN_ROLE.md). Never claim publication/IRB/clinical validation before they exist; disclose AI use.

## Navigator evidence + paper (2026-10-05)
Pre-registered study on 37,915 NHANES adults: 3 of 4 bars missed (alerts not cancer-specific among decedents; no AUC gain over age+sex out of era; 80+ burden up to 27%); context layer (age/sex/smoking) passed calibration and ships in the API/UI. 24-page PDF paper at docs/paper/Navigator_Research_Paper.pdf, rebuilt by docs/paper/build_paper.py from result JSONs; figures by experiments/make_paper_figures.py. Paper is a DRAFT: references unchecked, Dr. Chavan's role/authorship not agreed, rules not clinician-reviewed. Headless Chrome prints the PDF; CDP script needs --remote-allow-origins=*. Never `taskkill chrome.exe` (kills user's Chrome).
Risk-ranking study (docs/RISK_RANKING_PREREG.md, experiments/risk_ranking.py, data/nhanes_wide_8c.csv.gz via WIDE_CYCLES=8): age+sex+smoking+BMI AUC 0.743 out of era; labs +0; 380-var whole picture 0.629 (missing-flag/cycle artefact, explanation untested). Paper now 35 pages, "four of seven bars missed". User believes risk-finding is also "unique": told honestly it is established (QCancer etc.); the added value is measurement + patient-facing reading.
User's real name is Palash Rakshit (GitHub palashraks-afk). "Rahul" is only the Windows account name; do not address or credit them as Rahul.
Fix pass 2026-10-05: added /screening (USPSTF lookup, verified against official pages), clinician review packet (tools/make_review_packet.py), docs/PROBLEMS_STATUS.md, MESSAGE_TO_DR_CHAVAN, COMPREHENSION_STUDY_PROTOCOL, CLINICAL_DATA_ACCESS. Curated risk model missed its bar (+0.009); era-artefact supported (F2); NHANES III vs 1999-2014 differ p=0.0005. Paper: 37 pages, "five of eight bars missed". NICE site blocks automated fetch (403); firecrawl unauthorized. User's goal in their words: "cancer screening for cancer risk".
Audit paper (2026-10-07): docs/audit_paper/NHANES_ML_Cancer_Audit.pdf, built by build_audit_paper.py from experiments/audit_*.py. 79 records, 24 included; A1 and A2 supported, A3 and A4 not. Coding by AI only; needs human second coder (15 PMIDs listed). Broader search shows ~1,400 OA NHANES+ML+cancer-ish records, so extending the census is the way to raise novelty and impact. User asked for max novelty/impact/paper-quality: I told them honest ratings, not a guarantee.
Main paper angle now (2026-10-08): docs/marker_paper/Cheap_Blood_Markers_Cancer_Risk.pdf, prereg docs/CHEAP_MARKERS_PREREG.md, scoring docs/ANGLE_ASSESSMENT.md. 0 of 12 cheap blood-count markers qualified; inflammation indices = frailty markers (OR non-cancer death up to 1.43, cancer 0.92-0.99); RDW the only marker with a positive CI (+0.009, below the 0.01 bar). User insisted the angle stay on original idea "cancer risk prediction for cheap" and wants novelty/impact/paper-quality maximised and everything in GitHub. Never claim 5/5; the next real step is replication in a cohort with registry diagnoses (UK Biobank/PLCO).
2026-10-12 (angle E, current main): Medicare MCED law paper docs/mced_paper/ (Pub. L. 119-75 ENACTED text: from 1 Jan 2029, floor 50, cap 65 in 2029 rising 1/yr = birth cutoff, anyone 65+ on 1/1/2029 never covered; prereg first quoted the INTRODUCED bill (2028, cap 68) - caught by a re-check agent and corrected openly; always verify enacted text, govinfo BILLS-119hr7148enr). Results: 2029 covers ~12% of Medicare-age 5y cancer deaths; free-question triage 2.3x yield (smoking does it); Medicare 50-64 have 2.3x cancer death. Free-data ladder (NHANES 2003-06 accelerometry, fetch_nhanes_activity.py): activity/checkup add nothing for cancer death, +0.032 for other-cause. App page /multicancer (client-side, model.json). Honest scores 5/4/4 (impact not 5). User is entering the Congressional App Challenge (deadline 26 Oct 2026 noon ET; user registers). UI rules user gave: one font family, one accent, no icon tiles/nested cards, one fade-in, real empty states.
2026-10-09: replicated cheap-marker null in NHANES III (data/nhanes3_markers.csv.gz via fetch_nhanes3_markers.py); RDW +0.0092 vs +0.0093; decision analysis; recorded novelty search found close prior work (PMIDs 41958568, 38805227, 39010980) so novelty stays 3. User asked for 5/5/4 on novelty/impact/quality; told them honestly that 5 on novelty and impact is not reachable without outside data, registry cohort and clinicians, and that I did not move the scale. Pending decision: which new question to pursue, or the human steps (register CHARLS/UKB, second analyst, clinician).
