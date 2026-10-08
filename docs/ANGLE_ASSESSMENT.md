# Is the paper angle novel, high-impact and a good research topic? An honest scoring

Written 8 October 2026. You asked for three things from the angle: **genuinely new, high impact,
a good research paper topic**, and for it to stay on the original idea, **cheap cancer risk
prediction**. This scores every angle tried, with the evidence, and says plainly what would raise
each score.

## The scale (set before scoring, so it cannot drift to flatter the result)

| Score | Novelty | Impact | Paper quality |
|---|---|---|---|
| 1 | The exact question has been answered many times | Nobody would change what they do | Unsound, or cannot be checked |
| 3 | Known question, new evidence | Informs researchers in a niche | Sound but small or with a clear weak point |
| 5 | A new question and nothing close found | Would change clinical or policy practice | Rigorous, pre-registered, independently replicated, clear |

A short search cannot prove something has not been done, so no novelty score is above 4 here.

## The angles, re-scored after a second round of work (9 October 2026)

| Angle | Stays on "cheap cancer risk"? | Novelty | Impact | Paper quality | Verdict |
|---|---|---|---|---|---|
| **A. Symptom-and-lab navigator** | Partly | 2 | 2 | 3 | A fair student project; QCancer is public; the lab alerts failed their bars. |
| **B. Audit of NHANES machine-learning cancer papers** | Not really | 3 | 3 | 3 | Real finding, drifts from the original idea, 24 papers, AI-coded. |
| **C. Cheap blood markers: cancer signal or general frailty?** | **Yes** | **3** | **3** | **4 to 5** | **Strongest feasible angle. Quality is now close to the ceiling reachable without outside data; novelty and impact are not.** |

### What changed for C in the second round

1. **Independent replication** in NHANES III (1988-94), a separate survey with different staff and
   analysers, using three-part-differential analogues. Pre-registered bar met: 0 of 12 qualify, all six
   inflammation analogues show the frailty pattern. RDW replicates its small gain (+0.0092 against
   +0.0093), again just under the 0.01 bar. This raises **quality** (4 to 4.5 or more).
2. **Decision analysis** among adults 60 and over: NLR and SII rules find no more ten-year cancer deaths
   than flagging the oldest at the same rate. RDW's top-fifth rule finds 7 more per 100 in NHANES III
   (interval just above zero) and fewer other-cause deaths than age; PPV only about 11%. This moves
   **impact** from "informs researchers" toward "informs a tool design question", but not to a practice change.
3. **A recorded novelty search** (2,469 records) found **closer prior work than I first thought**:
   single and paired markers have already been compared across causes of death in general-population
   cohorts (for example SII and SIRI in a Chinese community cohort, where SII was not linked to cancer
   death; RDW-to-albumin in NHANES and the UK Biobank). Nothing found does twelve markers head to head
   with a negative control, an unseen era and a replication, so the contribution is the design, not the
   idea. **Novelty therefore stays at 3, and I lowered my earlier "3 to 4".**

### Why 5 on novelty and 5 on impact cannot be reached from here, and why I am not relabelling

- **Novelty 5 means a new question with nothing close found.** The recorded search found close work
  on the cause-specific question. A question that is new cannot be manufactured by re-running the
  same analysis on more markers.
- **Impact 5 means changing clinical or policy practice.** No retrospective survey analysis does that. It
  needs cancer diagnoses (registry outcomes), a prospective or at least a clinical-records evaluation,
  and clinicians who would use the result. None is available without access to data and people.
- **Quality 5 means independent replication.** This now has a replication in a separate survey, which is real,
  but it is the same national programme, the same analyst, and death rather than diagnosis. A 5 needs a
  cohort with registry cancer diagnoses and a second person re-running the work.

Moving the scale to make these come out would be dishonest, so it has not been moved.

### The steps that need a person (the only way to raise the scores further)

1. Register for a dataset with cancer diagnoses (CHARLS, UK Biobank, PLCO, All of Us; see
   `docs/CLINICAL_DATA_ACCESS.md`) and replicate Tests 1 and 2 there. Quality to 5, novelty up a point.
2. A second person reruns everything from the repository.
3. A clinician co-author who can say what decision a cheap marker could change. Impact up a point.
4. Choose a new question rather than a bigger version of this one (options on request).

## Why C scores as it does

**What it did.** Twelve cheap blood-count markers, two tests, bars pre-registered, 12,645 US
adults with ten-year follow-up. Result: none qualified. The popular inflammation indices track
dying of other causes (odds ratio per SD up to 1.43) and not dying of cancer (0.92 to 0.99), and
none adds more than 0.01 AUC to age, sex, smoking and BMI in an unseen era. The associations the
literature reports do reproduce (up to 1.22 per SD) but add at most 0.006 AUC.

**Novelty, 3 to 4.** A Europe PMC search on 7 October 2026 found 103 NHANES papers on these
indices and cancer, only 5 mentioning incremental prediction and 6 mentioning specificity,
negative controls or competing causes. We found none that tests twelve markers against both
properties with an unseen era and a negative-control outcome. The ingredients (negative control,
incremental AUC) are standard; the application and the head-to-head are the contribution. Capped
at 4 because a short search is not proof, and 3 because the question "do inflammation markers
reflect frailty?" is partly known from the association papers themselves.

**Impact, 3.** It matters to a large and active literature and to anyone tempted to use NLR or SII
as a cheap cancer marker, and it gives a rule they can apply (report the baseline and a
negative-control outcome). It will not change clinical practice, because no clinical decision
rests on these markers yet, and the result is negative and from one US survey.

**Paper quality, 4.** Clear question, bars written down before running, an expected result that
was met and said so, robustness checks that did not change the answer (lag, sex, age, horizon),
all code and numbers public. Held back from 5 by: death rather than diagnosis as the outcome, one
survey, no independent replication, and the analysis having been run by an AI assistant with no
second person yet re-running it.

## Why the original idea cannot score 5 on all three with this data

The original idea was a cheap, general cancer risk predictor. The measurable truth in public
data is that **the cheapest information (age, sex, smoking, BMI) is already the best risk
information, AUC about 0.74, and routine blood values add nothing reliable to it.** That is a
real and useful result, and it limits the ceiling on all three scores. A paper that claimed a
working cheap blood-based cancer risk tool would score higher on impact and novelty and would not
be true.

## What would raise the scores (in order of payoff)

1. **Replicate Tests 1 and 2 in a cohort with cancer-registry diagnoses** (UK Biobank, PLCO,
   All of Us; see `docs/CLINICAL_DATA_ACCESS.md`). Moves quality 4 to 5 and novelty up one point,
   and removes the "death, not diagnosis" limit. This is the single most valuable step.
2. **Have a second person re-run the analysis from the repository** and confirm the numbers.
3. **Test RDW first, site by site and over time** (it was the one marker with a positive interval
   for cancer death). A real positive finding would raise impact more than any negative result.
4. **A clinician co-author** who can say which decisions a cheap marker could realistically change.
5. **Extend the search** so the novelty claim rests on a systematic search, not a short one.

## Where everything is

- Paper: `docs/marker_paper/Cheap_Blood_Markers_Cancer_Risk.pdf` (build: `docs/marker_paper/build_marker_paper.py`)
- Pre-registration: `docs/CHEAP_MARKERS_PREREG.md`
- Analysis: `experiments/cheap_markers.py`, `experiments/cheap_markers_sensitivity.py`; figures `experiments/make_marker_figures.py`
- Earlier angles: `docs/paper/` (navigator and risk ranking) and `docs/audit_paper/` (audit)

## Angle D: free information and cost-matched screening invitation (added 10 Oct 2026)

Paper: `docs/cancer_age_paper/Free_Information_Cheaper_Cancer_Screening.pdf`. Pre-registered in
`docs/CANCER_AGE_PREREG.md`. 183,585 NHIS adults, two external surveys, equity analysis.

Honest scores on the same 1 / 3 / 5 scale (my judgement, not an external review):

| Criterion | Score | Why not higher |
|---|---|---|
| Novelty | 4 | Risk-based screening start ages exist for lung, bowel and breast; the general free-information, cost-matched, three-survey framing was not found in a recorded 6-query search (not proof). |
| Impact | 4 | Concrete cheaper-invitation result (about 21% fewer invitations to reach half of cancer deaths), but outcome is all cancer deaths, gain is mostly smoking, and weaker for non-White adults. |
| Research-paper quality | 4 to 5 | Pre-registered, replicated in two surveys, honest miss on the fairness bar. Not 5 because no second analyst has re-run it and there are no registry diagnoses. |

**The target of 4 on one criterion and 5 on the other two was not reached.** Raising impact and
novelty needs a registry-outcome cohort (cancer sites, diagnoses), a clinician co-author and a
second analyst; those are human steps.
