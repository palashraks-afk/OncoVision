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

## The angles

| Angle | Stays on "cheap cancer risk"? | Novelty | Impact | Paper quality | Verdict |
|---|---|---|---|---|---|
| **A. Symptom-and-lab navigator** (guideline reader) | Partly (a guideline reader, not a risk predictor) | 2 | 2 | 3 | A fair student project. QCancer is public and CRUK has a clinician guide; the lab alerts failed their bars. |
| **B. Audit of NHANES machine-learning cancer papers** | Not really (it is about other people's papers) | 3 | 3 | 3 | Real finding (88% internal-only validation, none with an age baseline), but it drifts from the original idea, covers only 24 papers and was coded by an AI. |
| **C. Cheap blood markers: cancer signal or general frailty?** | **Yes. It is the original idea tested directly: can the routine blood count give cheap cancer risk information?** | **3 to 4** | **3** | **4** | **The strongest angle that is feasible now. See below.** |

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
