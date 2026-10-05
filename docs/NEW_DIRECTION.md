# A new direction: from "do you have cancer" to "will screening help you"

## The idea in one paragraph

The original idea was to read the labs and information a person already has and say
something useful about cancer. The evidence says routine labs cannot find a solid
tumour. They are good at something else: telling how much heart, kidney and metabolic
disease a person carries, which is what decides whether they are likely to live long
enough to benefit from cancer screening. Guidelines say not to screen people whose life
expectancy is under about ten years, because screening takes that long to pay off while
its harms arrive immediately. Today that judgement is made by age, or by clinicians'
impression. So the app does not detect cancer. **It reads the report a person already
holds and helps them and their doctor decide whether each screening is likely to help
them or not.**

## What the app does for a person

1. They upload the lab report they already have and answer a few questions. The parser
   and the form already exist.
2. It shows, in plain words, where their health outlook sits compared with other people
   their age and sex: better than typical, typical, or worse. It also shows how much of
   that is just their age and how much their own results add.
3. For each recommended screening they are due (breast, colorectal, lung, cervical,
   prostate), it says whether it is likely to be worth it, worth discussing, or probably
   low value for them now, and why.
4. It produces a one-page summary to take to the doctor. It does not tell anyone to stop
   or to start.

**Who it helps, in both directions.** People with a poor outlook who are being screened
anyway face the procedure harms and overdiagnosis with little chance of benefit. And
healthy older adults who stop screening at a birthday because "75 is the cutoff" lose
years of benefit they would have had. An age rule is wrong in both directions, and this
is the first use of the lab report in the project that addresses both.

**The cost side of the original goal.** "Cheap enough to be routine" was about getting
more people screened. This works from the other end: stop spending procedures on people
who cannot benefit, so more can go to people who can. No new test is needed.

## Evidence so far

Everything below is on public data, with the bar written before the run, and the labs
alone from one report.

**Do labs add anything beyond age and sex? Yes, on an outside cohort.** Trained on
NHANES 1999 to 2008, applied unchanged to NHANES III (1988 to 1994), ten-year death from
any cause:

| | Age and sex | Plus labs | Gain (95% CI) |
|---|---|---|---|
| All adults | 0.879 | 0.901 | +0.022 (+0.019 to +0.025) |
| **Ages 65 to 84, where the decision is made** | 0.700 | 0.762 | **+0.062 (+0.048 to +0.074)** |

At a matched 30% flagged as least likely to benefit, 14.3% of adults 65 to 84 are
classified differently. The people only the labs flag died within ten years 66% of the
time, against 56% for the people only age and sex flag. Observed ten-year mortality in
the flagged group rises from 69.3% to 71.7% (+2.3 points, CI +0.3 to +4.4).

**Do labs add anything beyond what the existing tools use? Yes, modestly.** The standard
tool (the Lee-Schonberg index) uses age, sex, BMI, smoking, diabetes, emphysema, heart
failure, past cancer, hospital stays and difficulty walking, and no lab values. Adults 65
to 84 in NHANES 1999 to 2008, 4,579 people, 1,731 deaths, repeated cross-validation:

| Inputs | AUC |
|---|---|
| Age and sex | 0.713 |
| Broad history and function, no labs (203 inputs, a harder comparator than the 12-item index) | 0.796 |
| **22 lab values from one report, nothing else** | **0.769** |
| History, function and labs | 0.814 |
| **Labs added on top of history and function** | **+0.018 (+0.012 to +0.023)** |

A lab report alone, which needs no questionnaire, gets most of the way to a 203-variable
history. That is the practical point for an app that starts from the report.

Evidence: `experiments/screening_benefit_reclassification.py`,
`experiments/screening_benefit_vs_history.py`.

## Is it new?

I cannot prove that nothing like it exists. What I found: the screening-cessation
indices that clinicians use contain no lab values; the lab-based mortality scores I know
of (PhenoAge-type) are built to estimate biological age and are not aimed at cancer
screening decisions; and my searches turned up no patient-facing tool that reads a lab
report to inform whether screening is likely to help. **Do a proper literature search
before claiming novelty in print.** The components exist separately. The combination is
the contribution.

## What is not yet known, and has to be said

- **Rank, not risk.** The models here are trained with class balancing, so their
  probabilities are not true risks. A deployed tool needs calibration to current
  population mortality. That is straightforward and not yet done.
- **Mortality is not benefit.** Ten-year survival is the input to the screening
  decision, not the decision. Net benefit also depends on how likely the cancer is,
  and this project found that labs add nothing to cancer risk itself. The cancer side
  would use age-specific incidence, which is standard.
- **The history comparison is internal.** The history and function questions do not
  exist in NHANES III, so "labs on top of history" has not been tested on an outside
  cohort. HRS and CHARLS carry labs, function, conditions and mortality for the same
  people, so registering for them tests this too.
- **The effect is modest.** +0.018 over history, and +2.3 points in observed mortality
  among those flagged. It is real and it is not large. The honest framing is a better
  conversation aid, not a verdict.
- **A cohort from the early 1990s.** Mortality has fallen since. That is one more
  reason it needs recalibration.

## Risks, and how the design handles them

Telling someone to skip a screening is a sensitive act. The app must not do it. It frames
every output as something to discuss with their clinician, it never says "stop", and
guidelines already tell clinicians to individualise on exactly this basis, so the tool
supports the standard of care and does not oppose it.

## What would make it a paper

A pre-registered external validation of whether routine laboratory values improve
ten-year survival estimation beyond history and function in adults 65 to 84, with the
decision-relevant analysis (who is classified differently, and what happened to them) as
the primary result. The first external cohort would be HRS or CHARLS. That is a paper a
geriatrician or a primary-care physician would read, and it can sit beside the
methods-and-negative-results paper in `docs/PAPER_ANGLE.md`.

## A practical question for Dr. Chavan

CHOC is a children's hospital. This direction is about older adults. Ask whether an
adult oncology or geriatrics co-mentor would be appropriate, and whether he would
rather the project stay with the report-based panels (breast and prostate), where the
clinical connection is closer.
