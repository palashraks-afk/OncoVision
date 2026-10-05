# Children: where lab reports and cancer screening actually meet

You suggested aiming this at children, and CHOC is a children's hospital, so any real
data study would involve children. I checked what is already known before proposing
anything. The honest result is that the obvious idea is taken, one narrow gap is open,
and the project's own evidence points at it.

## What is already known (searched, not assumed)

- **Population screening of children is not recommended, and cancer is rare.** About 19
  new cases per 100,000 children a year in the US. Leukaemia is the largest share
  (about 25 to 30% of cases, roughly 4.8 per 100,000 a year).
- **Blood-count machine learning for leukaemia already exists**, including paediatric
  work. A 2025 multi-modal model combining the blood count with white-cell scattergrams
  reported 96.7% sensitivity on an external set. A leukaemia classifier would not be new.
- **Where children's diagnoses are delayed, labs do not help.** In an older
  population-based study, the median lag from first symptoms to diagnosis was about 3
  weeks for leukaemia (the doctor's part about 0) and about 9 weeks for brain tumours.
  The UK HeadSmart campaign, which was symptom guidance and awareness and not a lab
  test, halved brain-tumour time to diagnosis, from a median of 14.4 to 6.7 weeks.
- **The one place childhood cancer screening is recommended is children with a
  cancer-predisposition syndrome** (Li-Fraumeni, Beckwith-Wiedemann and others). The
  AACR consensus protocols exist, and children who followed surveillance had better
  outcomes. Real families struggle with it: in one Li-Fraumeni cohort only 29% were up to
  date with surveillance, and in a Beckwith-Wiedemann survey 33.8% of parents were unsure
  of their child's tumour risk.
- **An app for these families already exists** (HomeTown, published 2024): reminders,
  syndrome-specific schedules, a place to record visits and results, links to reliable
  information. **The surveillance-tracker idea is therefore not new.**

## The gap

What I did not find in that app, or elsewhere, is a layer that **reads the child's actual
lab results against age-appropriate norms, against the child's own baseline, and explains
the result in plain words with a clear next step.** Existing tools remind and record. They
do not interpret.

Children make interpretation hard in a way adults do not, and that is measurable.

## New evidence from this project

`experiments/pediatric_reference_flags.py`, 21,346 children aged 1 to 19 in NHANES
2005 to 2018:

| Age | Healthy-sample children flagged abnormal by fixed adult limits |
|---|---|
| 1 to 2 | **84%** |
| 3 to 5 | 71% |
| 6 to 8 | 59% |
| 9 to 11 | 48% |
| 12 to 14 | 35% |
| 15 to 19 | 26% |
| **All ages** | **50%** |

Haemoglobin alone flags 62% of 1- and 2-year-olds, and red cell size flags 57%. Against
limits for the child's own age and sex, 16.4% fall outside on at least one of four values,
which is what chance alone gives. Any tool that flags "high" and "low" against adult
limits will tell the parent of a healthy toddler that something is wrong. The fix is
age- and sex-specific limits, which is what dedicated paediatric reference programmes
(CALIPER is the main one) exist to provide.

**A red flag has to be rare in healthy children to be usable.** Using a pattern that is
adjusted for age, two or more of haemoglobin, platelets and absolute neutrophils falling
together (the classic blood-count sign of marrow failure and leukaemia), 0.25% of
children meet it. The arithmetic for leukaemia at 4.8 per 100,000 a year, applied to
unselected children and before sensitivity is even considered:

| Flag | False-positive rate | False alarms per leukaemia case |
|---|---|---|
| Any value outside fixed adult limits | 49.8% | about 10,400 |
| Two or more lineages low for the child's age | 0.25% | about 52 |

So a blood-count flag in children is only responsible where a count is already being
done for a reason, in a symptomatic child, and it must be age-adjusted.

## The idea

**An age-aware lab companion for children under cancer surveillance.** Parents upload the
child's lab reports.

1. **Reads each value against limits for the child's exact age and sex**, not adult limits.
2. **Compares with the child's own earlier results.** Children with some inherited
   conditions have chronically abnormal counts, and what matters for them is change from
   their own baseline, not distance from a population norm. This is the "your own change
   over time" idea the project identified as the most promising untested one.
3. **Explains in plain words** what each result means and what it does not.
4. **Says what to do:** nothing, mention it at the next visit, or call today. Those rules
   are written by the child's clinicians, not by a model.
5. **Tracks the schedule** the family has been given, so a late test is noticed.

It never says a child has or does not have cancer, and it never overrides the care team.

## What this is not, and what is unverified

- It does **not** screen healthy children. No evidence supports that, and the false-alarm
  arithmetic above says why.
- It does **not** replace a leukaemia classifier, which already exists.
- Details of the existing app come from search summaries and a published description, not
  from using it. Check it before claiming a gap in print.
- NHANES percentiles describe a general sample that includes ordinary childhood illness, so
  they are not healthy-volunteer reference intervals. NHANES has blood counts from age 1 and
  chemistry only from age 12, and has no AFP, so a full tool needs published paediatric
  intervals for the rest.
- The adult limits used are widely quoted values and are stated in the script as an
  assumption, not as any one laboratory's.
- No sensitivity is measured, because NHANES has no children with a leukaemia diagnosis
  and a blood count.

## Research questions a CHOC chart review could answer

1. In children under surveillance for a predisposition syndrome, does **change from the
   child's own blood-count baseline** flag malignant transformation earlier than population
   limits? The trajectory pipeline built earlier (`fetch_mimic_trajectory.py`,
   `experiments/trajectory_vs_snapshot.py`) is the starting point.
2. In symptomatic children who had a blood count, how well does the age-adjusted
   multi-lineage pattern separate leukaemia from the benign causes of low counts (viral
   suppression, immune thrombocytopenia, aplastic anaemia)? That is the decision a
   clinician actually faces.
3. Do parents understand a plain-language result better than the standard report? In the
   Beckwith-Wiedemann survey a third were unsure of their child's risk, so the room for
   improvement is measurable.

Each needs IRB approval and a clinician as principal investigator, which is the
conversation to have with Dr. Chavan.

## What I can build now

The age- and sex-specific limits for the blood count (ages 1 to 19) are public and
already computed. They can go into the app today. Chemistry below age 12 and AFP need
published paediatric intervals, which is a literature task.

## What to ask Dr. Chavan

1. Does CHOC run a cancer-predisposition or surveillance clinic, and would the families
   find an interpretation tool useful?
2. Is a retrospective review of serial blood counts in children under surveillance
   feasible, and who would be principal investigator?
3. Which clinician would write the "nothing / next visit / call today" rules?
