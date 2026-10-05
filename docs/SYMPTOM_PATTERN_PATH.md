# Reaching the original idea through patterns: symptoms and labs read together

Figures below come from search summaries of the published sources and must be checked
against the papers before they are cited.

## The original idea, stated plainly

Read everything a person has (symptoms, lab results, history) as one pattern, notice a
combination that matches cancer before they or anyone else has connected it, and say so in
plain words.

## Why "a healthy person with no prompt" is the wrong target, and what the right one is

No approach has been shown to find cancer in someone with no symptoms and no reason to be
tested, apart from special blood tests that detect a third to a half of cancers. This
project's tests of routine labs, and of 337 survey variables, found nothing.

But there is a large, real gap just before that: **people who already have early, vague
symptoms and have not realised they matter.** Most cancers are not found by a screening
test. They are found when a person with a symptom reaches a doctor, and the delay between
the first symptom and the diagnosis is where earlier recognition happens. In that setting
the pattern idea has strong evidence.

## What has been shown

| Source | What it did | Result |
|---|---|---|
| **QCancer** (QResearch, UK primary care) | Symptoms, risk factors and blood tests combined, 15 cancers | Discrimination for any cancer 0.876 in men and 0.844 in women with blood tests included; higher with blood tests than without |
| **NICE suspected-cancer guidance (NG12)** | Symptom patterns with a referral threshold of 3% risk | Two symptoms together raise the predictive value sharply: for lung cancer from 0.63% (fatigue with cough) to over 10% (coughing blood with loss of appetite, abnormal spirometry or a high platelet count) |
| **Ovarian cancer Symptom Index (Goff)** | Eight symptoms, new within a year, more than 12 days a month | Sensitivity 56.7% for early and 79.5% for advanced disease; specificity about 90% over age 50 |
| **UK HeadSmart campaign** (children) | Symptom guidance and awareness | Halved brain-tumour time to diagnosis, 14.4 to 6.7 weeks |

These work for the people who present with symptoms. They are not screening tests.

## What this means for the idea

**Symptoms and labs together beat either alone**, which is the pattern idea, and it holds
up. It is also the first version of the project's idea that does not run into the
"cancer is rare" arithmetic, because the people in it are already symptomatic, so the
background rate is far higher than in the general public.

## What is already built, and what is left

- QCancer exists as clinician and online calculators. A tool of this kind is therefore
  **not new in itself**. Check how much a public calculator already does before claiming a
  gap.
- What I did not find is one that **reads the person's own lab report automatically**,
  combines it with symptoms they describe in their own words (how long, how often, new or
  old), covers several cancers in one place, and says in plain language what to do next and
  how much of the result is just their age. That is the part Oncovision already partly has:
  the report parser and the readable cards.

## The proposed tool

A **symptom-and-lab pattern navigator.** The person describes what they have noticed (what,
since when, how often) and uploads the labs they hold. It does not predict "you have
cancer". It says one of:

1. **This combination is one that guidelines say deserves a doctor's visit soon.** Here is
   the single page to take.
2. **Nothing here meets that bar. Here is what would change that.**
3. **Not enough information.** Here is what is missing.

It uses published, validated risk patterns and thresholds, not a model invented here, so it
does not need a new dataset to be defensible. Its added value is reading the labs
automatically, putting the pieces together, and being honest about uncertainty.

## What would make it a study

- **Safety first.** A symptom tool aimed at patients can cause harm: false alarm,
  false reassurance, and over-referral. It needs a clinician-written escalation rule set and
  a usability and safety study before anyone relies on it.
- **Evaluation.** The honest test is on records that hold symptoms, labs and a later
  diagnosis: primary-care linked data in UK Biobank, the electronic-record data in All of Us,
  or a partner health system. Each needs an application and a named investigator.
- **The question to answer.** Does reading the person's own lab results alongside their
  symptoms move more people across the 3% referral threshold, correctly, than symptoms alone?
- **Comparators.** Age and sex, symptoms alone, and QCancer.

## What cannot be claimed

That it finds cancer in people with no symptoms. That it replaces a doctor. That it has been
tested in patients, which it has not.

## Next steps

1. Check the QCancer calculators and their licences, and read the validation papers.
2. Ask Dr. Chavan whether an adult-oncology or primary-care collaborator would help write
   the escalation rules and sign a data application.
3. Register for CHARLS; it also asks about symptoms and conditions alongside blood.
