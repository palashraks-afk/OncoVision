# What only you and Dr. Chavan can do

Everything below needs a human with a name, an institution or a login. None of it
has been submitted. The wording is a draft to edit, and anything marked "check"
is something I could not confirm from outside the sign-up forms.

The reason these matter: every free dataset reachable without them records blood
AFTER the diagnosis, which is why three bloodwork panels were withdrawn. What is
missing is blood drawn BEFORE anyone knew, with a later diagnosis date.

## 1. CHARLS, registration only, and the cheapest route to the main question

**What it unlocks.** Blood drawn from the SAME people in 2011 and again in 2015
(confirmed from the study's blood-data release notes), with cancer asked about in
every later wave. That is two draws four years apart per person, which is the
"your own change over time" test: did a value drift in the people who went on to a
cancer diagnosis? Later waves are reported to have blood as well (check the
release notes). Nothing else free has this.

Steps:
1. Register on the CHARLS data site (charls.pku.edu.cn, "Data" section). Registration
   is what the site asks for; I could not confirm whether approval is instant.
2. Download the wave 1 (2011) and wave 3 (2015) **blood** files and the health-status
   files that carry the cancer question for waves 2 to 4, plus the user guides.
3. Put the unpacked folder anywhere and tell me where. I will read the real variable
   names from the files rather than guess them, then build the cohort and run
   `experiments/trajectory_vs_snapshot.py` against it. It is not built yet on purpose.

Caveats worth knowing before it is run: cancer is self-reported, the population is
Chinese adults over 45, and participants who died or dropped out between waves are a
selection problem of exactly the kind this project just found in NHANES, so the first
thing I will do with the files is check who is missing and why.

## 2. MIMIC-IV, an afternoon of work, then days of waiting

**What it unlocks.** Repeat blood counts per patient with timestamps, plus
diagnosis codes. That is the only way to test whether a person's own trend beats a
single draw, which is how ColonFlag works. The pipeline is already written and
tested on the open demo: `fetch_mimic_trajectory.py` and
`experiments/trajectory_vs_snapshot.py`. On the full data it runs unchanged.

Steps, in order:
1. Make a PhysioNet account at physionet.org.
2. Take the CITI course **"Data or Specimens Only Research"** (confirmed as the
   required course). It is free through the CITI program; choose your institution
   or "Independent Learner" if you do not have one.
3. Apply for credentialed access in your PhysioNet account settings. As I
   understand it the form asks for a reference who can vouch for you. Dr. Chavan
   is the natural one. (check)
4. Sign the data use agreement for MIMIC-IV, then download the `hosp` folder.
5. Run: `python fetch_mimic_trajectory.py --dir path/to/mimiciv/hosp`
   then `python experiments/trajectory_vs_snapshot.py`.

## 3. HRS 2016 Venous Blood Study, one form

**What it unlocks.** Blood count and metabolic panel drawn in 2016 from about
9,900 US adults over 50, with cancers recorded in later interview waves. A
diagnosis endpoint, and blood that really came first.

It is distributed through the HRS Data Portal under the Sensitive Health Data
order form, not as an open download (confirmed). Draft of the project description:

> We are developing and externally validating risk models that read routine blood
> test results. Existing public cohorts record blood at or after diagnosis, so
> they cannot show whether routine laboratory values carry early signal. We
> request the 2016 Venous Blood Study biomarkers, linked to the later cancer
> diagnoses recorded in the 2018, 2020 and 2022 core interviews, to test whether
> complete blood count and metabolic panel values drawn before diagnosis predict
> later diagnosis beyond age and sex. Analyses are secondary, use de-identified
> data only, and will be reported with full methods whatever the result.

## 4. ELSA, registration only

- **ELSA** (England, adults 50 and over): nurse-visit blood samples every four
  years; free download after registering with the UK Data Service (confirmed).
  Whether the interview waves carry a usable cancer-diagnosis variable is not
  something I could confirm. (check)

It would give a second country for the same two-draw test, and it is free.

## 5. Dr. Chavan: the three asks

Draft message:

> Dr. Chavan, the project has reached the point where public data cannot take it
> further, and three things need your name or your institution.
>
> 1. **PLCO.** The request to NCI CDAS for the PLCO data is written
>    (`docs/PLCO_CDAS_PROPOSAL.md`). It needs a named investigator. PLCO is the
>    only screening population with CA-125 and PSA, so it is what would let the
>    ovarian and prostate panels be tested on people who were not already
>    referred.
> 2. **CHOC records.** Does CHOC hold a cancer registry linked to laboratory
>    results? A retrospective review of blood counts before a first diagnosis,
>    with IRB approval, would give the design nothing public offers: real
>    patients, labs that came first, and dates.
> 3. **IRB.** Whether a protocol like `docs/PROSPECTIVE_STUDY_PROTOCOL.md` could go
>    through CHOC's IRB, and who would sign as principal investigator.
>
> Everything measured so far, including the panels that failed, is in the paper.

## 6. The key rotation

The Nimble API key was pasted into chat earlier in this project. It was never
written into the repository, but rotate it anyway from the Nimble dashboard.

## What to do first

Do 1 and 2 this week. One is a registration, the other a free course. Item 5 waits on
a meeting, so send the message now and let it run in the background.
