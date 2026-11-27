# Pre-registration: who Medicare's new multi-cancer blood-test law reaches, and what free questions add

Written and committed on 12 October 2026, before the analysis below is run. The NHIS cohort, the
free-information model and its performance are already known from `docs/CANCER_AGE_PREREG.md` and the
paper in `docs/cancer_age_paper/`; that is stated plainly so nobody mistakes this for a blind analysis. What is
new and not yet computed is everything about the law, the test and the Medicare population.

## The law (primary source)

The Nancy Gardner Sewell Medicare Multi-Cancer Early Detection Screening Coverage Act (H.R. 842, 119th
Congress), enacted in the Consolidated Appropriations Act, 2026 (signed 3 February 2026). Text read from
govinfo (BILLS-119hr842ih):

- covers FDA-cleared or approved multi-cancer early detection (MCED) tests furnished on or after 1 January 2028,
  subject to a national coverage determination;
- **no payment if, as of 1 January of the year, the person has attained 68 (for 2028), the age rising by one
  year each later year**; one test per 11 months;
- payment before 2031 equals the multi-target stool DNA test rate in effect at enactment. CPT 81528 on the 2026
  Clinical Laboratory Fee Schedule (Q1 and Q4 2026 files): **$508.87**.

The enacted text should be checked against H.R. 7148 before submission; the age rule is as quoted in
reports of the enacted law (AZBio, Prevent Cancer Foundation).

Consequence of the age rule: someone who is 68 or older on 1 January 2028 (born in 1959 or earlier) never
becomes eligible, because the cap rises one year per year, exactly as fast as they age. The law covers
birth cohorts, not ages. The analysis asks what that means.

## Data

NHIS 1997-2009 linked to the National Death Index through 2019 (`fetch_nhis_cancer_risk.py`), now also
with the NHIS Medicare-coverage recode. Interview years 2005-2009 (the temporal test set) for everything
descriptive, survey-weighted with the sample-adult weight. Outcome: death from cancer within **five years**
of interview (closer to the one-year detection window of an MCED test than ten years). The risk model is
the pre-registered F-ladder refitted on 1997-2002 with the five-year outcome.

## Test performance (published, fixed in advance)

PATHFINDER 2 (Nabavizadeh et al., Nature Medicine 2026, PMID 42773207), adults 50 and older:
cancer detection rate 0.54%, specificity 99.64%, positive predictive value 60.3%.

## Questions and analyses

1. **Coverage of lethal-cancer risk.** Among Medicare-age adults (65-84) plus under-65 Medicare
   beneficiaries, what share of five-year cancer deaths falls in people the law would pay for in 2028
   (under 68, and under-65 beneficiaries), and in a stylised 2035 (the 2028-eligible cohorts aged 65-74,
   plus under-65 beneficiaries)? Weighted shares with bootstrap 95% intervals.
2. **Expected yield and false alarms per test, by group.** Mapping assumption (stated as an assumption):
   the chance that a test finds a cancer is proportional to the person's predicted five-year cancer-death
   risk, scaled so that adults 50 and older in NHIS 2005-2009 reproduce PATHFINDER 2's 0.54% detection rate.
   False positives = (1 - 0.9964) x (1 - detection probability). PPV, cancers found per 1,000 tests, and
   Medicare cost per cancer found at $508.87 per test (test cost only; diagnostic workup excluded and said so).
   Sensitivity analysis: detection proportional to risk^0.5 (a weaker link between lethality and detection).
3. **Free-information triage inside the covered group.** For a fixed number of covered tests, testing those
   with the highest predicted risk (age, sex, smoking status) versus testing at random among the eligible:
   cancers found, PPV and cost per cancer found.
4. **Under-65 Medicare beneficiaries** (mostly disability): five-year cancer-death risk, smoking prevalence and
   expected PPV against same-age adults without Medicare.

## What would count as a finding (fixed now)

- Q1: the 2028-eligible group holds less than half of the five-year cancer deaths among Medicare-eligible
  adults, with an interval excluding 50%.
- Q3: risk-based triage raises cancers found per 1,000 tests by at least 25% over random testing among the
  eligible, with an interval above zero.
- Q4: under-65 beneficiaries have higher predicted and observed five-year cancer-death risk than same-age
  adults without Medicare, with intervals excluding no difference.
A miss on any of these is reported as a miss.

## Limits stated in advance

Cancer death, not MCED-detectable incidence; the mapping is an assumption and is varied. NHIS 2005-2009 age
structure and smoking are older than the 2028 Medicare population. No MCED test has shown a reduction in
cancer deaths; NHS-Galleri missed its primary endpoint (fewer stage III-IV cancers), so nothing here claims
lives saved by MCED testing. This is an analysis of who a law reaches and how to spend it, not of whether
the test works.

## Correction, 12 October 2026 (after the first run): the enacted law differs from the introduced bill

The section above quoted H.R. 842 **as introduced**. An independent re-check found, and I confirmed in the
enrolled text of H.R. 7148 (govinfo BILLS-119hr7148enr; Pub. L. 119-75), that the enacted version differs:

- tests furnished **on or after 1 January 2029** (not 2028);
- no payment if the person "is under 50 years of age" or, "as of January 1 of such year, has attained" the
  specified age, which is **65 for 2029**, rising by one year each later year;
- one test per 11 months and the stool-DNA payment rate are unchanged.

So anyone aged 65 or older on 1 January 2029 (born before about 1964) never qualifies, and in 2029 the people
covered are Medicare beneficiaries aged 50-64 (mostly people entitled through disability) plus those turning 65
that year. The first run, on the introduced text, is kept as `experiments/mced_law_result_introduced_bill.json`
and is not reported as the result. The analysis is re-run with the enacted rule, everything else unchanged:

- Medicare-eligible pool: adults 65-84 plus Medicare beneficiaries aged 50-64 (the law's floor is 50).
- Covered in 2029: Medicare beneficiaries aged 50-64, plus adults aged 65 at interview (a stand-in for those
  turning 65 during the year).
- Stylised 2035: the covered birth cohorts are then 65-70, plus beneficiaries aged 50-64.
- Q2 groups, Q3 triage group and Q4 follow these definitions. Bars are unchanged.

Second caveat found by the re-check: Q3's yield ratio is a ratio of observed death rates, so the PATHFINDER 2
anchor only scales absolute numbers; and most of the triage gain comes from smoking (reported).

## External check against the NHS-Galleri trial, written 25 November 2026 before the NHIS side is computed

Source: NHS-Galleri test-performance paper, Nature Medicine, 22 September 2026, PMID 42773209, open access full text
via Europe PMC (PMC13668664). Aggregate cancer detection rate over three annual rounds: ages 50-59 0.18% (95% CI
0.14-0.22) and 70-79 0.78% (0.71-0.85); male 0.58% (0.53-0.63) and female 0.38% (0.34-0.42). Ratios: age 4.33
(about 3.4 to 5.5), sex 1.53 (about 1.33 to 1.75). Enrolment was ages 50-77 (maximum 78).

Our paper assumes the chance a test finds a cancer is proportional to the group's cancer-death rate. **Check:** in
NHIS 2005-2009 (weighted, five-year cancer death), the ratio of ages 70-78 to 50-59, and of men to women aged 50-78.
**Pass:** the 95% bootstrap interval of each NHIS ratio overlaps the trial ratio's interval. The square-root mapping
(the sensitivity analysis) is compared the same way, and both are reported whatever happens.

Not blind: yields by other age bands from the same NHIS data had already been seen. Different country, years and
outcome (the trial counts cancers the test found, not deaths).

**Already known to be wrong:** the same trial paper reports that the false-positive rate rose with age (0.18% to
0.76%) and that positive predictive value was relatively homogeneous across age and sex. The paper and the app assumed
a constant false-positive rate, which produced positive predictive values that rose with risk. That was an error. All
group-specific predictive values and false-alarm counts are to be removed.

## Replication in newer data, written 27 November 2026 before NHIS 2010-2014 is downloaded or opened

Interview years 2010-2014 have a full five years of follow-up by 31 December 2019 and have not been used anywhere in this
project. They are processed by the same code (`fetch_nhis_cancer_risk.py` layouts and cleaning). The ranking model stays
frozen as fitted on 1997-2002. Population, definitions (enacted law), outcome and weights are unchanged.

Bars, the same as before: **Q1** the 2029-covered share of Medicare-eligible five-year cancer deaths has a 95% interval
below 50%. **Q3** testing the highest-risk 20% of the covered group on free answers raises the observed cancer-death rate
at least 1.25 times, with the interval lower bound at least 1.25. **Q4** Medicare members aged 50-64 have higher observed
and predicted five-year cancer-death risk than same-age adults without Medicare, intervals excluding no difference.
Also reported, no bar: the same numbers pooled 2005-2014, and the yield-assumption check against NHS-Galleri age and sex
ratios. If a year's file cannot be read with the same code, it is left out and said so. Results are reported whatever they are.
