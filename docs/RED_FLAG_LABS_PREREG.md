# Pre-registration: hidden red flags on ordinary lab reports

Written and committed on 26 November 2026, **before** the 2017-2020, 2001-2002 and NHANES III ferritin data used here
have been downloaded or looked at. Known going in: in this project, routine blood values as continuous markers added
almost nothing to cancer-death prediction and behaved like frailty markers (`docs/CHEAP_MARKERS_PREREG.md`), and NICE
NG12 rule alerts were not cancer-specific (`docs/NAVIGATOR_EVIDENCE_PREREG.md`). Iron-deficiency anaemia defined by
ferritin, the guideline construct below, has not been tested here.

## Why this design is different

Studies of abnormal blood tests and cancer (UK, Danish and Swedish primary-care records) can only see people whose
doctor ordered a test, usually because something was wrong. That makes the cancer signal of a red flag hard to separate
from the reason for testing. NHANES measured every participant's blood, whatever their symptoms, and asked whether they
were being treated. It can therefore answer two questions records cannot:

1. **Societal:** how many US adults are walking around today with an ordinary lab result that a US guideline says should
   trigger a cancer work-up, and do not know? Who are they, and had they been seeing a doctor?
2. **Scientific:** in people tested regardless of symptoms, does that guideline red flag predict death from cancer
   specifically, or only death in general?

## The red flag (fixed now, from the guideline text)

**Iron-deficiency anaemia (IDA)** per the American Gastroenterological Association 2020 guideline (Ko et al.,
Gastroenterology 2020;159:1085-1094): anaemia (haemoglobin below 13 g/dL in men, below 12 g/dL in women, the WHO
definition) **and** serum ferritin below 45 ng/mL. The guideline strongly recommends bidirectional endoscopy for
asymptomatic men and postmenopausal women with IDA. Population: men aged 50+ and women aged 50+ (age 50+ stands in for
postmenopausal; women whose reproductive-health answers say they still menstruate are excluded where that item exists).

Comparison groups: **non-iron anaemia** (anaemia with ferritin 45 or higher; mostly inflammation, kidney disease, age)
and **no anaemia**.

Secondary (descriptive only, no bar): microcytosis without anaemia (MCV < 80 fL), thrombocytosis (platelets > 400 x10^9/L),
each in adults 50+.

## Data

- **Societal (current):** NHANES 2017-March 2020 pre-pandemic (files P_DEMO, P_CBC, P_FERTIN, P_MCQ, P_HIQ, P_HUQ, P_RHQ),
  MEC exam weights, strata and PSUs. If ferritin for all adults exists in 2021-2023 (FERTIN_L), the same estimate is repeated
  there as a check.
- **Scientific (follow-up):** NHANES 2001-2002 (ferritin measured in adults) and NHANES III 1988-1994 (ferritin in adults),
  both linked to the National Death Index through 2019. NHANES III is the independent replication dataset.

## Societal estimates (descriptive, survey-weighted, with design-based 95% intervals)

1. Prevalence of IDA among US adults 50+, and the national count (weighted sum of MEC weights).
2. Share of those with IDA who report **no treatment for anaemia in the past 3 months** (MCQ053) = "unrecognised".
3. Among the unrecognised: share with a routine place of care (HUQ030) and with 2+ health-care visits in the past year
   (HUQ051), insurance (HIQ011), family income-to-poverty ratio, race and ethnicity (as a stratifier only).
   **The societal finding is registered as:** at least half of unrecognised IDA occurs in people who have a routine place of
   care (a missed opportunity, not only an access problem), with the interval above 50%. A miss is reported as a miss.

## Scientific test (pre-registered bars)

Adults 50-84 without reported earlier cancer, 2001-2002 and NHANES III analysed separately and pooled. Outcome: cancer
death within 5 years (primary) and 10 years (secondary). Non-cases include other-cause deaths; nobody is dropped for how
they died. Model: logistic, adjusted for age (spline), sex and smoking (ever/never; current where available).

- **Bar S1 (signal):** IDA vs no anaemia, adjusted odds ratio for 5-year cancer death above 1 with the 95% interval above 1,
  in the pooled data **and** pointing the same way in each dataset.
- **Bar S2 (cancer-specific, the key test):** the IDA odds ratio for cancer death is larger than its odds ratio for
  other-cause death, tested as the ratio of the two log-odds ratios with a bootstrap interval above 1. A flag that predicts
  all deaths equally is frailty, not cancer.
- **Bar S3 (contrast):** the cancer-death odds ratio for IDA exceeds that for non-iron anaemia (bootstrap interval of the
  difference above 0). The hypothesis: iron deficiency signals occult bleeding from a tumour; non-iron anaemia signals
  frailty.
- **Yield (reported, no bar):** people with IDA per cancer death within 5 years, against the same for no anaemia.

## Limits stated in advance

Death from any cancer, not gastrointestinal cancer specifically (public mortality files have no site). Ferritin rises with
inflammation, so some IDA is missed; that biases toward finding nothing. Self-reported treatment is a crude proxy for
"known". Cross-sectional societal estimates show a gap at one time point and cannot say what happened next. Small numbers of
IDA cases in 2001-2002; NHANES III carries most of the power. Associations, not causes.

## Addendum, 26 November 2026 (after download, before any analysis result)

Seen so far: row counts, ferritin coverage, and median haemoglobin, MCV and ferritin per dataset (to check units).
No group, rate or model has been computed.

1. **Bar S2 wording.** "The ratio of the two log-odds ratios" was a slip: the stated intent is that the IDA odds ratio for
   cancer death is larger than its odds ratio for other-cause death. It is tested as the **ratio of the two odds ratios**
   (equivalently the difference of the log-odds ratios), with the bootstrap interval above 1. A ratio of log-odds ratios is
   undefined when the other-cause effect is near zero, so it would not have been a usable test.
2. **Societal population.** All adults 50+ with haemoglobin and ferritin, including people who report an earlier cancer (the
   question is who has an unworked-up red flag today). Women 50+ who report regular periods in the past 12 months (RHQ031 = 1)
   are excluded. A sensitivity estimate without people reporting earlier cancer is also given.
3. **Codes.** "Treated for anaemia" = MCQ053 = 1; refused or don't know is left out of that denominator. Routine place of care
   = HUQ030 of 1 or 3 (one place, or more than one). Two or more visits = HUQ051 code 2 or higher. Insured = HIQ011 = 1.
4. **Design-based intervals** by Taylor linearisation with strata and PSUs (SDMVSTRA, SDMVPSU) and the pre-pandemic MEC weight
   (WTMECPRP). National counts are prevalence times the weighted population 50+, with the prevalence interval.
5. **Follow-up data.** 2001-2002 had earlier cancer removed when it was built; NHANES III uses its own earlier-cancer item.
   Bootstrap uses 500 resamples to keep the run light on this machine.

## Addendum 2, 26 November 2026 (after the main result; everything below is labelled exploratory)

- 2021-2023 cannot repeat the estimate: FERTIN_L measured ferritin only in children 1-5 and women 12-49. 2017-March 2020 is
  the latest US ferritin data for adults 50+.
- Robustness checks added after seeing the main result (exploratory, reported whatever they show): the untreated share and
  usual-care share without people reporting earlier cancer; a stricter ferritin cut-off of 30 ng/mL; men only.
- The 5-year versus 10-year pattern (signal stronger at 5 years) was noticed after the run and is exploratory.

## Addendum 3, 26 November 2026 (after an independent re-check; everything here is labelled exploratory)

The re-check (`docs/independent_check/red_flag/`) reproduced every number and raised points that change how the result
is read:

1. **The registered societal bar was uninformative.** 92% of all adults 50+ have a usual place of care, so "90% of untreated
   IDA has a usual place of care" restates the population rate. The bar is reported as met but is **not** evidence of a missed
   opportunity on its own. Added, exploratory: the untreated share of IDA among people who have a usual place of care and
   two or more visits in the past year (the people a routine blood test would reach), against the same share among people
   without; and the usual-care rate in all adults 50+ as the baseline.
2. **"Untreated" is not "unrecognised".** MCQ053 asks only about treatment in the past 3 months; people who were treated and
   recovered are no longer anaemic. The paper and app say "not being treated", never "unknown".
3. **Follow-up months.** 441 NHANES III records (416 of them deaths) had no months from the exam; months from the interview
   are now used for them (flagged `months_source`). Part B is re-run with this fix; the first run's numbers are kept in git.
4. **Fragility.** Part B rests on 14 cancer deaths among people with IDA, unweighted. Bar S3 crossed 1 in one bootstrap seed and
   not in another; it is reported as not shown.

## Addendum 4, 26 November 2026: replication in an independent era and a competing-risks model (written before either is run)

**R1, replication of the societal estimate.** NHANES 2001-2002 measured ferritin in adults and asked the same anaemia-treatment
question (MCQ053) and usual-care questions (HUQ030, HUQ050/HUQ051). Using the same definitions, MEC weights (WTMEC2YR), strata and
PSUs: **pass if the share of adults 50+ with IDA who are not being treated has a 95% interval above 50%**, as in 2017-2020.
Also reported, no bar: prevalence, the untreated share among people with a usual place of care and two or more visits, and the
change between eras. The 2001-2002 demographics, questionnaire and lab files have not been opened before this addendum; the
follow-up file used for Part B carries only age, sex, smoking and labs.

**M1, competing risks.** A multinomial logistic model with three outcomes at five years (alive, died of cancer, died of other
causes), same covariates, pooled follow-up data. It replaces the two separate logistic models as the main specificity check;
the earlier models stay in the results. Reported: the relative-risk ratio for cancer death and for other-cause death with
bootstrap intervals (500 resamples), and their ratio. **Pass for specificity if the ratio's interval is above 1** (same bar as S2).

Code tests are added for the design-based estimator (against a hand-computed example) and for the red-flag definitions.

## Addendum 5, 26 November 2026: more data and three new tests (written before any of them is run)

Data added: `data/nhanes_navigator.csv.gz`, NHANES 1999-2014 (eight cycles) with haemoglobin, MCV, platelets, the unintentional
weight-loss flag and deaths through 2019 (built earlier for the navigator study; its outcomes for these definitions have not
been computed). Pooled with NHANES III for follow-up. Adults 50-84; earlier cancer removed (NHANES III item; navigator cohort
already excludes it). Five-year outcome as before.

**T1, a trigger that needs only a complete blood count.** Ferritin is not on a routine blood count; MCV is. In NHANES 2001-2002
and 2017-2020 (ferritin measured), among adults 50+: how well does **microcytic anaemia** (anaemia with MCV below 80 fL) and,
separately, **any anaemia** identify guideline IDA? Reported: sensitivity, positive predictive value, and the number of ferritin
tests per IDA case found if ferritin were added automatically. **Bar T1:** reflexing ferritin for **any** anaemia in adults 50+
finds at least 90% of guideline IDA (interval above 80%) with no more than 10 ferritin tests per case. (Microcytic anaemia is
expected to miss many cases; that is reported, not barred.)

**T2, the cancer signal on four times the data.** Exposures, each against people without it, in the pooled 1999-2014 + NHANES III
data: microcytic anaemia; non-microcytic anaemia; unintentional weight loss (5%+ in a year, not trying; navigator definition;
NHANES III not available, so 1999-2014 only); platelets above 400 x10^9/L. Multinomial competing-risks model (alive, cancer
death, other-cause death) with age spline, sex, ever smoking and cohort. **Bar T2:** for microcytic anaemia, the cancer-death
relative-risk ratio has an interval above 1 **and** its ratio to the other-cause one has an interval above 1. The other
exposures are reported with the same statistics and no bar.

**T3, timing.** If the cancer signal of microcytic anaemia comes from cancers already present, it should be strongest soon after
the blood test. **Bar T3:** the cancer-death odds ratio for deaths in months 0-24 exceeds that for months 25-60 (people alive at
24 months), with the bootstrap interval of their ratio above 1.

Bootstrap 300 resamples (light, for this machine). A recorded novelty search for T1-T3 is run after the analysis and reported.

## Addendum 6, 26 November 2026: four more angles (written before any is run; NOT yet run)

Files downloaded (columns inspected only): data/audit/P_DSQIDS, P_RXQ_RX, P_HIQ (NHANES 2017-March 2020).

- **A1 Treated, measured three ways (rigor, data).** "Treated" = MCQ053 = 1, OR a prescription whose name contains IRON, FERROUS,
  FERRIC or CARBONYL (P_RXQ_RX), OR an iron-containing supplement in the past 30 days (P_DSQTOT DSQTIRON > 0; also a stricter
  45 mg/day). Bar A1: the untreated share of guideline IDA among adults 50+ stays above 50% (interval) under the broadest definition.
- **A2 Expected cancers if the untreated were scoped (impact, projection).** Untreated count times the pooled malignancy yield of
  endoscopy in IDA from a systematic review (candidate: PMID 41784434, Eur J Gastroenterol Hepatol 2026, upper-GI malignancy 5.0%;
  colorectal figure to be read from the paper, not from a truncated abstract). Reported as a projection with the yield's interval;
  no bar.
- **A3 Who is left untreated, by payer (solvability).** Untreated IDA split by Medicare, private, Medicaid and uninsured (HIQ032A/B/D,
  HIQ011), so each payer's share and count is known. Descriptive; no bar.
- **A4 A cause in plain sight (solvability).** Share of adults 50+ with IDA on a prescription anticoagulant or antiplatelet, an NSAID,
  or a proton-pump inhibitor (drug-name lists fixed in code before running), against adults 50+ without anaemia. Descriptive; no bar.
