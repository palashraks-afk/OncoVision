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
