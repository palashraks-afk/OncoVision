# Navigator evidence study: what was decided before looking

Written and committed BEFORE `experiments/navigator_evidence.py` was run, so the bars
cannot move to fit the result. The git history shows the order.

## The cohort

`fetch_nhanes_navigator.py` builds `data/nhanes_navigator.csv.gz`: 39,692 adults aged 20+
from eight NHANES cycles (1999-2000 through 2013-2014), linked by NCHS to the National Death
Index through 31 December 2019, with 6,941 deaths and 1,337 from cancer. A second, earlier
cohort, NHANES III (1988-1994), is already in `data/nhanes3_mortality_full.csv`.

Each person carries haemoglobin, MCV, platelets, white count, ferritin in some cycles,
smoking, and four symptom proxies (weight loss that was not intentional, chronic cough,
breathlessness). NHANES never asked about bleeding, lumps, bowel change, appetite or
tiredness, so most of the guideline's symptoms are invisible here and **every alert rate is
a lower bound** on what someone reporting those symptoms would see.

Everyone is 40 or over with a haemoglobin and platelet count. Nobody is dropped for how they
died. The outcome is death from cancer (NCHS underlying cause 2) within 60 months of the
exam, which is a later and harsher endpoint than a diagnosis.

## What the engine is given

The real engine (`backend/navigator.py`), not a re-implementation. Labs as measured.
`ever_smoked` when known. A proxy symptom is passed only when it is present; an absent or
unasked symptom is not passed. Alert = the engine returns `talk_soon`, `worth_raising` or
`mention`.

## Questions and bars

| # | Question | Bar to call it a success |
|---|---|---|
| 1 | Is the alert burden stable across eras? Labs-only alerts for adults 40+ in NHANES III, 1999-2006, 2007-2014 | "Talk soon" at most 3% in every age band and era, and any alert at most 20%, in all three eras |
| 2 | Are alerts cancer-specific? Among people who died within 10 years, are alerted people more likely to die of cancer than of something else? Age and sex adjusted logistic regression. | Adjusted odds ratio with 95% interval above 1 in the pooled 1999-2014 cohort AND point estimate above 1 in NHANES III |
| 3 | Does an alert add to age and sex? Five-year cancer death, age + sex versus age + sex + alert, trained on 1999-2006 and tested on 2007-2014. | Bootstrap 95% interval of the AUC gain excludes 0 in the held-out era |
| 4 | Is the disclosed "small red cells means iron deficiency" assumption sound? In cycles with ferritin, how often is MCV below 80 in iron-deficient people (sensitivity) and how often is it iron deficiency when MCV is below 80 (PPV)? | Reported, no bar. It tests an assumption the tool already discloses. |
| 5 | Which rules drive alerts and which carry the cancer signal? | Descriptive, no bar. Nothing here may be used to claim a rule works. |
| 6 | Is an age-and-sex context layer ("how much of this is just your age") calibrated out of era? Train 1999-2006, test 2007-2014. | Calibration slope between 0.8 and 1.2 in the held-out era. Shown to users only if it passes. |

## How a failure is handled

A bar that is missed is reported as missed, in the paper's results and abstract, and the
tool's own text is changed to match. The earlier project found that blood work alone barely
beats age for cancer death, so Question 3 is expected to be close. If it fails, the paper
says the navigator's value is its explanation and its guideline fidelity and not a gain in
prediction, and says that the guideline rules were never built as a predictor.

## What this cannot show

Whether any alert is a correct referral. The outcome is death, not diagnosis. The symptoms
are proxies. The cohort is US adults examined in the 2000s and 2010s, and the guideline is
UK 2015. A good result would be consistent with the rules being useful. It would not
validate them.

## Deviations, recorded after the first run

The bars and questions above were not changed. These things were done after the first run
of `navigator_evidence.py`, in this order, and are listed so a reader can judge them.

1. **2015-2016 and 2017-2018 were added to Question 1 only.** They have too little follow-up
   for an outcome, so they feed the outcome-free burden question and Question 4, and nothing
   else. The cohort therefore grew from 39,692 to 49,782 adults (20+) and from 22,583 to
   28,580 aged 40+ with a blood count.
2. **The mesothelioma rule was restricted to its asbestos and X-ray paths.** A smoker with a
   cough was being shown wording about asbestos. The other paths led to the same chest X-ray
   the lung rule already recommends. No bar depends on this.
3. **Question 7 was added** (each flag on its own, among decedents) after Question 2 failed,
   to see which flag was responsible. It is exploratory and carries no bar.
4. **A lower-tier match for the same cancer site is no longer shown beside a higher-tier one.**
   Display only; no alert state changed.
5. **The context layer** (Question 6) was put in the app because its bar was met, as the
   table above says it would be. Lab-only alerts now carry a caveat because Question 3 was
   missed.

The first run's results for Questions 1 to 6 were unchanged by items 2 and 4 except for the
per-rule counts in Question 5.
