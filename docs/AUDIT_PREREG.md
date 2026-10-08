# How much NHANES machine-learning cancer prediction survives honest validation?
## A pre-registered audit of the literature and a controlled re-analysis

Written and committed on 7 October 2026, BEFORE any paper was coded and before the
cross-sectional data was analysed. Search counts below were taken the same day and are the
only things looked at.

## Why this question

Public health surveys, above all NHANES, are the data behind a very large number of
machine-learning papers that claim to predict or "detect" cancer. This project's own work found
that (a) in NHANES, age alone already ranks cancer outcomes well, (b) extra variables added
nothing reliable once a later survey era was held out, and (c) a model using every variable
looked good inside its own era and fell apart in the next one, because missing-value flags
identify the survey cycle. If those three things are general, a large part of the published
claims should not survive the same tests. Nobody appears to have measured how much.

A search on 7 October 2026 for a prior audit of this kind found only a scoping review of 16
methodological papers on survey weights and design (Oh et al., arXiv 2605.08963, May 2026), which
does not report baselines, cycle-held-out validation or cancer outcomes in what could be read, and a
general risk-of-bias review of machine-learning prediction studies (Andaur Navarro et al., BMJ 2021)
that is not specific to NHANES. This is not proof none exists.

## Part 1. The literature audit

**Search (fixed).** Europe PMC, run 7 October 2026:

    (ABSTRACT:"NHANES" OR ABSTRACT:"National Health and Nutrition Examination Survey")
    AND (ABSTRACT:"machine learning" OR ABSTRACT:"deep learning" OR ABSTRACT:"random forest"
         OR ABSTRACT:"XGBoost" OR ABSTRACT:"gradient boosting" OR ABSTRACT:"neural network"
         OR ABSTRACT:"support vector")
    AND (ABSTRACT:cancer OR ABSTRACT:carcinoma OR ABSTRACT:neoplasm OR ABSTRACT:malignan*
         OR ABSTRACT:tumor OR ABSTRACT:tumour)
    AND PUB_YEAR:[2015 TO 2026] AND SRC:MED

It returned **79 records, of which 56 are open access with full text**. Every record is
screened; there is no sampling. Papers without open full text are coded from the abstract only and
the items that need the methods section are marked "not assessable", not guessed.

**Included.** A paper that builds, trains or reports a prediction or classification model whose
target is cancer (any site) or death from cancer, with NHANES as a data source. **Excluded:**
reviews, protocols, papers that only report associations, papers where cancer is a covariate and not a target.

**Coded for each paper** (rules written here, applied by regular-expression cues and then
adjudicated by reading the matching text):

| Item | Values |
|---|---|
| Task | cross-sectional status (cancer history reported at the same visit as the predictors); mortality; incidence or other |
| Validation | none or in-sample; random split or cross-validation only; external cohort outside NHANES; held-out NHANES cycles or years |
| Baseline | age and/or sex only reported; logistic regression or an existing clinical score compared; none |
| Best reported AUC | from the abstract or results |
| Survey weights used in fitting | yes / no / not assessable |
| Calibration reported | yes / no |
| Resampling (SMOTE, oversampling) | yes / no |
| Missing data | method reported; missing-indicators used |
| Number of cancer cases | if reported |

**Reliability.** The cue-based coding and the adjudicated coding are two independent passes; their
agreement (Cohen's kappa) is reported per item. The adjudication was done by an AI assistant. A
random 15 papers are listed so a human can re-code them; their agreement is reported if done.

## Part 2. The controlled re-analysis

`fetch_nhanes_crosssectional.py` builds the task most of these papers use: predicting a
person's self-reported cancer history from the other items recorded at the same visit, NHANES 1999-2014,
using the same ~200 laboratory, examination and questionnaire columns as the earlier work. On it:

1. **Benchmark gap.** AUC of age alone, age and sex, and the best broad model, in a random split and in
   a split that holds out the later cycles (fit 1999-2006, test 2007-2014).
2. **Leakage mechanism.** The same pipeline with and without missing-value flags, and the AUC of
   predicting the survey cycle from the flags alone.
3. **Mortality version** of the same comparison, from the earlier analysis.

## Hypotheses and bars, fixed now

| # | Claim | Counts as supported if |
|---|---|---|
| A1 | Most papers validate only inside one pool of data | At least 50% of included prediction papers use only a random split or cross-validation, with no external cohort and no held-out cycle |
| A2 | Baselines are rare | At most 20% report an age-only or age-and-sex-only baseline |
| A3 | Reported performance on cross-sectional cancer status is not clearly above what age alone achieves | The median best reported AUC among those papers is at most 0.05 above the age-only AUC computed in Part 2 |
| A4 | The mechanism is real | In Part 2, the whole-picture model with missing flags scores at least 0.03 AUC higher in a random split than in the later-cycle split, and the flags alone predict the cycle with AUC above 0.9 |

**If a claim is not supported, the paper says so.** A result that the literature is mostly fine
would be a useful, publishable finding too.

## What this cannot show

Why any one paper reached its numbers; whether a paper's authors checked things they did not
report; whether performance in a paper's own setting is wrong, only that it is not shown to
hold outside it. Open-access papers may differ from the rest. Coding is by cue and AI
adjudication, not by two independent humans.

## Honest limits on how new it is

A 79-paper audit in one literature is small. The contribution would be the number, the
benchmark, and a controlled demonstration of the mechanism, not a claim that these problems were unknown.
That validation shortcuts inflate performance is well understood in general; what is new here, as far
as a short search could tell, is measuring it in NHANES and cancer.
