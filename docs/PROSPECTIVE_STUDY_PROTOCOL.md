# Draft protocol outline: prospective evaluation of Oncovision triage panels

**Status: draft for review. Not submitted, not approved, not registered.** Every
result Oncovision reports today comes from retrospective public data. No patient
has used the tool and been followed to an outcome. This outline exists so that the
gap can be closed properly, under an IRB and a named principal investigator.

## Question

Among adults attending routine care who already have a recent lab report, does an
Oncovision panel's output agree with what subsequent standard-of-care work-up finds
— and, for the one panel with a validated rule-out cut, does that cut exclude
patients who turn out not to have cancer at the rate the retrospective data
promised?

The study **observes**. Oncovision output is not shown to patients or clinicians and
does not change care. That is deliberate: no panel has earned the right to change a
clinical decision, and an observational design removes the risk that it does.

## Panels in scope

Only panels whose retrospective evidence supports a prospective test:

| Panel | Why included |
|---|---|
| Breast, mammogram report | Validated on 597,859 held-out BCSC mammograms; beats age alone at matched sensitivity |
| Liver disease | Gain over age and sex confirmed on a withheld later survey cycle |

Excluded, with the reason: **bowel** (withdrawn — its lab values added nothing to age
and sex); **general** (gain over age and sex of about +0.002); **lung** (gain
unconfirmed on unseen data); the **case-control panels** (ovarian, pancreatic,
prostate, biopsy breast), whose cohorts cannot support a screening claim.

## Design

- **Type:** prospective observational cohort, single site.
- **Population:** adults with a routine lab report or screening mammogram in the
  prior 90 days. Exclusions: known active cancer; inability to consent.
- **Procedure:** after consent, the research team enters the values from the
  existing report and history into Oncovision offline. Output is stored, sealed from
  the care team, and not returned to the participant.
- **Follow-up:** record review at 12 months for any diagnosis in scope (breast
  cancer; liver disease), using the standard-of-care work-up the participant would
  have had anyway.

## Endpoints

- **Primary (breast):** among participants below the rule-out cut, the share later
  diagnosed with breast cancer within 12 months, against the retrospective promise
  of catching 95% of cancers.
- **Primary (liver):** discrimination (AUC) for a liver disease diagnosis within
  12 months, and gain over an age-and-sex model on the same participants.
- **Secondary:** calibration; agreement by race and ethnicity; proportion of reports
  from which the required inputs could be read.

## Sample size (to be finalised with a statistician)

Breast cancer within a year of a screening mammogram occurred in about 0.49% of BCSC
mammograms. Confirming that a rule-out cut catches at least 90% of cancers with a
useful interval needs on the order of 100 cancers, which at that rate means roughly
20,000 screened women followed for a year. That is the honest reason a single-site study is more
likely to be a feasibility study — measuring recruitment, data completeness and
agreement — than a definitive test, and the protocol should say so.

## Risks and protections

- Minimal risk: no intervention, no change to care, output not disclosed.
- Data: de-identified at entry; stored on institutional systems; no data sent to the
  public web service.
- Incidental concern: because output is sealed, there is no duty-to-warn trigger
  from the tool; any clinically concerning *lab value* is already handled by the
  standard-of-care team that ordered it.

## Roles

- **Principal investigator (required for IRB submission):** _[to be confirmed]_
- **Student researcher:** _[name]_
- **Statistician:** _[to be identified]_

## Before submission

1. Confirm PI and institutional IRB pathway.
2. Statistician review of the sample-size section.
3. Freeze the model versions and thresholds under test, with a commit hash, so the
   prospective result describes exactly the models that were evaluated.
