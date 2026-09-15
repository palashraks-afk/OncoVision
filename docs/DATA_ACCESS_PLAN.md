# Data access plan: the gaps public data cannot close

**Status: a planning document, nothing submitted.** Every dataset below needs a named
investigator to apply. It exists so that one conversation can decide which applications
to make, in what order.

The public data search behind this was broad: PLCO, UK Biobank, All of Us, BCSC's public
and research datasets, UKCTOCS, ERSPC, PCPT, China Kadoorie Biobank, pancreatic
new-onset-diabetes cohorts (UK-EDI and others), hepatitis B and hepatitis C surveillance
cohorts, and Zenodo, Dryad, Figshare and PhysioNet. Nothing freely downloadable carries
the measurements and outcomes these gaps need. What remains are the controlled-access
sources below.

## Gap by gap

| Gap | What would close it | Source | Access | Rough effort |
|---|---|---|---|---|
| Ovarian and prostate rest on case-control data | CA-125 and PSA in a screening population, with later cancer outcomes | **PLCO** via NCI CDAS: about 78,000 women with CA-125 screens; 76,682 men, 35,875 of them screened with yearly PSA | Project proposal on CDAS, named investigator, data use agreement. No fee. Most requests are approved. | Weeks |
| Breast panel uses banded age and BMI; two groups too small to measure | Exact age and BMI, more cancers in small groups | **BCSC research data** (the public release is banded for privacy) | Data request to the consortium with a named investigator | Months |
| Liver panel predicts liver disease, not liver cancer | Routine liver blood tests with later liver-cancer diagnoses in ordinary adults | **UK Biobank** (about 500,000 adults, blood chemistry at baseline, cancer registry follow-up) | Application plus access fee | Months |
| Several panels could use one larger, more recent population | Blood tests and registry-linked cancer outcomes, all sites | **UK Biobank**, or **All of Us** (US electronic health records, including lab values) | UK Biobank: application and fee. All of Us: registered-tier access through an institution with a data use agreement. | Months |
| No lab-report panel screens for a named cancer (lung withdrawn 2026-09, bowel and general before it) | Routine blood work drawn BEFORE diagnosis, with site-specific cancers recorded afterwards, so the target stops being a lifetime diagnosis in survivors | **HRS 2016 Venous Blood Study**: about 9,900 US adults over 50, with a complete blood count and metabolic panel (albumin, ALT, alkaline phosphatase and more), followed for new cancers in the 2018, 2020 and 2022 waves. **CHARLS** (China, adults 45+): complete blood count, glucose, BUN, creatinine, CRP and lipids in 2011 and 2015, with new cancer diagnoses self-reported in every later wave | HRS: Sensitive Health Data order form through the HRS Data Portal. CHARLS: request through the study's own site. Neither needs a consortium proposal | Weeks |
| Pancreatic has no screening population | People at raised risk followed with blood tests | **High-risk surveillance consortia** (familial pancreatic cancer and new-onset diabetes cohorts) | Research collaboration, not a download | Long term |

## Suggested order

1. **PLCO first.** It closes the largest gap (two panels), costs nothing, and the draft
   request is already written: `docs/PLCO_CDAS_PROPOSAL.md`.
2. **BCSC research data second.** It would sharpen the one panel that already beats age
   alone on a screening population, and settle the two unmeasured groups.
3. **UK Biobank or All of Us third**, if the project continues past a first paper. Either
   could turn the liver panel toward liver cancer, and test several panels at once on a
   population larger than anything used so far.

## What stays true regardless

Pancreatic cancer has no screening programme anywhere, so no dataset will turn that panel
into a screening test. It stays labelled for people already being investigated.

None of these datasets replaces a prospective study on real patients. That needs the
protocol in `docs/PROSPECTIVE_STUDY_PROTOCOL.md` and ethics approval.
