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
| Ovarian has never been tested outside the hospital that produced it | The same decision, malignant against benign in a found mass, measured somewhere else | **No public cohort found.** The prostate half of this gap closed with PI-CAI (`fetch_picai.py`), which is a downloadable clinical spreadsheet of 1,500 worked-up men; searches for an equivalent adnexal-mass file with routine bloodwork returned papers, not data | Would need a collaboration, or PLCO's ovarian arm | Open |
| ~~Breast panel uses banded age and BMI; two groups too small to measure~~ **Closed, and not by asking** | Nothing. Cross-fitting over all 2.39M mammograms measured every subgroup (experiments/bcsc_subgroups_full.py), and banding was priced: re-banding cohorts that carry exact age and BMI costs at most 0.0035 AUC and nothing on average (experiments/banding_cost.py) | ~~BCSC research data~~ | Not worth requesting for resolution alone | — |
| Liver panel predicts liver disease, not liver cancer | Routine liver blood tests with later liver-cancer diagnoses in ordinary adults | **UK Biobank** (about 500,000 adults, blood chemistry at baseline, cancer registry follow-up) | Application plus access fee | Months |
| Several panels could use one larger, more recent population | Blood tests and registry-linked cancer outcomes, all sites | **UK Biobank**, or **All of Us** (US electronic health records, including lab values) | UK Biobank: application and fee. All of Us: registered-tier access through an institution with a data use agreement. | Months |
| No lab-report panel screens for a named cancer (lung withdrawn 2026-09, bowel and general before it) | Routine blood work drawn BEFORE diagnosis, with site-specific cancers recorded afterwards, so the target stops being a lifetime diagnosis in survivors | **HRS 2016 Venous Blood Study**: about 9,900 US adults over 50, with a complete blood count and metabolic panel (albumin, ALT, alkaline phosphatase and more), followed for new cancers in the 2018, 2020 and 2022 waves. **CHARLS** (China, adults 45+): complete blood count, glucose, BUN, creatinine, CRP and lipids in 2011 and 2015, with new cancer diagnoses self-reported in every later wave | HRS: Sensitive Health Data order form through the HRS Data Portal. CHARLS: request through the study's own site. Neither needs a consortium proposal | Weeks |
| Pancreatic has no screening population | People at raised risk followed with blood tests | **High-risk surveillance consortia** (familial pancreatic cancer and new-onset diabetes cohorts) | Research collaboration, not a download | Long term |

| Every panel reads ONE blood draw, so none can see a value moving | Repeat blood counts per person, with a later first cancer diagnosis. The level of a value carries little; the change from a person's own baseline is how the one deployed model of this kind (ColonFlag, AUC 0.74-0.82) works | **MIMIC-IV** (~300,000 patients, every lab result timestamped, diagnosis codes per admission) | PhysioNet credentialed account plus CITI "Data or Specimens Only Research" training. Free. Hours of training, then days for approval | Days |

The pipeline for that last row is already written and tested end to end on the
openly downloadable 100-patient demo: `fetch_mimic_trajectory.py` and
`experiments/trajectory_vs_snapshot.py`. It drops survivors, refuses labs drawn
within 90 days of the diagnosis, and refuses to report anything below the event
floor. On the full cohort it runs unchanged.

## Suggested order

1. **HRS 2016 venous blood first.** It is the only request here that could restore what the
   project lost when the lung panel was withdrawn: a lab-report panel that screens for a
   named cancer, built on blood drawn before the diagnosis rather than after it. Weeks, one
   form, no consortium proposal.
2. **PLCO second.** It closes the largest remaining modelling gap (ovarian and prostate in a
   screening population rather than a referred one), costs nothing, and the draft request is
   already written: `docs/PLCO_CDAS_PROPOSAL.md`.
3. **UK Biobank or All of Us third**, if the project continues past a first paper. Either
   could turn the liver panel toward liver cancer, and test several panels at once on a
   population larger than anything used so far.

**Not on this list any more:** the BCSC research file, priced above and not worth the wait.
What the prostate panel needed turned out to be free: PI-CAI's clinical marksheet is a
public download, and it gave the project its first external validation of a case-control
panel. Worth checking for an equivalent file before assuming any gap needs an application.

## What stays true regardless

Pancreatic cancer has no screening programme anywhere, so no dataset will turn that panel
into a screening test. It stays labelled for people already being investigated.

None of these datasets replaces a prospective study on real patients. That needs the
protocol in `docs/PROSPECTIVE_STUDY_PROTOCOL.md` and ethics approval.
