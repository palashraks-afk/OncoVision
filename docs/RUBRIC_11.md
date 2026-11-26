# The 11-criterion rubric (set 26 November 2026, before the red-flag-labs study is run)

Palash Rakshit asked for every criterion at 9 or above. The anchors below are fixed now so that a score cannot be
nudged afterwards. Scores are my own judgement, given with the reason and the evidence; where a criterion needs
something only a person can supply (peer review, a clinician, real users), the ceiling says so.

**The brief this is graded against (Palash's words, 26 Nov 2026):** use the easy-to-find, ordinary lab reports and
other non-specific findings people already have to find cancer risk, and find both a societal and a scientific discovery.

| # | Criterion | 5 | 7 | 9 | 10 |
|---|---|---|---|---|---|
| 1 | Impact | Interesting to researchers | A clear, quantified problem many people have, with a named group that could act | A national-scale, quantified gap with a concrete, low-cost action and a named decision-maker | Shown to change practice or policy (needs uptake: not reachable by analysis alone) |
| 2 | Creativity | Standard analysis of a standard question | A fresh framing of a known question | A design that answers a question existing data designs cannot | Opens a new line of work others adopt |
| 3 | Novelty | Done before in similar data | Done before, but not in this population or with this design | No prior study found in recorded searches for the core question, with the closest work named | Confirmed new by independent experts |
| 4 | Fit to the brief | Uses routine data or finds risk, not both | Routine reports and risk, one discovery | Routine reports plus non-specific findings, one societal and one scientific discovery | Same, and the app puts it in a person's hands |
| 5 | Scientific rigor | Single analysis, no pre-registration | Pre-registered, CIs, some checks | Pre-registered, replicated in an independent dataset, negative control (other-cause death), survey design honoured, every miss reported | Plus an independent human re-analysis |
| 6 | Depth of real data | One small dataset | One national dataset | Several national datasets, decades of follow-up, measured (not self-reported) labs | Plus registry cancer diagnoses |
| 7 | Feasibility | Needs new infrastructure | Needs moderate change | Uses tests already done and guidance already written; the action is a referral or a letter | Already being done somewhere because of this |
| 8 | Profile fit (biomedical) | Generic data science | Health data science | Clinical laboratory medicine, haematology, gastroenterology, epidemiology together | Plus clinician co-author |
| 9 | Visual and interactive | Static tables | Figures | Interactive app a non-expert can use, plus publication-quality figures | User-tested with real people |
| 10 | Technical depth | Off-the-shelf model | Careful statistics | Survey-weighted estimation, competing-risk outcomes, external validation, reproducible pipeline with tests | Plus methods others reuse |
| 11 | Uniqueness | Many similar projects | Few similar projects | No similar student or published project found | Recognised as distinctive by outside judges |

Rules: a 10 needs evidence from outside this repository, so the maximum reachable by my own work is 9. If a criterion
stays below 9, the report says exactly what is missing and who could supply it.

## Grades for angle G, "A red flag already on the lab report" (26 November 2026, after two improvement rounds)

Work: `docs/RED_FLAG_LABS_PREREG.md` (with four addenda), `experiments/red_flag_labs.py`, `tests/test_red_flag_labs.py`,
`docs/red_flag_paper/Red_Flag_On_The_Lab_Report.pdf`, app page `/redflag`, independent re-check `docs/independent_check/red_flag/`.

| # | Criterion | Round 1 | Final | Evidence, and what is missing for a 9 or 10 |
|---|---|---|---|---|
| 1 | Impact | 8 | **8** | 2.1 million older Americans with guideline-defined IDA, about 80% untreated in two surveys 17 years apart; cheap actions named (reflex ferritin, record flag, patient page). Not 9: the survey never asks about endoscopy, so the investigation gap is inferred, and the cancer signal is not specific. Needs measured endoscopy follow-up (claims or record data, a person must get access) or a lab or health system adopting reflex ferritin. |
| 2 | Creativity | 8 | **9** | Uses a test-everyone survey to answer what record studies cannot (people never tested for a reason; untreated count). |
| 3 | Novelty | 8 | **8** | Societal question: no prior study found in 13 recorded Europe PMC queries and 3 web searches; closest work named. Scientific question: anaemia and cancer is well studied. Half new, so 8. |
| 4 | Fit to the brief | 9 | **9** | Routine report values plus a non-specific finding (anaemia); one societal and one scientific result; an app page. Not 10: the scientific result is a qualified negative. |
| 5 | Scientific rigor | 8 | **9** | Pre-registered with dated addenda; replicated in an independent era (bar met); negative control and competing-risks model; survey design; every miss reported; independent re-analysis by a second AI agent. Not 10: no human re-analysis. |
| 6 | Depth of real data | 9 | **9** | Three NHANES samples, 1988-2020, measured haemoglobin and ferritin, deaths through 2019. Not 10: no cancer registry. |
| 7 | Feasibility | 9 | **9** | Tests already done; guideline already written; the action is a referral. |
| 8 | Profile fit (biomedical) | 9 | **9** | Laboratory medicine, haematology, gastroenterology, survey epidemiology. Not 10: no clinician co-author. |
| 9 | Visual and interactive | 9 | **9** | `/redflag`: report-style reference bars, guideline verdict, questions to copy; four figures. Not 10: not tested with real users. |
| 10 | Technical depth | 7 | **9** | Design-based (Taylor-linearised) estimation, tested against a hand calculation; competing-risks multinomial model; external replication; reproducible fetch and analysis scripts. |
| 11 | Uniqueness | 8 | **9** | No similar student or published project found in the recorded searches. Not 10: needs outside judges. |

**Nine of eleven at 9 or above. Impact and novelty stay at 8**, and more analysis of public data cannot honestly move them:
impact needs measured follow-up (endoscopy) or real-world uptake, and novelty is held back because the scientific half is
known territory. Both are written down here rather than rounded up.
