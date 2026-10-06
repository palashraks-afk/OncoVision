# The twelve open problems: what was done, what only a person can do

Written 5 October 2026 after the fix pass. "Fixed" means fixed and checked. "Reduced" means
improved but not gone. "Needs a person" means I cannot do it from here.

| # | Problem | Status | What was done | What is left |
|---|---|---|---|---|
| 1 | No clinician has checked the rules | **Needs a person**, packet ready | `docs/clinician_review/Rule_Review_Packet.pdf`: every rule in plain words with tick boxes, eight design questions, sign-off. `docs/MESSAGE_TO_DR_CHAVAN.md` | A clinician has to do the review |
| 2 | Old, foreign guidance | **Reduced** | New US module (`/screening`) built from the five USPSTF recommendation pages, each read on 5 Oct 2026 and tested at every age boundary. Bowel results now say NICE changed its pathway in 2023. Women under 50 with low iron get a periods note. | NICE's website blocked automated access, so I could not read the current NG12 myself. Someone must compare the 34 rules with it (the packet's first question) |
| 3 | No real patient data | **Needs a person** | `docs/CLINICAL_DATA_ACCESS.md`: sources, what each has, an order to try, what to write | Applications need you, an institution, or both |
| 4 | Dr. Chavan has not agreed | **Needs a person** | Message drafted. Surname fixed (Palash Rakshit) | You send it |
| 5 | Lab alerts do not find cancer | **Reported, not fixable** | The tool's wording now says so. Paper reports it and the pooled estimate differs from NHANES III (p = 0.0005) | The property is real: guideline thresholds are for people with symptoms |
| 6 | No stronger model | **Tested again, not found** | Pre-registered a curated 29-variable set. AUC 0.752 vs 0.743, gain +0.009 (-0.007 to +0.027), missed the bar. | A stronger model needs symptoms and diagnoses, not more survey data |
| 7 | Too many alerts at the oldest ages | **Reduced** | Periods note for women under 50. Lab-only caveat. The rule itself is the guideline's, so only a clinician should change it. | Packet question 4 asks whether lab-only alerts should be shown |
| 8 | References unchecked | **Reduced** (5 of 9 items checked) | Looked up in Crossref: Hippisley-Cox, Goff, Collins (TRIPOD+AI), Wilson, Flesch. All match. | NICE NG12, WHO anaemia report, NCHS mortality files and Kincaid 1975 are not in Crossref; check by hand |
| 9 | Explanation was a guess | **Tested** | Survey year is predicted from the missing-value flags alone with AUC 0.927. The same model on a random half of all years scores 0.755 vs 0.629 across eras. | The explanation is now supported, not proven |
| 10 | Novelty not checked | **Checked, claim softened** | `docs/NOVELTY_AND_REFERENCES_CHECK.md`: QCancer is public online; CRUK has a clinician-facing NG12 guide; commercial calculators exist | Paper no longer implies novelty beyond what was found |
| 11 | Readability is only a score | **Needs a person** | `docs/COMPREHENSION_STUDY_PROTOCOL.md`: three fixed outputs, five questions, scoring key, bars | Needs readers and an ethics determination |
| 12 | NHANES III went the other way | **Tested** | Heterogeneity test: z = -3.47, p = 0.0005. The paper now says the effect is not stable across eras | The honest reading is "alerts never identified cancer deaths", not "alerts protect" |
