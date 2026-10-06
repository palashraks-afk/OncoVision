# Novelty and reference check, 5 October 2026

## Reference check

Each reference was looked up by title in Crossref (api.crossref.org). Results are in
`docs/paper/reference_check.json`.

| Reference | Result |
|---|---|
| Hippisley-Cox & Coupland, QCancer. *BMJ Open* 2015;5:e007825 | Found, matches (DOI 10.1136/bmjopen-2015-007825) |
| Goff et al., ovarian cancer symptom index. *Cancer* 2007;109:221-227 | Found, matches (DOI 10.1002/cncr.22371). The search engine's top hit was an editorial commentary (Markman, 110:226-227), which is a different item |
| Collins et al., TRIPOD+AI. *BMJ* 2024;385:e078378 | Found, matches (DOI 10.1136/bmj-2023-078378). Two other BMJ items about it have similar titles |
| Wilson, 1927. *J Am Stat Assoc* 22:209-212 | Found, matches |
| Flesch, 1948. *J Appl Psychol* 32:221-233 | Found, matches |
| NICE NG12 | Not in Crossref. Real, and NICE's own site was blocked to automated access. Check by hand |
| WHO, haemoglobin and anaemia, WHO/NMH/NHD/MNM/11.1 | Not in Crossref. Check by hand |
| NCHS linked mortality files | A data product, not an article. Check the NCHS site |
| Kincaid et al., 1975 Navy report | A report, not in Crossref. Check by hand |

## Is it new? What a short search found

Searches on 5 October 2026 for patient-facing cancer symptom checkers and risk calculators found:

- **QCancer** is online and public (qcancer.org). It turns symptoms and risk factors into a
  percentage risk for several cancers. It does not read a pasted lab report or explain a
  guideline in its own words.
- **Cancer Research UK** publishes an interactive NG12 symptoms guide, aimed at health
  professionals, not patients.
- **A cancer decision-support tool** is built into UK GP systems, for clinicians.
- **Commercial calculators** exist (for example one from Everlab). I did not assess them.

I found no tool that does what this one does (read a lab report with symptoms, apply a published
referral guideline, show the guideline page behind each alert, in plain language). A short web
search is **not** evidence that none exists, so the paper does not claim novelty. It claims a
patient-facing implementation and an honest measurement.

## What I could not check

NICE's current NG12 text. The NICE website refused automated access (403), and a second
fetcher failed on authorisation. Secondary sources report that NG12 was updated in
August 2023 (bowel pathway, using a quantitative FIT) and October 2023 (a change to the
definition of the suspected-cancer pathway). Those are reported, not read at source.
