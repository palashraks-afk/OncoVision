# Independent re-analysis of the cancer-risk-age study (AI second pass)

Done by a separate AI agent that wrote its own spline, logistic, AUC and coverage code. This is **not** a
human second analyst; it is evidence that the code does what the paper says. Files: `part1_reproduce.py`,
`part1_result.json`, `part2_new_angles.py`, `part2_result.json`.

**Verdict: reproduced.** AUC age+sex 0.7562 (paper 0.756), F2 0.7854 (0.785), calibration slope 1.007 (1.003),
coverage at 20% invited 55.0% by risk against 47.6% by age (paper 55% / 48%). No fit/test leakage; no early
censoring among those counted as non-cases; outcome coding (ucod 2, malignant neoplasms) correct; dropping the
first 24 months, using strict < 120 months, and survey weighting change nothing material.

## Issues found
1. **Moderate-high.** Age-only is a weak comparator. Age plus never/former/current smoking (model F1) reaches
   AUC 0.7847 against 0.7854 for F2, so BMI, cigarettes per day and years since quitting add about 0.0007 AUC and
   about 0.4 coverage points. The extra deaths covered by the risk rule are all in smokers (current +212,
   former +13, never-smokers -113). The paper says the gain is mostly smoking; it should say it is essentially
   smoking status.
2. Low-moderate. Outcome is all cancer deaths; other-cause deaths count as non-cases. Observed/expected 0.95
   overall and 0.88 in ages 35-49.
3. Low-moderate, process. Dates in the pre-registration and paper were written ahead of the commit dates
   (commits are 7 Oct 2026) - corrected. The spline in the code differs from the pre-registered natural spline
   (4 df) - now disclosed. A docstring said "about 350,000 adults"; the analysed set is 183,585 - corrected.
   The test years were seen during development.
4. Low. Intervals ignore survey clustering; a dead duplicate `read_fwf` line; downloads use `verify=False`.

## Additional angles
- Model decay (run): discrimination stable (F2 over F0 about +0.03 AUC in every block); absolute risk drifts
  down (observed/expected 0.90 to 0.96 after 1999); the coverage advantage appears to shrink from about +8.8 to
  +6.3 points but that trend was not tested. Novelty low-moderate; data stop at interview year 2009.
- Lung vs non-lung (proxy only): the public file has no cancer site, so split by smoking status; the gain is the
  smokers. Moderate novelty as an honest framing only.
- Published relative risks plus SEER: rejected (would require typed-in numbers that cannot be validated here).
