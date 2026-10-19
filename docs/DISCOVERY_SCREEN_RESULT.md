# Result of the discovery screen

Run on 18 October 2026 under `docs/DISCOVERY_SCREEN_PREREG.md` (with Addendum 2, written before any result existed). The
hit lists were committed with their hashes before replication was run (commits `e14ef39` and `6dd5457`). Nothing below was
rescued after the fact, and no threshold moved.

## The answer

**No question in the screen passed all five stages.** Among the 79 items that the National Health Interview Survey asked in
a stable form across 1997-2002 and 2005-2009, none adds to cancer-death risk beyond age, sex, smoking and body mass index in a way
that replicates, is specific to cancer, and is large enough to matter. The largest replicated AUC gain was +0.0016, against a
bar of 0.005.

This agrees with the earlier tests in this project (routine blood work, 337 NHANES variables, activity and checkup data), which
also found that these few free facts carry nearly all of the free signal.

## Every stage, all adults

Adults 35-84 without earlier cancer, interview years 1997-2009: 200,330, with 5,930 cancer deaths within ten
years. Discovery years 1997-2002: 110,282 adults, 3,379 cancer deaths. Replication years 2005-2009: 73,303 adults, 2,076 cancer
deaths.

| Stage | Count |
|---|---|
| Columns in the questionnaire files | 2,800 |
| Excluded by name (survey machinery, baseline, cancer, screening, race) | 288 |
| Excluded by label (cancer, screening, baseline, race) | 508 |
| No label in the NCHS programs | 178 |
| Asked in fewer than two discovery or two replication years | 1,636 |
| Asked in enough years but with changed wording (not screened) | 111 |
| **Candidate items screened** | **79** |
| Passed discovery (false discovery rate q < 0.05) | 13 |
| Replicated (p < 0.05 after Bonferroni over the hits, same direction) | 3 |
| Cancer-specific (cancer gain at least half the other-cause gain) | 0 |
| **Discoveries** | **0** |

All 79 fits converged. Before the run, the test was checked on shuffled copies of the items: null p-values averaged
0.47, so it neither invents nor hides associations.

## All 13 discovery hits, and what happened to each

AUC gains are for ten-year cancer death, trained on the discovery years and scored on the replication years, over the baseline
(age spline, sex, smoking, years since quitting, BMI).

| Item | Discovery p | Replication p | Same direction | AUC gain, cancer (95% interval) | AUC gain, other-cause death | Outcome |
|---|---|---|---|---|---|---|
| freq drank alcohol: days in past year | 1.1e-07 | 6.3e-03 | yes | n/a | n/a | not replicated |
| freq drank alcohol: days per week | 2.7e-06 | 5.7e-03 | no | n/a | n/a | not replicated |
| average # drinks on days drank | 1.5e-05 | 8.9e-03 | yes | n/a | n/a | not replicated |
| number of days had 5+ drinks past year | 4.6e-05 | 7.2e-03 | yes | n/a | n/a | not replicated |
| have more than one job | 1.2e-04 | 1.6e-02 | yes | n/a | n/a | not replicated |
| lost all upper & lower natural teeth | 1.6e-04 | 6.1e-07 | yes | +0.0014 (+0.0006 to +0.0022) | +0.0029 | replicated, **not** cancer-specific |
| ever been told you had emphysema | 3.3e-04 | 3.0e-02 | yes | n/a | n/a | not replicated |
| time since last saw a dentist | 8.2e-04 | 4.4e-05 | yes | +0.0016 (-0.0001 to +0.0032) | +0.0063 | replicated, **not** cancer-specific |
| ever worked | 8.7e-04 | 4.2e-02 | yes | n/a | n/a | not replicated |
| work status: last week, past 12 m | 9.5e-04 | 6.8e-02 | no | n/a | n/a | not replicated |
| number of work loss days, past 12 months | 2.4e-03 | 9.7e-01 | yes | n/a | n/a | not replicated |
| donated blood since march 1985 | 5.1e-03 | 6.5e-03 | yes | n/a | n/a | not replicated |
| told you had liver condition, past 12 m | 7.3e-03 | 6.6e-04 | yes | +0.0002 (-0.0010 to +0.0012) | +0.0023 | replicated, **not** cancer-specific |

Three items replicated: losing all natural teeth, time since last saw a dentist, and a liver condition in the past year. In
every case the gain for **other-cause** death was larger than for cancer, and every cancer gain was well below the 0.005 bar
(coverage of cancer deaths at 20% invited moved by +0.26, +0.33 and +0.07 points against a bar of 1 point). They behave like
markers of general ill health, not of cancer.

Near misses, reported as misses: four alcohol items passed discovery very strongly (p down to 1.1e-07). In the replication years
all four had p between 0.006 and 0.009 and three of them pointed the same way, but the registered cut was 0.0038 (0.05 over 13
hits), so none replicated. This is a hint, not a finding, and it is not claimed.

## Equity screens (who the free model fails)

| Group | Adults | Cancer deaths | Screened | Skipped by size floors | Passed discovery | Replicated | Discoveries |
|---|---|---|---|---|---|---|---|
| Non-Hispanic Black | 28,798 | 1,045 | 69 | 10 | 2 | 1 | 0 |
| Hispanic | 29,315 | 511 | 67 | 12 | 0 | 0 | 0 |
| Other | 9,016 | 217 | 0 | 79 | 0 | 0 | 0 |

The registered equity bar is an AUC gain of at least 0.01 with the interval above zero. In Black adults one item replicated
(days per week drinking alcohol, replication p = 5.1e-04) and was cancer-specific, but its gain was
+0.0025 (interval -0.0029 to +0.0078), which misses the bar and includes zero. The "other" group
could not be screened: it has 3,840 adults and 107 cancer deaths in the discovery years, below the 5,000-person floor.
Replication has only 397 cancer deaths in Black adults and 194 in Hispanic adults, so a miss here means "not shown", not "absent".

## What this does and does not show

- It shows that, among questions asked consistently across both periods, nothing beyond age, sex, smoking and BMI helps predict
  cancer death in a way that survives replication, with a rule fixed in advance. That is a negative result with a fixed
  false-discovery rule and a registered replication rule behind it.
- It does **not** cover the whole questionnaire. Of 2,800 columns only 79 were eligible. Most of the rest are
  rotating modules and items renamed in the 2004 redesign. The 111 items lost only to changed wording are listed in
  `experiments/discovery_screen_excluded.csv`. The pre-registration expected "300+" items; the honest figure is 79.
- It is one survey, self-reported answers, death from any cancer (not a site), and ten-year follow-up. It finds predictors, not
  causes, and says nothing about biomarkers or tests that cost money.
- It does not reach the project goal of a discovery to make the app spread. The routes that could still lift novelty and impact
  need a person: see `docs/NEXT_STEPS_FOR_RAHUL.md` and `docs/CLINICAL_DATA_ACCESS.md` (for example the NCHS Research Data Center
  for NHIS linked to cancer registries).

## Reproduce

```
python fetch_nhis_full.py                                       # about 5 minutes
python experiments/discovery_screen.py discover                 # then commit the hit list and hash
python experiments/discovery_screen.py replicate
python experiments/discovery_screen.py equity 1 discover        # and equity 2, 3; then replicate the same way
```

Outputs are `experiments/discovery_screen_*`. The tests are in `tests/test_discovery_screen.py`.

## Disclosure

This analysis was written and run by an AI assistant (Claude) at the request of Palash Rakshit. No human second analyst has
re-run it, and no clinician has reviewed it. An AI cannot be an author.
