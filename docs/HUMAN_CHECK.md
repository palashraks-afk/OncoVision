# For the person checking this work

Thank you. An AI assistant wrote the code and text. A second AI agent re-ran the first version. **No person has checked it.**
This sheet lists what is worth an hour of your time.

## 1. Do the numbers reproduce? (about 10 minutes)

    pip install -r requirements.txt scipy
    python tools/check_headline_numbers.py

It re-runs the three red-flag analyses on the committed data and prints MATCH or DIFFERS for ten headline numbers. A match
shows the code gives the same answer again. It does **not** show the code is right.

## 2. Is the code right? Please look at these in particular

| Question | Where |
|---|---|
| Is guideline iron-deficiency anaemia defined correctly (haemoglobin below 13 men, 12 women; ferritin below 45)? | `experiments/red_flag_labs.py`, `flags` |
| Is the survey weighting and the interval method right? | same file, `ratio_est`; test in `tests/test_red_flag_labs.py` |
| Is "not treated" a fair use of the survey question (MCQ053, past 3 months)? | paper Sections 2.3 and 4.2 |
| Are people alive with short follow-up handled correctly, and non-cancer deaths kept? | `load_followup` |
| Are the iron and medicine name matches right? (one error was found and fixed) | `experiments/red_flag_angles.py` |
| Did any decision made after a result get presented as planned? | `docs/RED_FLAG_LABS_PREREG.md`, addenda in order; compare with `git log` |

## 3. Sources to confirm at the original

- AGA 2020 guideline: the ferritin cut-off and who the endoscopy recommendation covers.
- References whose author lists are incomplete in the paper (it says which).
- Medicare's ferritin price ($13.63, CPT 82728, 2026 Clinical Laboratory Fee Schedule).
- The closest earlier work named in paper Section 4.3: is anything closer missing?

## 4. Known weak points (please judge them)

- No endoscopy records: the work-up gap is inferred from treatment.
- The cancer follow-up rests on about 14 cancer deaths among people with the flag, and the signal is not specific to cancer.
- Several analyses were added after earlier results; they are labelled in the addenda.
- Data are cancer deaths, not cancer sites or diagnoses.

## 5. What to send back

Anything wrong or overstated, in any form. If you are willing to be named as having re-checked it, say so; otherwise you
will not be named.

## The Medicare law paper (`docs/mced_paper/`)

Please check: the law's age rule against the enacted text (govinfo, H.R. 7148, enrolled); `experiments/mced_law.py` (who is
covered); the replication (`experiments/mced_replication.py`: do 2010-2014 results in Table 1 reproduce?); and the claim
that the first people covered are mostly Medicare members aged 50-64 (survey-weighted about 61% in 2005-2009).
