# Handoff: everything needed to continue on another machine (written 18 October 2026)

For Palash Rakshit, to run on a Mac with Claude Code. Open the repo there and Claude Code reads `CLAUDE.md`
automatically; this file is the full detail. Nothing important lives only on the old Windows machine except
large downloaded data files, which the scripts below rebuild.

## 1. What you want (your words, kept)

- Cancer risk assessment and screening **to catch cancer earlier and cheaper**, using **free information people already have**,
  and for it to actually reach and help people at scale. The work must stay on "cheap cancer risk prediction".
- Research a mentor and college admissions would respect: **novelty 5, impact 5, paper quality 4** (the 1/3/5 scale
  in `docs/ANGLE_ASSESSMENT.md`). "Keep going until you can, then come back." Honest scores only; never moved to fit.
- A **discovery** that helps reach that goal and could give the app a reason to spread. You asked for 5 in every category.
- Also in the repo: a Congressional App Challenge option (deadline Mon 26 Oct 2026, 12:00 ET). The 2026 rules require
  the student to write the code (AI only for disclosed, specific parts). Most of this repo was AI-written, so it should not
  be entered as your own work unless you write the core code. Check the rules PDF yourself before deciding.

## 2. Where things stand

| Work | Where | State |
|---|---|---|
| Free information, cost-matched invitation | `docs/cancer_age_paper/` (PDF built by `build_age_paper.py`) | Done. Novelty 4, impact 4, quality 4-5. Fairness bar missed (reported). |
| Medicare multi-cancer test law | `docs/mced_paper/` (PDF built by `build_mced_paper.py`) | Done. Novelty 5, impact 4, quality 4. Law covers ~12% of Medicare-age cancer deaths in 2029; free-question triage ~2.3x yield. |
| Free-data ladder (activity, checkup) | `experiments/free_data_ladder.py`, `docs/FREE_DATA_LADDER_PREREG.md` | Done. Negative for cancer death. |
| App page for the above | `frontend/src/app/multicancer/` (runs in the browser) | Done, tested in a browser. |
| Independent AI re-checks | `docs/independent_check/` | Done. Found and fixed the introduced-vs-enacted law error. |
| **Discovery screen** | `docs/DISCOVERY_SCREEN_PREREG.md`, `fetch_nhis_full.py`, `experiments/discovery_screen.py` | **Pre-registered and committed. Data downloader worked. The screening script was written but has NEVER completed a run (the Windows machine crashed while it ran). It is untested: expect bugs. Nothing from it has been looked at.** |

Tag `locked-2026-10-18` marks the repo state before the discovery screen.

## 3. The next job: run the discovery screen (steps for the Mac)

1. Set up Python 3.11+, then `pip install -r requirements.txt joblib scipy matplotlib requests pymupdf`.
2. `python fetch_nhis_full.py` downloads every NHIS adult questionnaire variable 1997-2009 (about 20 minutes to an hour;
   it resumes; files go to `data/audit/nhis_full/`, git-ignored). It needs internet access to ftp.cdc.gov.
3. `cd experiments && python discovery_screen.py` runs stages 1-5 of the pre-registered screen (all adults).
   It is expected to take a long time (hundreds of logistic fits across 4 processes); run it in the background and poll.
   Then for the equity screen: `python discovery_screen.py equity 1` (Black adults), `equity 2` (Hispanic), `equity 3` (other).
4. **Test the script on a handful of items first** (it was never run end to end): load the data, call `screen_one` on three
   items, check the baseline and LR test behave, then run the whole thing. Fix bugs freely; do **not** change the pre-registered
   thresholds, the discovery/replication split, or the hit rule. Record any departure in the prereg addendum, dated, before looking at results.
5. Outputs: `experiments/discovery_screen_all.csv`, `..._hits.csv` (frozen before replication), `discovery_screen_result.json`.
   Report every stage count and every miss. If nothing passes all stages, say so: "the free-information ceiling is firm" is the result.
6. If something does pass: independently re-run it with fresh code (the earlier second-analyst agent pattern), check it is not
   frailty (cancer gain at least half the other-cause gain, already in the script), check it is not a coding artifact
   (look at the item's categories and years), then write it up as a short paper using `docs/cancer_age_paper/build_age_paper.py`
   as the template, update `docs/ANGLE_ASSESSMENT.md` honestly, commit, push.

## 4. Rules the project holds itself to (do not break these)

Full text in `docs/handoff/oncovision-project-notes.md` (copied from the assistant's notes) and `docs/OBJECTIVE.md`. The short version:

- **Pre-register, then obey it.** Bars and analyses are committed before running; misses are reported as misses.
- **Measure, don't assume.** Nothing is claimed that was not measured. No lives-saved claim without trial evidence.
- **Never drop people by what happens after the index date**; keep non-cancer deaths as non-cases.
- **Check the enacted text, not the introduced bill** (a real error caught here).
- **Disclose AI use**; an AI cannot be an author; a human second analyst and a clinician are still needed; mentor Dr. Rishikesh Chavan
  has not agreed to anything, so never assert his role or authorship.
- Race is a stratifier, never a feature. No fabricated history, no invented personal stories, no secrets in the repo.
- Commit with `git commit -F <file>` and end with the attribution trailer the tool tells you to use.
- Before committing: `python tools/audit.py`, `python tools/sync_docs.py --check`, `python -m pytest tests -q` (121 pass, 1 skipped on Windows).

## 5. Setup differences on a Mac

- The paper builders print PDFs with headless Chrome at a Windows path
  (`C:\Program Files\Google\Chrome\Application\chrome.exe`). On macOS change it to
  `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` in `docs/*/build_*.py`, or use any HTML-to-PDF tool.
- Scripts use `os.path`, so slashes are fine. `fetch_nhanes_activity.py` needs about 900 MB of NHANES accelerometer files
  already in `data/audit/pax/` (git-ignored; re-download from the URLs in the script history if you need to rerun it; the
  summary `data/nhanes_activity.csv` is committed, so you usually do not).
- Frontend: `cd frontend && npm install && npm run dev` (Next.js 16, port 3000).
- Backend: `pip install -r backend/requirements.txt`, deployed on Render (`render.yaml`).
- Never run `taskkill` on Chrome (Windows lesson); on a Mac, do not kill the browser the user is using.

## 6. Other live facts worth knowing

- Frontend: https://oncovisionai.vercel.app, backend: https://oncovisonai.onrender.com (note the spelling: one "i" short).
- NHANES III, NHIS and linked-mortality files all come from public ftp.cdc.gov paths; no accounts needed.
- Data that needs a person (cannot be done by an AI): UK Biobank, PLCO via Dr. Chavan, CHARLS registration, the NCHS Research Data
  Center application for NHIS linked to cancer registries (the single step most likely to lift impact and novelty), a human second
  analyst re-run, a clinician co-author. See `docs/CLINICAL_DATA_ACCESS.md` and `docs/NEXT_STEPS_FOR_RAHUL.md`.
- The Nimble API key you once pasted in chat must be rotated; it is not in the repo.
- A `.gitignore`d `.claude/` folder may exist locally; it is session tooling and not part of the project.

## 7. How to talk to Claude on the Mac

Paste: "Read CLAUDE.md and docs/HANDOFF.md, then do section 3 of the handoff: test discovery_screen.py on a few items, fix bugs without
changing the pre-registered rules, run the full screen in the background, and report every stage honestly." That is enough.


## Update, 18 October 2026 (Mac session, before any screen result)

Section 3 was followed. The screening script needed fixes before it could be trusted; they are recorded in **Addendum 2** of
`docs/DISCOVERY_SCREEN_PREREG.md` and none changes a registered rule. The commands are now
`python experiments/discovery_screen.py [equity 1|2|3] discover` then, after the hit list is committed, `... replicate`.
Setup notes: the install line also needs `pip install -r backend/requirements.txt` for the tests; xgboost needs an OpenMP library
on a Mac (not used by the screen). The screen takes minutes, not hours. The registered rules leave 79 candidate items, not 300+.

Result (18 October 2026): the screen ran and nothing passed all stages. 79 items screened, 13 passed discovery, 3 replicated, none
cancer-specific; the largest replicated AUC gain was +0.0016 against a bar of 0.005. See `docs/DISCOVERY_SCREEN_RESULT.md`. Section 3
is done; what remains needs a person (section 6).
