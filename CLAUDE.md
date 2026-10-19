# Oncovision: instructions for Claude Code

Read `docs/HANDOFF.md` first (objective, state, next job, setup on a Mac), then `docs/OBJECTIVE.md`.

The user is **Palash Rakshit** (GitHub palashraks-afk). Use they/them if pronouns come up. Write plainly, in short sentences,
without hype; they read through typos and want fast iteration, not ceremony.

**Goal:** free-information cancer risk to catch cancer earlier and cheaper, at scale; research scoring novelty 5, impact 5,
quality 4 on the 1/3/5 scale in `docs/ANGLE_ASSESSMENT.md`, judged honestly, never moved to fit.

**Non-negotiables**

1. Pre-register before running; keep thresholds fixed; report every miss.
2. Measure everything you claim; no lives-saved claims; verify law and source text at the primary source (enacted, not introduced).
3. Never drop people by what happens after the index date. Race is a stratifier, never a feature.
4. Disclose AI use; do not assert Dr. Chavan's role or authorship; do not invent stories or history; never put secrets in the repo.
5. Use `git commit -F <file>` and the attribution trailer your tool instructs. Run `tools/audit.py`, `tools/sync_docs.py --check` and
   `pytest tests -q` before committing. Do not commit large downloaded data (see `.gitignore`).
6. Long jobs run in the background and are polled; do not block on multi-minute commands. Never kill the user's browser.

**Status:** the discovery screen has been run (18 Oct 2026) and is a clean negative; see `docs/DISCOVERY_SCREEN_RESULT.md`. Do not
re-run it expecting a different answer. What remains needs a person (data access, a human second analyst, a clinician): see
`docs/HANDOFF.md` section 6, `docs/NEXT_STEPS_FOR_RAHUL.md` and `docs/CLINICAL_DATA_ACCESS.md`.
