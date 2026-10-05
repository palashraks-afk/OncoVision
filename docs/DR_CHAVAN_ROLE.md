# Dr. Chavan's role: the best honest case, and how this counts as research

This is a plan to discuss with him, not something he has agreed to. Every part of his role
below is a request, and he may say no to any of it.

## How this counts as research

Research has a question, a method written before the result, data, an analysis, a
conclusion that can be wrong, and a way for others to check it. This project has those:

- **Questions** that can fail: does the tool match a clinician applying the guideline? How
  many people would it alert? Can a person without medical training read it?
- **Methods fixed first.** `docs/NAVIGATOR_VALIDATION_PROTOCOL.md` sets the bars before the
  clinician study. Earlier work in the repository did the same and reported what failed.
- **Real data and real analysis.** NHANES (public, de-identified) for the alert burden and
  a measured reading level; the full history is in the repository's commit log.
- **Checkable.** Every result has a script that regenerates it, and 73 automated tests.

What would make it *published* research is a clinician-validated result written up and
posted as a preprint, then submitted. A preprint is a legitimate public record. Say "posted
a preprint" and not "published" until a journal accepts it.

## What would be Dr. Chavan's best role

The most valuable role is one where a clinician is **indispensable** and the student's work
is real. In this project that is the validation, which only a clinician can do.

| Role | What he does | Time |
|---|---|---|
| **Clinical reviewer of the rules** | Reads the rule file against the current guideline and signs off, changes or removes each rule. The tool says "not yet reviewed" until he does. | A few hours |
| **Rater or recruiter of raters** | Rates vignettes himself, or brings two colleagues, blind to the tool | A few hours each |
| **Mentor on the question** | Tells you which clinical questions matter and which findings a clinician would find believable | A regular short meeting |
| **Ethics and data guidance** | Confirms with his institution whether any step needs an ethics determination | One conversation |
| **Senior author, if he qualifies** | See authorship below | Writing and review |

The student's role is real and distinct: transcribing and structuring the guideline,
building and testing the engine, designing and running the burden and readability analyses,
writing the protocol, and drafting the paper. A strong letter from him can then describe
specific things he saw you do, which only works if he is involved while it happens, so the
earlier he is in the work, the better.

## Authorship

The usual standard (the ICMJE criteria) is that an author made a substantial contribution,
helped draft or critically revise the paper, approved the final version, and is accountable
for it. If Dr. Chavan reviews the rules, rates vignettes and revises the manuscript, he
likely qualifies. Settle authorship order with him early and in writing, so nobody is
surprised later. It is his decision and yours, not an assumption.

## AI assistance has to be disclosed, and you must understand every part

Much of this was built with an AI assistant. The 2026 ICMJE rules say an AI cannot be an
author and that its use must be disclosed in the submitted work: name the tool and
version, say what it was used for, and confirm that the human authors reviewed and take
responsibility. Analysis and code go in the Methods; writing help goes in the
Acknowledgments. Check each journal's own policy as well.

This matters for you as much as for a journal. Dr. Chavan and any reviewer will ask how a
number was made. You should be able to open any script, say what it does, run it, and
explain why a result came out as it did. A practical way to get there: read
`backend/navigator.py` and `backend/navigator_rules.json`, run the tests, change one rule
and see which test fails, and write a page in your own words describing the project and its
limits. Ask for that page and it is yours to keep.

## For a college application

Describe it accurately. What admissions readers tend to find convincing is a specific,
honest account, not a large claim:

- **State what you did.** For example: "I built a tool that applies published cancer
  referral guidelines to symptoms and lab results, measured how many people it would alert
  and how readable it is, and wrote a protocol for a clinician validation study."
- **State what you did not do.** It has not been tested on patients and the rules have not
  been clinician-reviewed until he has done that.
- **Name the mentor's role exactly** and let him describe it in his own words.
- **Be open about AI use** if asked, and be able to explain the work without it.
- **Do not claim a publication, an IRB approval or clinical validation** that has not
  happened. Several things in this project failed its own tests and were withdrawn. That is
  a strength to describe, because it shows you can tell when something does not work.

Competitions and summer programmes have their own rules about human participants,
mentors and software projects. Check them before relying on this work for entry.

## What to bring to the first meeting

1. `docs/PAPER_ANGLE.md`, `docs/NAVIGATOR_VALIDATION_PROTOCOL.md` and this file.
2. The running tool, with its "research prototype" banner, and a case or two.
3. The three requests that need him: review the rules, help find raters, advise on ethics.
4. One honest sentence about the limits: it applies a guideline and does not discover one.

## If he says he does not have time

The work still stands as a documented student project, and the protocol and rule file are
written so another clinician could pick up the review. Look for a primary-care physician,
a geriatrician, or an adult oncologist, since the tool is aimed at adults and CHOC is a
children's hospital. The children's variant in `docs/CHILDREN_DIRECTION.md` is the part
closest to his own field.
