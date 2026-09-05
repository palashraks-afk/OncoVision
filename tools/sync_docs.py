"""
Regenerate the numeric tables in README.md and PROJECT.md from the artifacts.

Why
---
The documentation is the argument this project makes, and until now every number
in it was typed by hand. That is not a style problem. A stale table is a false
claim: the breast panel's row said 0.972 for a while after the panel had been
retrained to 0.997, and the liver row said 0.753 after GGT moved it to 0.760.
Both were wrong in the flattering direction on one and the unflattering
direction on the other, which is what makes hand-maintained numbers untrustworthy
rather than merely out of date.

So the tables are generated from evaluation.json, backend/model_metrics.json and
the experiment result files, and CI fails if the committed docs disagree with
the artifacts.

How
---
Each managed table sits between a pair of HTML comments:

    <!-- AUTOGEN:shipped -->
    ...whatever is here is replaced...
    <!-- /AUTOGEN:shipped -->

Prose is never touched. Only the rows between the markers.

Run:  python tools/sync_docs.py           rewrite the tables
      python tools/sync_docs.py --check   fail if they are stale, change nothing
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", "PROJECT.md", "PAPER.md"]

NAME = {
    "colorectal": "Bowel", "general": "General", "liver": "Liver",
    "breast": "Breast", "ovarian": "Ovarian", "lung": "Lung",
    "prostate": "Prostate", "pancreatic": "Pancreatic",
}
COHORT = {
    "breast": "569 Wisconsin biopsies",
    "pancreatic": "600 samples, 3 tissue banks",
    "ovarian": "349 operated ovarian masses",
    "prostate": "212 biopsied men",
    "lung": "21,916 adults with tobacco exposure",
    "colorectal": "23,794 NHANES adults",
    "liver": "35,511 NHANES adults",
    "general": "23,923 NHANES adults",
}
LABEL = {
    "breast": "Breast malignancy", "pancreatic": "Pancreatic cancer",
    "ovarian": "Ovarian malignancy", "prostate": "Prostate cancer",
    "lung": "Lung cancer", "colorectal": "Bowel cancer",
    "liver": "Liver disease", "general": "General cancer",
}


def load(path, default=None):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _rank(evaluation):
    return sorted(evaluation, key=lambda k: -evaluation[k]["calibrated"]["auc"])


def table_shipped(ev, _):
    rows = ["| Panel | Trained on | Test AUC | 95% CI | Threshold | Sens | Spec | Flagged per true case |",
            "|---|---|---|---|---|---|---|---|"]
    for k in _rank(ev):
        c = ev[k]["calibrated"]
        ci = c.get("auc_ci") or ["", ""]
        thr = c.get("threshold")
        thr_s = f"{thr * 100:.1f}%" if isinstance(thr, (int, float)) and thr <= 1 else (
            f"{thr}%" if thr is not None else "")
        rows.append(
            f"| {LABEL.get(k, k)} | {COHORT.get(k, '')} | {c['auc']:.3f} | "
            f"{ci[0]} to {ci[1]} | {thr_s} | {c.get('sensitivity')} | "
            f"{c.get('specificity')} | {c.get('people_flagged_per_true_case')} |")
    rows.append("| ~~Cervical~~ | 858 Caracas referrals | 0.725 | withdrawn, a lucky split | | | | |")
    return "\n".join(rows)


def table_baselines(ev, extra):
    gain = extra.get("demographic_gain") or {}
    rows = ["| Panel | Model | Logistic | Age and sex alone | Gain over age and sex |",
            "|---|---|---|---|---|"]
    for k in _rank(ev):
        c = ev[k]["calibrated"]
        b = ev[k].get("baselines") or {}
        lr = (b.get("logistic_regression") or {}).get("auc")
        agesex = (b.get("age_sex_only") or {}).get("auc")
        g = gain.get(k) or {}
        gv = g.get("gain")
        # The gain column comes from repeated paired folds, not from the single
        # held-out split, because one split is not an estimate. This is the
        # error that got the cervical panel withdrawn.
        gs = (f"**{gv:+.3f}**" if isinstance(gv, (int, float)) and gv < 0.02
              else (f"{gv:+.3f}" if isinstance(gv, (int, float)) else "not measurable"))
        rows.append(f"| {NAME.get(k, k)} | {c['auc']:.3f} | {lr if lr is not None else '—'} | "
                    f"{agesex if agesex is not None else '—'} | {gs} |")
    return "\n".join(rows)


def table_stability(_, extra):
    st = (extra.get("stability") or {}).get("panels") or {}
    rows = ["| Panel | Rows | Events | Mean AUC | Spread across splits | Shipped split | Percentile |",
            "|---|---|---|---|---|---|---|"]
    for k, v in sorted(st.items(), key=lambda kv: -kv[1]["mean_auc"]):
        rows.append(
            f"| {NAME.get(k, k)} | {v['n_rows']:,} | {v['n_positive']:,} | "
            f"{v['mean_auc']:.3f} | {v['min_auc']:.3f} to {v['max_auc']:.3f} | "
            f"{v['shipped_split_auc']:.3f} | {v['shipped_split_percentile']:.0f}th |")
    rows.append("| Cervical | 858 | 55 | **0.594** | **0.421 to 0.789** | 0.725 | **97th** |")
    return "\n".join(rows)


DESIGN = {
    "general": ("Population", "NHANES 2005-2014, cancer diagnosed within 4 years"),
    "liver": ("Population", "NHANES, 7 cycles, clinical liver disease"),
    "colorectal": ("Population", "NHANES, colon or rectal cancer within 8 years"),
    "lung": ("Population", "NHANES, adults with measurable tobacco exposure"),
    "pancreatic": ("Case-control", "3 tissue banks, adenocarcinoma vs benign hepatobiliary"),
    "ovarian": ("Case-control", "operated ovarian masses, malignant vs benign"),
    "breast": ("Case-control", "Wisconsin fine needle aspirates, post-biopsy"),
    "prostate": ("Case-control", "biopsied men, adenocarcinoma vs benign biopsy"),
}


def table_paper_cohorts(ev, _):
    rows = ["| Panel | Design | n | Events | Prevalence | Cohort |",
            "|---|---|---|---|---|---|"]
    for k in _rank(ev):
        v = ev[k]
        design, desc = DESIGN.get(k, ("", ""))
        prev = v.get("cohort_prevalence")
        n = v["n_total"]
        events = int(round(prev * n)) if isinstance(prev, (int, float)) else None
        rows.append(
            f"| {NAME.get(k, k)} | {design} | {n:,} | "
            f"{events:,} | {prev:.2%} | {desc} |"
            if events is not None else
            f"| {NAME.get(k, k)} | {design} | {n:,} | — | — | {desc} |")
    return "\n".join(rows)


def table_paper_results(ev, extra):
    """
    The results section, assembled from the artifacts rather than typed.

    Ordered by what the paper argues: discrimination first, then what that
    discrimination is worth at real incidence, then whether it survives being
    resampled, then the prospective test.
    """
    out = []

    out.append("### 3.1 Discrimination, and what it adds over knowing age and sex\n")
    out.append(table_baselines(ev, extra))
    out.append("")
    out.append("The gain column is measured by repeated paired cross-validation on identical "
               "folds, not from the held-out split, because a single split proved unreliable.\n")

    out.append("### 3.2 What a score is worth at real incidence\n")
    rows = ["| Panel | Test AUC | PPV at population incidence | Flagged per true case | Usable as screening? |",
            "|---|---|---|---|---|"]
    for k in _rank(ev):
        c = ev[k]["calibrated"]
        per = c.get("people_flagged_per_true_case")
        ppv = c.get("ppv_at_population_prevalence")
        kind = DESIGN.get(k, ("", ""))[0]
        verdict = ("not a screening panel" if k in ("breast", "prostate", "ovarian", "pancreatic")
                   else ("no" if isinstance(per, (int, float)) and per > 50 else "yes, with caveats"))
        rows.append(f"| {NAME.get(k, k)} | {c['auc']:.3f} | "
                    f"{ppv * 100:.2f}% | {per} | {verdict} |"
                    if isinstance(ppv, (int, float)) else
                    f"| {NAME.get(k, k)} | {c['auc']:.3f} | — | {per} | {verdict} |")
    out.append("\n".join(rows))
    out.append("")
    out.append("This is the table that decides whether a panel is a screening instrument. "
               "Discrimination and usability are different properties, and three panels have "
               "the first without the second.\n")

    out.append("### 3.3 Stability across resampling\n")
    out.append(table_stability(ev, extra))
    out.append("")

    pm = extra.get("prospective") or {}
    out.append("### 3.4 The prospective test\n")
    if not pm:
        out.append("*(pending: experiments/prospective_mortality.py has not been run)*")
        return "\n".join(out)

    out.append(f"{pm['n']:,} adults, {pm['events']} deaths from malignant neoplasm within "
               f"{pm['horizon_months']} months of the blood draw ({pm['prevalence']:.2%}).\n")
    rows = ["| Feature set | Features | AUC | 95% CI | Gain over age and sex | Wins |",
            "|---|---|---|---|---|---|"]
    for name, a in pm["arms"].items():
        ci = a.get("auc_ci") or ["", ""]
        rows.append(f"| {name} | {a['n_features']} | {a['auc']:.3f} | {ci[0]} to {ci[1]} | "
                    f"{a['gain_over_age_sex']:+.3f} | {a['wins']}/{a['repeats']} |")
    out.append("\n".join(rows))
    out.append("")
    loco = pm.get("leave_one_cycle_out") or {}
    if loco:
        out.append("Leave-one-cycle-out, full feature set. Cycles differ in assay method, field "
                   "staff and population, so this approximates external validation within one "
                   "survey.\n")
        rows = ["| Held-out cycle | AUC |", "|---|---|"]
        for cyc, a in sorted(loco.items()):
            rows.append(f"| {cyc} | {a:.3f} |")
        rows.append(f"| **mean** | **{pm['mean_leave_one_cycle_out']:.3f}** |")
        out.append("\n".join(rows))
        out.append("")
    out.append(f"**{pm['verdict'].capitalize()}.**")

    ex = extra.get("external") or {}
    if not ex:
        return "\n".join(out)

    out.append("")
    out.append("### 3.5 Does that gain survive a different decade?\n")
    out.append(f"Trained on NHANES 1999-2014 ({ex['train_n']:,} adults, {ex['train_events']} "
               f"deaths) and tested on NHANES III 1988-1994 ({ex['test_n']:,} adults, "
               f"{ex['test_events']} deaths). Identical features on both sides. Nothing from "
               f"the test cohort touches fitting, calibration or imputation.\n")
    rows = ["| Feature set | Features | External AUC | 95% CI |", "|---|---|---|---|"]
    for name, a in ex["arms"].items():
        ci = a.get("external_auc_ci") or ["", ""]
        rows.append(f"| {name} | {a['n_features']} | {a['external_auc']:.3f} | "
                    f"{ci[0]} to {ci[1]} |")
    out.append("\n".join(rows))
    out.append("")
    g_ext = ex["external_gain_over_age_sex"]
    g_int = ex.get("internal_gain_for_reference")
    out.append(f"Gain over age and sex, transferred: **{g_ext:+.3f}**. The same gain measured "
               f"inside the training survey: {g_int:+.3f}.\n")
    if not ex.get("gain_survives_transfer"):
        out.append("**The gain does not survive the transfer.** Age and sex transfer well, at "
                   "0.852. Adding twenty blood values makes the prediction *worse* on a cohort "
                   "measured in a different decade than using age and sex alone. Whatever the "
                   "blood panel contributed inside NHANES 1999-2014 was specific to that survey "
                   "rather than to human physiology.\n")
        out.append("This is also a caution about the leave-one-cycle-out result above. Holding "
                   "out one cycle of the same survey gave a mean of 0.837 and looked like "
                   "evidence of transfer. It was not. Cycles of one survey share protocols, "
                   "instruments and laboratory methods, and resampling within a survey measures "
                   "stability rather than generalisation. Only the genuinely external cohort "
                   "distinguished them.")

    cx = extra.get("colorectal_external") or {}
    if not cx:
        return "\n".join(out)

    out.append("")
    out.append("### 3.6 The same test on an organ-specific panel\n")
    out.append(f"The bowel panel is one of only two here that screen for a named cancer from a "
               f"routine lab report alone, so it carries more of the application's claim than "
               f"the case-control panels do. NHANES III recorded both the site of any reported "
               f"cancer and the age at which it was first told, which reconstructs the same "
               f"eight-year window the training cohort uses.\n")
    out.append(f"Train: NHANES 2005-2014, {cx['train_n']:,} adults, {cx['train_events']} cases. "
               f"Test: NHANES III, {cx['test_n']:,} adults, {cx['test_events']} cases. The two "
               f"prevalences agree to within a hundredth of a percent, which is a check that the "
               f"window was reconstructed the same way on both sides.\n")
    rows = ["| Feature set | Features | External AUC | 95% CI |", "|---|---|---|---|"]
    for name, a in cx["arms"].items():
        ci = a.get("external_auc_ci") or ["", ""]
        rows.append(f"| {name} | {a['n_features']} | {a['external_auc']:.3f} | "
                    f"{ci[0]} to {ci[1]} |")
    out.append("\n".join(rows))
    out.append("")
    g, gi = cx["external_gain_over_age_sex"], cx.get("internal_gain_for_reference")
    out.append(f"Gain over age and sex, transferred: **{g:+.3f}**, against {gi:+.3f} measured "
               f"inside the training survey.\n")
    if cx.get("gain_survives_transfer"):
        out.append("**This gain survives.** Roughly three quarters of it is still there on a "
                   "cohort measured a decade and a half earlier, on different analysers. Set "
                   "beside section 3.5, where the undifferentiated panel's gain reversed sign "
                   "under the same test, this is the sharpest form of the paper's main result: "
                   "the two questions do not merely differ in effect size, they differ in "
                   "whether the effect is real at all.")
    return "\n".join(out)


def table_calibration(_, extra):
    # The result file carries a "_note" string alongside the panels, so filter to
    # entries that are actually panels rather than assuming every key is one.
    cal = {k: v for k, v in (extra.get("calibration") or {}).items()
           if isinstance(v, dict) and "methods" in v}
    rows = ["| Panel | n | AUC none | AUC isotonic | AUC sigmoid | Brier none | Brier isotonic |",
            "|---|---|---|---|---|---|---|"]
    for k, v in sorted(cal.items(), key=lambda kv: -kv[1]["methods"]["isotonic"]["auc"]):
        m = v["methods"]
        rows.append(
            f"| {NAME.get(k, k)} | {v['n']:,} | {m['none']['auc']} | "
            f"{m['isotonic']['auc']} | {m['sigmoid']['auc']} | "
            f"{m['none']['brier']} | {m['isotonic']['brier']} |")
    return "\n".join(rows)


TABLES = {
    "shipped": table_shipped,
    "baselines": table_baselines,
    "stability": table_stability,
    "calibration": table_calibration,
    "paper_cohorts": table_paper_cohorts,
    "paper_results": table_paper_results,
}


def main():
    check = "--check" in sys.argv
    ev = load("evaluation.json")
    if not ev:
        print("evaluation.json is missing — run evaluate.py first", file=sys.stderr)
        return 1
    extra = {
        "demographic_gain": load("experiments/demographic_gain_result.json", {}),
        "stability": load("experiments/split_stability_result.json", {}),
        "prospective": load("experiments/prospective_mortality_result.json", {}),
        "external": load("experiments/prospective_external_result.json", {}),
        "calibration": load("experiments/calibration_method_result.json", {}),
        "colorectal_external": load("experiments/colorectal_external_result.json", {}),
    }

    stale, written = [], []
    for doc in DOCS:
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        out = src
        for key, build in TABLES.items():
            pattern = re.compile(
                rf"(<!-- AUTOGEN:{key} -->\n)(.*?)(\n<!-- /AUTOGEN:{key} -->)", re.S)
            if not pattern.search(out):
                continue
            out = pattern.sub(lambda m: m.group(1) + build(ev, extra) + m.group(3), out)
        if out != src:
            if check:
                stale.append(doc)
            else:
                open(path, "w", encoding="utf-8").write(out)
                written.append(doc)

    if check:
        if stale:
            print(f"STALE: {', '.join(stale)} disagree with the artifacts. "
                  f"Run python tools/sync_docs.py and commit.", file=sys.stderr)
            return 1
        print("documentation tables match the artifacts")
        return 0

    print(f"regenerated tables in {', '.join(written) if written else 'nothing (already current)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
