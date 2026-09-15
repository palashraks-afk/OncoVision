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

Only what sits between the markers is touched. Most markers hold tables; a few
hold sentences, where a sentence states a figure the pipeline produces -- the
paper's abstract quoted a cost result for a full retrain cycle after it had
stopped being true, and a claim that a gain "survives transfer" outlived the
baseline it was measured against.

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
    "breast": "Breast (biopsy)", "breast_screening": "Breast (mammogram)",
    "ovarian": "Ovarian", "lung": "Lung",
    "prostate": "Prostate", "pancreatic": "Pancreatic",
}
COHORT = {
    "breast": "569 Wisconsin biopsies",
    "breast_screening": "400,000 BCSC mammograms",
    "pancreatic": "600 samples, 3 tissue banks",
    "ovarian": "349 operated ovarian masses",
    "prostate": "212 biopsied men",
    "lung": "19,866 adults with tobacco exposure",
    "colorectal": "28,527 NHANES adults",
    "liver": "30,624 NHANES adults",
    "general": "28,711 NHANES adults",
}
LABEL = {
    "breast": "Breast malignancy", "breast_screening": "Breast cancer within a year",
    "pancreatic": "Pancreatic cancer",
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


def _withdrawn(extra, k):
    return ((extra or {}).get("metrics") or {}).get(k, {}).get("shipped") is False


def _name(extra, k):
    n = NAME.get(k, k)
    return f"~~{n}~~ withdrawn" if _withdrawn(extra, k) else n


def _rank(evaluation):
    return sorted(evaluation, key=lambda k: -evaluation[k]["calibrated"]["auc"])


def table_shipped(ev, extra):
    """
    What ships, with the held-out split marked where it is not representative.

    A held-out AUC is one draw. Where split_stability.py puts that draw at an
    extreme percentile of its own distribution, quoting it unqualified is the
    error that got the cervical panel withdrawn, and the general panel's split
    landed at the 100th percentile the moment its cohort grew. The stable mean
    is shown beside it so the flattering number never appears alone.
    """
    st = (extra.get("stability") or {}).get("panels") or {}
    rows = ["| Panel | Trained on | Test AUC | 95% CI | Threshold | Sens | Spec | Flagged per true case |",
            "|---|---|---|---|---|---|---|---|"]
    for k in _rank(ev):
        c = ev[k]["calibrated"]
        ci = c.get("auc_ci") or ["", ""]
        thr = c.get("threshold")
        thr_s = f"{thr * 100:.1f}%" if isinstance(thr, (int, float)) and thr <= 1 else (
            f"{thr}%" if thr is not None else "")
        # Mark a split that sits at an extreme of its own distribution, and
        # give the stable mean next to it.
        sv = st.get(k) or {}
        pct = sv.get("shipped_split_percentile")
        auc_cell = f"{c['auc']:.3f}"
        if isinstance(pct, (int, float)) and (pct >= 90 or pct <= 10):
            auc_cell = (f"{c['auc']:.3f} ⚠️<br>_a lucky draw, {pct:.0f}th pct;_<br>"
                        f"_stable mean {sv['mean_auc']:.3f}_")
        rows.append(
            f"| {('~~' + LABEL.get(k, k) + '~~ withdrawn') if _withdrawn(extra, k) else LABEL.get(k, k)} | {COHORT.get(k, '')} | {auc_cell} | "
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
        rows.append(f"| {_name(extra, k)} | {c['auc']:.3f} | {lr if lr is not None else '—'} | "
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
    "general": ("Population", "NHANES 2005-2016, cancer diagnosed within 4 years"),
    "liver": ("Population", "NHANES, 7 cycles, clinical liver disease"),
    "colorectal": ("Population", "NHANES 2005-2016, colon or rectal cancer within 8 years"),
    "lung": ("Population", "NHANES, adults with measurable tobacco exposure"),
    "pancreatic": ("Case-control", "3 tissue banks, adenocarcinoma vs benign hepatobiliary"),
    "ovarian": ("Case-control", "operated ovarian masses, malignant vs benign"),
    "breast": ("Case-control", "Wisconsin fine needle aspirates, post-biopsy"),
    "breast_screening": ("Population", "BCSC screening mammograms, cancer within 1 year"),
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
    out.append(f"Gain over age and sex, transferred, blood work alone: **{g_ext:+.3f}**. For "
               f"reference, the full panel inside the training survey, which also includes "
               f"BMI, smoking and alcohol: {g_int:+.3f}.\n")
    if not ex.get("gain_survives_transfer"):
        # Both figures used to be typed into these sentences. They are read from
        # the results now, because the age-and-sex baseline was refitted with
        # both model kinds and the numbers moved.
        base_ext = max((a["external_auc"] for k, a in ex["arms"].items()
                        if k.startswith("age and sex only")), default=float("nan"))
        loco_mean = pm.get("mean_leave_one_cycle_out", float("nan"))
        out.append(f"**The gain does not survive the transfer.** Age and sex transfer well, at "
                   f"{base_ext:.3f}. Adding twenty blood values does not improve on that on a "
                   f"cohort measured in a different decade. Whatever the blood panel contributed "
                   f"inside NHANES 1999-2014 was specific to that survey rather than to human "
                   f"physiology.\n")
        out.append(f"This is also a caution about the leave-one-cycle-out result above. Holding "
                   f"out one cycle of the same survey gave a mean of {loco_mean:.3f} and looked like "
                   "evidence of transfer. It was not. Cycles of one survey share protocols, "
                   "instruments and laboratory methods, and resampling within a survey measures "
                   "stability rather than generalisation. Only the genuinely external cohort "
                   "distinguished them.")
    else:
        arms_ = ex["arms"]

        def best_ext(prefix):
            return max((a["external_auc"] for k, a in arms_.items() if k.startswith(prefix)),
                       default=float("nan"))

        gci = ex.get("external_gain_ci") or [float("nan"), float("nan")]
        old = ex.get("old_gain_ensemble_vs_ensemble")
        ens_full = arms_.get("full blood work, ensemble", {}).get("external_auc", float("nan"))
        out.append(f"**The gain survives the transfer.** The better blood-work model scores "
                   f"{best_ext('full blood work'):.3f} on NHANES III against "
                   f"{best_ext('age and sex only'):.3f} for the better age-and-sex model, a gain "
                   f"of {g_ext:+.3f} (95% CI {gci[0]:+.3f} to {gci[1]:+.3f}).")
        out.append("")
        if isinstance(old, (int, float)) and old < 0:
            out.append(f"An earlier version of this section reported the opposite. It fitted a "
                       f"tree ensemble on both arms, found the blood-work arm losing to age and sex "
                       f"by {old:+.3f}, and read that as the signal belonging to one survey. The "
                       f"signal did not belong to one survey; the ensemble did. On the same "
                       f"twenty-two features it scores {ens_full:.3f} externally, where logistic "
                       f"regression transfers. **An external test validates a model, not a "
                       f"hypothesis**, and a failed transfer can belong to the model.")

    cx = extra.get("colorectal_external") or {}
    if cx:

        out.append("")
        out.append("### 3.6 The same test on an organ-specific panel\n")
        out.append(f"The bowel panel was one of two here that claimed to screen for a named cancer "
                   f"from a routine lab report alone, so it carried more of the application's claim "
                   f"than the case-control panels did, and it has since been withdrawn for the "
                   f"reason this section records. NHANES III recorded both the site of any reported "
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
        # Both arms of that table were fitted with a calibrated tree ensemble, and
        # a tree ensemble given only age and a binary sex flag ranks people in
        # coarse steps. The paper used to call the resulting gain a transfer that
        # "survives"; refitted against a logistic age-and-sex model it is zero.
        # The claim now follows the stronger baseline, never the weaker one.
        xb = (extra.get("external_baseline") or {}).get("colorectal")
        if xb:
            a = xb["auc"]
            lo, hi = xb["honest_gain_ci"]
            olo, ohi = xb["old_gain_ci"]
            if xb["gain_survives_honest_baseline"]:
                out.append(f"**The gain survives a baseline that can use age:** "
                           f"{xb['honest_gain_best_vs_best']:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}.")
            else:
                out.append(
                    f"**That gain belongs to the baseline, not the bloodwork.** Both arms above "
                    f"were fitted with a calibrated tree ensemble, which given only age and a "
                    f"binary sex flag ranks people in coarse steps. Refitted with logistic "
                    f"regression, age and sex alone score {a['base_logistic']:.3f} on NHANES III "
                    f"and the full sixteen-feature panel scores {a['full_logistic']:.3f}. Against "
                    f"the stronger baseline the transferred gain is "
                    f"{xb['honest_gain_best_vs_best']:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}, and "
                    f"the original interval, {olo:+.3f} to {ohi:+.3f}, never excluded zero "
                    f"either. **On this evidence routine bloodwork adds nothing to age and sex "
                    f"for bowel cancer.** An earlier draft of this paper read the ensemble "
                    f"comparison as a gain that survived transfer; it was a weak baseline.")
    cost = extra.get("cost") or {}
    if cost:
        out.append("")
        out.append("### 3.7 Does triage on free bloodwork save money?\n")
        out.append("Per 100,000 people at real incidence, sending everyone for the "
                   "confirmatory procedure against sending only those the panel flags.\n")
        rows = ["| Panel | Procedure | Sent everyone | Sent if flagged | Cases missed | Apparent saving |",
                "|---|---|---|---|---|---|"]
        for k, v in cost.get("panels", {}).items():
            b, cfg = v["base_case"], v["settings"]
            total = b["cancers_found"] + b["cancers_missed"]
            rows.append(
                f"| {NAME.get(k, k)} | {cfg['procedure']} | {b['procedures_universal']:,} | "
                f"{b['procedures_triaged']:,} | {b['cancers_missed']} of {total:.0f} | "
                f"${b['saving']:,} |")
        out.append("\n".join(rows))
        out.append("")
        out.append("That apparent saving counts only treatment dollars. Charging a missed "
                   "case what a life is conventionally worth changes the answer. Each panel "
                   "is valued on its own endpoint: fifteen life-years for a cancer, five for "
                   "liver disease, at $150,000 per QALY.\n")
        out.append(cost_breakeven_table(cost))
        out.append("")
        out.append("**The operating point, not the model, decides this.** Choosing the point on "
                   "each panel's real ROC curve that maximises net benefit once a missed case "
                   "is priced at a life:\n")
        out.append(cost_best_table(cost))
        out.append("")
        liver = (cost.get("panels", {}).get("liver") or {}).get("best_operating_point")
        if liver and liver["procedures_per_100k"] >= 99_000:
            out.append("**The liver row is the interesting one.** That panel has the largest "
                       f"gain over age and sex of anything in this project, {_gain(extra, 'liver')}, and its "
                       "best operating point is to send everyone: no triage threshold beats "
                       "universal testing once a missed case is priced. Liver disease is "
                       "common at 4% and a FibroScan is cheap at $500, so the scans a "
                       "threshold saves are worth less than the cases it misses. "
                       "**Discrimination did not decide this; prevalence and procedure cost "
                       "did.** The panel that separates best is the one where triage helps "
                       "least, which is the clearest available demonstration that AUC and "
                       "decision value are different quantities.\n")
        out.append("An illustrative model, not a cost-effectiveness analysis: no discounting, no "
                   "quality-adjusted life years beyond the per-panel figure above, and no price "
                   "on the harm of an unnecessary procedure. The treatment costs are first-year "
                   "figures and understate the late-stage penalty, which biases the model "
                   "*towards* triage.")



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


def table_cv_vs_heldout(ev, extra):
    """
    Cross-validated against held-out, with the gap between them.

    The gap is the point of the table. Cervical's was 0.138 and that was the
    evidence it was a lucky split, read at the time as ordinary optimism. The
    cervical row is kept from the run that withdrew it, because the panel no
    longer trains and there is nothing current to regenerate it from.
    """
    rows = ["| Panel | CV AUC | Held-out AUC | Gap |", "|---|---|---|---|"]
    for k in _rank(ev):
        cv = ev[k].get("cv_auc_train_only")
        ho = ev[k]["calibrated"]["auc"]
        if not isinstance(cv, (int, float)):
            continue
        rows.append(f"| {NAME.get(k, k)} | {cv:.3f} | {ho:.3f} | {ho - cv:+.3f} |")
    rows.append("| ~~Cervical~~ | 0.587 | 0.725 | **+0.138** |")
    return "\n".join(rows)


# ------------------------------------------------------------- cost prose
#
# Every sentence below used to be typed by hand, and they went stale in the
# worst possible place: the paper's abstract kept saying the lung panel avoids
# 21,561 CT scans and nets $6.5M per 100,000 for a full retrain cycle after the
# cost model had concluded that lung's best operating point is to send
# everyone. A figure a pipeline produces belongs to the pipeline.

def _m(x):
    return "$0" if abs(x) < 50_000 else f"${x / 1e6:,.1f}M"


def _gain(extra, panel):
    g = ((extra.get("demographic_gain") or {}).get(panel) or {}).get("gain")
    return f"{g:+.3f}" if isinstance(g, (int, float)) else "n/a"


def _cases(v):
    return round(100_000 * v["settings"]["incidence"])


def _cost_panels(extra):
    return (extra.get("cost") or {}).get("panels", {}) or {}


def cost_breakeven_table(cost):
    rows = ["| Panel | Break-even per missed case | A case, valued | Verdict |",
            "|---|---|---|---|"]
    for k, v in cost.get("panels", {}).items():
        be = v.get("break_even_per_missed_cancer")
        soc = v.get("societal_cost_of_a_missed_cancer")
        if be is None:
            continue
        rows.append(f"| {NAME.get(k, k)} | ${be:,} | ${soc:,} | "
                    f"{'still saves' if be > soc else '**stops saving**'} |")
    return "\n".join(rows)


def cost_best_table(cost):
    rows = ["| Panel | Sensitivity | Specificity | Procedures avoided per 100,000 | Cases missed | Net benefit |",
            "|---|---|---|---|---|---|"]
    for k, v in cost.get("panels", {}).items():
        bo = v.get("best_operating_point")
        if not bo:
            continue
        avoided = 100_000 - bo["procedures_per_100k"]
        rows.append(f"| {NAME.get(k, k)} | {bo['sensitivity']} | {bo['specificity']} | "
                    f"**{avoided:,}** | {bo['cancers_missed']} | "
                    f"${bo['net_benefit']:,} |")
    return "\n".join(rows)


def _age_alone_sentence(extra, k):
    """Whether the lab values, rather than age, are what makes triage pay.

    The cost model only ever priced the panel. triage_on_age_alone.py runs the
    same operating-point search on an age-and-sex model with no lab values, on
    the same folds. For bowel, age and sex alone netted more than the panel and
    missed half as many cancers, so a sentence crediting the bloodwork with the
    saving has to be followed by that one.
    """
    t = (extra.get("triage_age") or {}).get(k)
    if not t:
        return ""
    base = t["arms"][t["best_baseline_arm"]]
    panel = t["arms"][t["best_panel_arm"]]
    if t["panel_minus_age_and_sex"] > 0:
        return (f"Triage on age and sex alone, with no lab values, nets "
                f"{_m(base['net_benefit'])}, so the lab values add "
                f"{_m(t['panel_minus_age_and_sex'])} on top.")
    return (f"**But the lab values are not what pays.** Triage on age and sex alone, with no lab "
            f"values at all, avoids {base['procedures_avoided']:,} procedures, misses "
            f"{base['cancers_missed']} cancers and nets {_m(base['net_benefit'])}, against "
            f"{_m(panel['net_benefit'])} for the best version of the panel on the same folds.")


def table_hero_cost(_, extra):
    panels = _cost_panels(extra)
    if not panels:
        return "_Run experiments/cost_model.py._"
    labels = {"colorectal": ("Colonoscopies", "Bowel cancers"),
              "lung": ("Lung CT scans", "Lung cancers"),
              "liver": ("FibroScans", "Liver disease cases")}
    rows = ["| | Send everyone | Triage on this panel | Difference |", "|---|---|---|---|"]
    paying, not_paying, withdrawn_paying = [], [], []
    for k, v in panels.items():
        bo = v.get("best_operating_point")
        if not bo:
            continue
        if not bo["pays_once_a_life_is_priced"]:
            not_paying.append(NAME.get(k, k).lower())
            continue
        if _withdrawn(extra, k):
            withdrawn_paying.append((k, v, bo))
            continue
        proc, case = labels.get(k, (v["settings"]["procedure"], "Cases"))
        rows.append(f"| **{proc}** | 100,000 | {bo['procedures_per_100k']:,} | "
                    f"**{100_000 - bo['procedures_per_100k']:,} avoided** |")
        rows.append(f"| {case} missed | 0 | {bo['cancers_missed']} of {_cases(v):,} | |")
        paying.append((k, bo))
    out = (["\n".join(rows), ""] if paying else
           [("**No lab-report panel's triage pays once a missed cancer is priced at a life.** "
             "The one decision that does is which women with dense breasts get a supplemental "
             "MRI, where the mammogram breast panel's triage beats both MRI for all and triage on "
             "age -- by a small amount at Medicare prices (see the paper, section 4.6)."
             if (extra.get("breast_mri") or {}).get("beats_age_in_every_sweep") else
             "**No shipped panel's triage pays once a missed cancer is priced at a life.**"), ""])
    for k, v, bo in withdrawn_paying:
        t = (extra.get("triage_age") or {}).get(k) or {}
        base = (t.get("arms") or {}).get(t.get("best_baseline_arm"), {})
        out.append(
            f"The lab-report panel that appeared to pay was {NAME.get(k, k).lower()}: at its best "
            f"operating point it "
            f"avoided {100_000 - bo['procedures_per_100k']:,} procedures per 100,000 and netted "
            f"{_m(bo['net_benefit'])}."
            + (f" Triage on age and sex alone, with no lab values, avoided "
               f"{base['procedures_avoided']:,}, missed {base['cancers_missed']} cancers instead of "
               f"{bo['cancers_missed']}, and netted {_m(base['net_benefit'])}, so the panel was "
               f"withdrawn: the saving was the patient's age, not their lab report."
               if base and t.get("panel_minus_age_and_sex", 1) <= 0 else ""))
    for k, bo in paying:
        out.append(f"That is a net benefit of **{_m(bo['net_benefit'])} per 100,000 people** on "
                   f"{NAME.get(k, k).lower()}, counted *after* pricing every missed cancer at "
                   f"fifteen life-years.")
        if _age_alone_sentence(extra, k):
            out.append(_age_alone_sentence(extra, k))
    if not_paying:
        out.append(f"For {' and '.join(not_paying)}, no threshold beats sending everyone once a "
                   f"missed case is priced, so those panels offer no rule-out call.")
    return "\n".join(out)


def table_cost_ranking(_, extra):
    panels = _cost_panels(extra)
    gains = extra.get("demographic_gain") or {}

    def order(k):
        g = (gains.get(k) or {}).get("gain")
        return g if isinstance(g, (int, float)) else float("-inf")

    rows = ["| Panel | Gain over age and sex | Best operating point | Procedures avoided per 100,000 | Net benefit |",
            "|---|---|---|---|---|"]
    for k in sorted(panels, key=order, reverse=True):
        bo = panels[k].get("best_operating_point")
        if not bo:
            continue
        point = (f"sens {bo['sensitivity']:.3f}" if bo["pays_once_a_life_is_priced"]
                 else "send everyone")
        rows.append(f"| {NAME.get(k, k)} | {_gain(extra, k)} | {point} | "
                    f"**{100_000 - bo['procedures_per_100k']:,}** | {_m(bo['net_benefit'])} |")
    return "\n".join(rows)


def _bowel_base(extra):
    c = _cost_panels(extra).get("colorectal")
    if not c or "break_even_per_missed_cancer" not in c:
        return None
    b = c["base_case"]
    return c, b, round(b["cancers_found"] + b["cancers_missed"])


def text_abstract_cost(_, extra):
    got = _bowel_base(extra)
    if not got:
        return "_Run experiments/cost_model.py._"
    c, b, total = got
    be, soc = c["break_even_per_missed_cancer"], c["societal_cost_of_a_missed_cancer"]
    s = ("**Cost.** Discrimination is not the outcome that matters for a tool whose purpose is to "
         "reduce spending on diagnostics, so we modelled it directly: per 100,000 people, sending "
         "everyone for the confirmatory procedure against sending only those a panel flags, "
         "charging missed cancers the difference between early and late-stage treatment. "
         f"At the balanced operating point the bowel panel ships, triage appears to save "
         f"${b['saving'] / 1e6:,.0f}M per 100,000 — by missing {b['cancers_missed']:.0f} of "
         f"{total:,} cancers. Break-even is ${be / 1e6:.2f}M per missed cancer against "
         f"${soc / 1e6:.2f}M for fifteen life-years at conventional willingness-to-pay, so "
         + ("the saving survives even once a life is priced. " if be > soc
            else "the saving disappears once a life is priced. ")
         + "Choosing instead the point on each panel's ROC curve that maximises net benefit "
           "**after** charging a missed cancer at a life: ")
    parts, none = [], []
    for k, v in _cost_panels(extra).items():
        bo = v.get("best_operating_point")
        if not bo:
            continue
        if bo["pays_once_a_life_is_priced"]:
            parts.append(f"{NAME.get(k, k).lower()} avoids "
                         f"{100_000 - bo['procedures_per_100k']:,} procedures per 100,000 while "
                         f"missing {bo['cancers_missed']} of {_cases(v):,} cases "
                         f"(+{_m(bo['net_benefit'])})")
        else:
            none.append(NAME.get(k, k).lower())
    s += ("; ".join(parts) + "." if parts else "no panel's triage pays.")
    for k, v in _cost_panels(extra).items():
        if ((v.get("best_operating_point") or {}).get("pays_once_a_life_is_priced")
                and _age_alone_sentence(extra, k)):
            s += " " + _age_alone_sentence(extra, k)
    if none:
        s += f" For {' and '.join(none)}, no threshold beats sending everyone."
    return s


def text_operating_point(_, extra):
    got = _bowel_base(extra)
    if not got:
        return "_Run experiments/cost_model.py._"
    c, b, total = got
    cfg = c["settings"]
    be, soc = c["break_even_per_missed_cancer"], c["societal_cost_of_a_missed_cancer"]
    out = [f"The consequence is measurable. At Youden, the colorectal panel flags "
           f"{b['procedures_triaged']:,} people per 100,000 and misses "
           f"{b['cancers_missed']:.0f} of {total:,} cancers. It appears to save "
           f"${b['saving'] / 1e6:,.0f}M, and the appearance survives only while a missed "
           f"cancer is priced at the ${cfg['late_cost'] - cfg['early_cost']:,.0f} difference "
           f"between early and late-stage treatment. Priced at fifteen life-years, the "
           f"break-even is ${be / 1e6:.2f}M against ${soc / 1e6:.2f}M and the saving "
           + ("survives." if be > soc else "evaporates."), ""]
    bo = c.get("best_operating_point")
    if bo and bo["pays_once_a_life_is_priced"]:
        out.append(f"Move along the same ROC curve — the same model, the same features, the "
                   f"same data — to the point that maximises net benefit once a missed cancer "
                   f"costs ${soc / 1e6:.2f}M, and the panel avoids "
                   f"{100_000 - bo['procedures_per_100k']:,} colonoscopies per 100,000 people "
                   f"while missing {bo['cancers_missed']} of {_cases(c):,} cancers.")
        if _age_alone_sentence(extra, "colorectal"):
            out.append(_age_alone_sentence(extra, "colorectal"))
    none = [NAME.get(k, k).lower() for k, v in _cost_panels(extra).items()
            if k != "colorectal"
            and not (v.get("best_operating_point") or {}).get("pays_once_a_life_is_priced", True)]
    if none:
        out.append(f"For {' and '.join(none)}, no point on the curve beats sending everyone, "
                   f"which is why those panels offer no rule-out call.")
    return "\n".join(out)


def table_bowel_external(_, extra):
    """Bowel on NHANES III, with both model kinds on both arms.

    The original table showed only the ensemble column and called the gap a
    gain that survived transfer. Showing the logistic column beside it is the
    whole correction, so the table carries both and the verdict follows the
    stronger baseline.
    """
    xb = (extra.get("external_baseline") or {}).get("colorectal")
    if not xb:
        return "_Run experiments/external_baseline_strength.py._"
    a = xb["auc"]
    lo, hi = xb["honest_gain_ci"]
    olo, ohi = xb["old_gain_ci"]
    rows = ["| Feature set | Tree ensemble | Logistic regression |",
            "|---|---|---|",
            f"| Age and sex only | {a['base_ensemble']:.3f} | **{a['base_logistic']:.3f}** |",
            f"| Full panel, 16 features | {a['full_ensemble']:.3f} | {a['full_logistic']:.3f} |"]
    verdict = ("**The gain survives a baseline that can use age.**"
               if xb["gain_survives_honest_baseline"] else
               "**On this evidence routine bloodwork adds nothing to age and sex for bowel "
               "cancer.** The contrast this section used to draw, an organ-specific gain "
               "that survived beside an undifferentiated one that reversed, does "
               "not hold in either direction: the bowel gain was a weak baseline, and "
               "the prospective reversal was an overfitted model.")
    return ("\n".join(rows) + "\n\n"
            f"Comparing the two ensemble cells gives {xb['old_gain_ensemble_vs_ensemble']:+.3f}, "
            f"with an interval of {olo:+.3f} to {ohi:+.3f} that never excluded zero. The best "
            f"panel against the best age-and-sex model gains "
            f"**{xb['honest_gain_best_vs_best']:+.3f}**, 95% CI {lo:+.3f} to {hi:+.3f}. " + verdict)


def text_breast_vs_age(_, extra):
    """Whether the breast panel's rule-out cut beats a cut on age alone.

    The question that withdrew the bowel panel, asked of the one population
    panel that still ships a cut, at matched sensitivity so neither cut can look
    better by sitting at a different point on its curve.
    """
    r = extra.get("breast_vs_age")
    if not r:
        return "_Run experiments/bcsc_rule_out_vs_age.py._"
    lo, hi = r["difference_ci"]
    head = (f"At the same share of cancers caught, {r['matched_sensitivity']:.1%}, on the "
            f"{r['n_mammograms']:,}-mammogram validation split, the panel's cut excluded "
            f"{r['panel_excluded_at_matched']:.1%} of women and a cut on age alone "
            f"{r['age_excluded_at_matched']:.1%}: a difference of {r['difference']:+.1%}, 95% CI "
            f"{lo:+.1%} to {hi:+.1%}.")
    if r["panel_beats_age"]:
        return head + (" **Unlike the bowel panel, this one earns its extra questions**: the "
                       "density grading and history exclude materially more women than their "
                       "age does, without catching fewer cancers.")
    return head + " That does not show the panel beating age alone."


def text_lung_loco(_, extra):
    """Lung's gain over age and sex with every survey cycle held out in turn.

    The withheld 2017-2018 cycle held thirteen lung cancers, too few to confirm
    or refute the panel's in-survey gain. Pooling every cycle, each scored by a
    model that never saw it, uses all of them. It is still one survey, and the
    wording says so whichever way the result falls.
    """
    r = extra.get("lung_loco")
    if not r:
        return "_Run experiments/lung_loco_gain.py._"
    lo, hi = r["gain_ci"]
    head = (f"Holding out every survey cycle in turn -- {r['cycles']} cycles, "
            f"{r['events']} lung cancers, each scored by a model that never saw its cycle -- "
            f"the lung panel scores {r['panel_auc']:.3f} against {r['age_sex_auc']:.3f} for the "
            f"stronger age-and-sex model, a gain of {r['gain']:+.3f} (95% CI {lo:+.3f} to "
            f"{hi:+.3f}).")
    if r["gain_confirmed_within_survey"]:
        return head + (" That confirms the in-survey gain across cycles the model never saw. "
                       "It remains one survey, with one protocol and one laboratory contract, "
                       "and section 4.2 records why that is not the same as an external cohort.")
    # Worded to match the lung card, which reads the same result. A point estimate
    # that agrees with the in-survey gain, with an interval whose lower end sits
    # at zero, is "not yet shown", not "absent".
    return head + (" The estimate agrees with the in-survey gain, but its range still reaches "
                   "zero, so the lung panel's advantage over age and sex is probably real and "
                   "not yet shown.")


def text_general_vs_age(_, extra):
    """Whether the general panel's rule-out cut beats a cut on age and sex.

    The test that withdrew the bowel panel and that the mammogram breast panel
    passed, applied to the only other panel that ships a rule-out call.
    """
    r = extra.get("general_vs_age")
    if not r:
        return "_Run experiments/general_rule_out_vs_age.py._"
    i, e = r["in_survey"], r["external"]
    def part(x, where):
        lo, hi = x["difference_ci"]
        return (f"{where}, at {x['matched_sensitivity']:.1%} of cancers caught, the panel's cut "
                f"excluded {x['panel_excluded']:.1%} of adults and a cut on age and sex alone "
                f"{x['age_sex_excluded']:.1%} ({x['difference']:+.1%}, 95% CI {lo:+.1%} to "
                f"{hi:+.1%})")
    head = part(i, "Inside its own survey") + "; " + part(e, "on NHANES III") + "."
    if r["panel_beats_age_sex_in_both"]:
        return head + " The general panel's rule-out call beats age and sex in both settings."
    return head + (" **The general panel's rule-out call does not beat age and sex**, which is "
                   "the rule that withdrew the bowel panel.")


def table_prospective_arms(_, extra):
    """The prospective mortality arms, and leave-one-cycle-out, from the result.

    This table was typed by hand, and every gain in it was measured against a
    tree ensemble fitted on age and sex -- the weak baseline that manufactured
    the bowel panel's result. The experiment now fits both model kinds on every
    arm, baseline included, and the table follows it.
    """
    pm = extra.get("prospective") or {}
    arms = pm.get("arms") or {}
    if not arms:
        return "_Run experiments/prospective_mortality.py._"
    labels = {"A age and sex": "Age and sex", "B + lifestyle": "+ BMI, smoking, alcohol",
              "C + blood count": "+ complete blood count",
              "D + chemistry": "+ metabolic and liver panel", "E everything": "Everything"}
    rows = ["| Feature set | Features | AUC | Gain over age and sex | Wins |",
            "|---|---|---|---|---|"]
    for k, a in arms.items():
        if k.startswith("A"):
            rows.append(f"| {labels.get(k, k)} | {a['n_features']} | {a['auc']:.3f} | — | |")
        else:
            rows.append(f"| {labels.get(k, k)} | {a['n_features']} | {a['auc']:.3f} | "
                        f"{a['gain_over_age_sex']:+.3f} | {a['wins']}/{a['repeats']} |")
    out = "\n".join(rows)
    if any("auc_by_model" in a for a in arms.values()):
        out += ("\n\nEvery row is the better of logistic regression and the ensemble on each "
                "repeat, the age-and-sex row included.")
    loco = pm.get("mean_leave_one_cycle_out")
    if isinstance(loco, (int, float)):
        out += (f"\n\nLeave-one-cycle-out, training on seven NHANES cycles and testing on the "
                f"eighth, gives a mean of {loco:.3f}.")
    return out


def text_prospective_verdict(_, extra):
    pm = extra.get("prospective") or {}
    arms = pm.get("arms") or {}
    if not arms:
        return "_Run experiments/prospective_mortality.py._"
    rest = {k: a for k, a in arms.items() if not k.startswith("A")}
    best = max(a["gain_over_age_sex"] for a in rest.values())
    every = all(a.get("wins") == a.get("repeats") for a in rest.values())
    if best >= 0.02:
        return (f"**Routine bloodwork carries prospective signal inside the survey.** The largest "
                f"gain over age and sex is {best:+.3f}.")
    if best > 0 and every:
        return (f"**The gain is consistent inside the survey and too small to act on.** Every arm "
                f"beats age and sex on every paired repeat, and the largest gain is {best:+.3f}.")
    return (f"**Against the stronger age-and-sex model, bloodwork adds little or nothing.** The "
            f"largest gain is {best:+.3f}.")


def table_prospective_external(_, extra):
    ex = extra.get("external") or {}
    arms = ex.get("arms") or {}
    if not arms:
        return "_Run experiments/prospective_external.py._"
    rows = ["| Feature set | Features | External AUC | 95% CI |", "|---|---|---|---|"]
    for k, a in arms.items():
        ci = a.get("external_auc_ci") or ["", ""]
        rows.append(f"| {k} | {a['n_features']} | {a['external_auc']:.3f} | {ci[0]} to {ci[1]} |")
    g = ex["external_gain_over_age_sex"]
    gci = ex.get("external_gain_ci")
    gi = ex.get("internal_gain_for_reference")
    text = f"**Transferred gain, best panel against best age-and-sex model: {g:+.3f}**"
    if gci:
        text += f" (95% CI {gci[0]:+.3f} to {gci[1]:+.3f})"
    if isinstance(gi, (int, float)):
        text += (f", against {gi:+.3f} inside the training survey for the full panel, which "
                 f"also includes BMI, smoking and alcohol")
    text += ". "
    text += ("Whatever the blood values contributed inside NHANES 1999-2014 belonged to that survey "
             "rather than to human physiology." if not ex.get("gain_survives_transfer") else
             "The gain survives a cohort measured in a different decade.")
    return "\n".join(rows) + "\n\n" + text


def text_prospective_short(_, extra):
    pm = extra.get("prospective") or {}
    ex = extra.get("external") or {}
    e = (pm.get("arms") or {}).get("E everything") or {}
    if not e or not ex:
        return "_Run experiments/prospective_mortality.py and prospective_external.py._"
    gci = ex.get("external_gain_ci")
    ci = f" (95% CI {gci[0]:+.3f} to {gci[1]:+.3f})" if gci else ""
    return (f"On the prospective cohort of {pm['n']:,} adults with NDI-confirmed outcomes, the full "
            f"panel, which adds BMI, smoking and alcohol to routine blood work, gains "
            f"{e['gain_over_age_sex']:+.3f} over the stronger age-and-sex model for cancer death "
            f"within five years inside its survey; blood work alone gains "
            f"{ex['external_gain_over_age_sex']:+.3f}{ci} on NHANES III."
            + (" That gain does not survive the transfer." if not ex.get("gain_survives_transfer")
               else " That gain survives the transfer."))


def text_breast_mri_cost(_, extra):
    """Supplemental MRI triage among women with dense breasts, priced and swept.

    The first decision in this project where a panel's triage pays and beats age.
    It is also a small effect at Medicare prices, and the wording says both.
    """
    r = extra.get("breast_mri")
    if not r:
        return "_Run experiments/breast_mri_triage_cost.py._"
    b = r["base"]
    bp = b["panel"]
    mri = r["base_inputs"]["mri_cost"]
    thousand = r["sweeps"].get("mri_cost", {}).get("1000.0", {})
    parts = [
        f"Among {r['dense_mammograms']:,} mammograms in women with dense breasts "
        f"({r['dense_cancers']:,} cancers within a year), the question priced here is which of them "
        f"should get a supplemental MRI. At the Medicare price of ${mri:,.0f}, sending every "
        f"dense-breast woman costs less than sending none once a cancer found late is charged, "
        f"and the panel's best threshold sends {bp['best_share_sent']:.0%} of them while catching "
        f"{bp['best_share_caught']:.0%} of the cancers. It saves ${b['panel_saves_vs_simple']:,} "
        f"per 100,000 women against the better simple policy, and ${b['panel_saves_vs_age']:,} "
        f"against triage on age alone."]
    if thousand:
        parts.append(
            f"The saving grows with the price of the scan: at $1,000 per MRI it is "
            f"${thousand['vs_simple']:,} per 100,000 against the better simple policy and "
            f"${thousand['vs_age']:,} against age.")
    if r.get("beats_simple_in_every_sweep") and r.get("beats_age_in_every_sweep"):
        parts.append("It beats both sending everyone and triage on age in every sweep of scan "
                     "price, benefit and life-years. **That is the first decision in this project "
                     "where a panel's triage pays and beats age**, and at Medicare prices it is a "
                     "small amount: the honest reading is that supplemental MRI for dense breasts "
                     "is worth doing broadly, and the panel mostly helps decide who can safely skip "
                     "it when scans are expensive.")
    else:
        parts.append("It does not beat both alternatives in every sweep, so no saving is claimed.")
    parts.append("Illustrative, like the other cost models: false-positive MRI work-ups, "
                 "discounting and the difference between trial and US practice are not priced.")
    return " ".join(parts)


def table_breast_subgroups(_, extra):
    """Breast panel accuracy by race and ethnicity across all 2.39M mammograms."""
    r = extra.get("breast_subgroups")
    if not r:
        return "_Run experiments/bcsc_subgroups_full.py._"
    rows = ["| Group | Cancers | AUC | 95% CI |", "|---|---|---|---|"]
    for g, v in r["groups"].items():
        if v.get("auc") is None:
            rows.append(f"| {g} | {v['events']:,} | too few | |")
        else:
            ci = v.get("auc_ci") or ["", ""]
            rows.append(f"| {g} | {v['events']:,} | {v['auc']:.3f} | {ci[0]} to {ci[1]} |")
    worse = [g for g, v in r["groups"].items() if v.get("materially_worse")]
    note = (f"Scored by cross-fitting over all {r['n_mammograms']:,} mammograms, so every prediction "
            f"comes from a model that did not see it; overall AUC {r['overall_auc']:.3f}. "
            + ("No group is more than 0.05 below the overall figure." if not worse else
               f"More than 0.05 below the overall figure: {', '.join(worse)}."))
    return "\n".join(rows) + "\n\n" + note


def text_youden_sentence(_, extra):
    got = _bowel_base(extra)
    if not got:
        return "_Run experiments/cost_model.py._"
    c, b, total = got
    gone = c["break_even_per_missed_cancer"] <= c["societal_cost_of_a_missed_cancer"]
    return (f"On treatment dollars alone, triage looks like it saves ${b['saving'] / 1e6:,.0f}M "
            f"per 100,000 people on bowel. It does that by missing {b['cancers_missed']:.0f} of "
            f"{total:,} cancers. Price a missed cancer at what health economics conventionally "
            f"prices a life-year and the saving "
            + ("disappears. " if gone else "survives. ")
            + "**A cost argument that counts only treatment dollars and not the person is not "
              "an argument.**")


TABLES = {
    "cv_vs_heldout": table_cv_vs_heldout,
    "shipped": table_shipped,
    "baselines": table_baselines,
    "stability": table_stability,
    "calibration": table_calibration,
    "paper_cohorts": table_paper_cohorts,
    "paper_results": table_paper_results,
    "hero_cost": table_hero_cost,
    "cost_ranking": table_cost_ranking,
    "cost_breakeven": lambda ev, extra: cost_breakeven_table(extra.get("cost") or {}),
    "cost_best": lambda ev, extra: cost_best_table(extra.get("cost") or {}),
    "abstract_cost": text_abstract_cost,
    "operating_point": text_operating_point,
    "youden_sentence": text_youden_sentence,
    "bowel_external": table_bowel_external,
    "breast_vs_age": text_breast_vs_age,
    "lung_loco": text_lung_loco,
    "general_vs_age": text_general_vs_age,
    "prospective_arms": table_prospective_arms,
    "prospective_verdict": text_prospective_verdict,
    "prospective_external": table_prospective_external,
    "prospective_short": text_prospective_short,
    "breast_mri_cost": text_breast_mri_cost,
    "breast_subgroups": table_breast_subgroups,
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
        "external_baseline": load("experiments/external_baseline_strength_result.json", {}),
        "triage_age": load("experiments/triage_on_age_alone_result.json", {}),
        "metrics": load("backend/model_metrics.json", {}),
        "breast_vs_age": load("experiments/bcsc_rule_out_vs_age_result.json", {}),
        "lung_loco": load("experiments/lung_loco_gain_result.json", {}),
        "general_vs_age": load("experiments/general_rule_out_vs_age_result.json", {}),
        "breast_mri": load("experiments/breast_mri_triage_cost_result.json", {}),
        "breast_subgroups": load("experiments/bcsc_subgroups_full_result.json", {}),
        "cost": load("experiments/cost_model_result.json", {}),
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
