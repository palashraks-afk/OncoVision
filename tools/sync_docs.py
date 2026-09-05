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
DOCS = ["README.md", "PROJECT.md"]

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


TABLES = {
    "shipped": table_shipped,
    "baselines": table_baselines,
    "stability": table_stability,
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
