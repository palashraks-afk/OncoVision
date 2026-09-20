"""
Regenerate the FALLBACK_METRICS block in page.tsx from backend/model_metrics.json.

Why this exists
---------------
The frontend shows measured numbers on every panel card. It reads them from the
live registry endpoint, and falls back to a copy compiled into the page for the
moment before that responds, or when the backend is cold.

That copy was maintained by hand. Which means that every time a panel was
retrained, the fallback kept displaying the previous run's numbers until someone
remembered to retype eight objects. It drifted, and a drifted fallback is worse
than none: it shows a number that was true once, with no indication that it is
stale, on a page whose entire argument is that its numbers are measured.

So it is generated. Run this after train_models.py and the block matches what
the backend will serve.

Run:  python tools/sync_fallback_metrics.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "frontend/src/app/page.tsx")
METRICS = os.path.join(ROOT, "backend/model_metrics.json")

START = "const FALLBACK_METRICS: Record<string, any> = {"
END = "};"

# Emitted in this order, so a diff between two runs shows changed numbers rather
# than reordered keys.
FIELDS = [
    "label", "auc", "auc_ci", "threshold", "sensitivity", "specificity",
    "brier", "calibration_slope", "ppv_at_population_prevalence",
    "people_flagged_per_true_case", "population_prevalence",
    "cohort_prevalence", "baseline_logistic_auc", "baseline_age_sex_auc",
    "n_samples", "n_test", "n_features",
]

# Roughly how the hand-written block grouped them, purely for readability.
LINE_BREAKS_AFTER = {"auc_ci", "threshold", "specificity", "calibration_slope",
                     "ppv_at_population_prevalence", "people_flagged_per_true_case",
                     "cohort_prevalence", "baseline_age_sex_auc"}


def render(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(render(v) for v in value) + "]"
    if isinstance(value, float):
        # Drop the trailing .0 that json would print for whole numbers.
        return repr(round(value, 6)).rstrip("0").rstrip(".") or "0"
    return str(value)


def flatten(entry):
    """
    Held-out numbers, with the panel-level fields the card also shows.

    This distinction matters and is easy to get wrong. `model_metrics.json` has
    an `auc` at the top level and another inside `held_out`, and they are not
    the same quantity: the top-level one is cross-validated on the training
    data, the nested one is measured on the 20 percent split that was cut before
    anything was fitted. The card says "held-out test results", so it must read
    the nested one. Taking the top level would silently relabel a training
    number as a test number, which flatters every panel.
    """
    held = entry.get("held_out") or {}
    flat = {
        "label": entry.get("label"),
        "threshold": entry.get("threshold"),
        "n_samples": entry.get("n_samples"),
        "n_features": entry.get("n_features"),
    }
    flat.update({k: v for k, v in held.items() if k in FIELDS})
    return {k: v for k, v in flat.items() if v is not None or k in held}


def main():
    raw = json.load(open(METRICS, encoding="utf-8"))
    # A withdrawn panel is still recorded in model_metrics.json, as evidence,
    # with "shipped": False and no held-out block. The page's fallback is what
    # the interface shows when the service is unreachable, and it must not show
    # a panel the service would refuse to serve.
    raw = {k: v for k, v in raw.items() if v.get("shipped", True) is not False}
    metrics = {k: flatten(v) for k, v in raw.items()}
    missing = [k for k, v in metrics.items() if v.get("auc") is None]
    if missing:
        print(f"ERROR: no held-out block for {', '.join(missing)}. "
              f"Run evaluate.py then train_models.py first.", file=sys.stderr)
        return 1
    page = open(PAGE, encoding="utf-8").read()

    start = page.index(START)
    end = page.index("\n" + END, start) + len("\n" + END)

    # Highest AUC first, which is the order the page sorts them into anyway.
    order = sorted(metrics, key=lambda k: -(metrics[k].get("auc") or 0))

    lines = [START]
    for name in order:
        m = metrics[name]
        lines.append(f"  {name}: {{")
        buf = []
        for f in FIELDS:
            if f not in m:
                continue
            buf.append(f"{f}: {render(m[f])},")
            if f in LINE_BREAKS_AFTER:
                lines.append("    " + " ".join(buf))
                buf = []
        if buf:
            lines.append("    " + " ".join(buf))
        lines.append("  },")
    lines.append(END)

    new = page[:start] + "\n".join(lines) + page[end:]
    new = sync_stability(new)
    if new == page:
        print("FALLBACK_METRICS already matches backend/model_metrics.json")
        return 0

    open(PAGE, "w", encoding="utf-8").write(new)
    print(f"regenerated FALLBACK_METRICS for {len(order)} panels:")
    for name in order:
        m = metrics[name]
        print(f"  {name:<12} AUC {m.get('auc')}  {m.get('n_features')} features")
    return 0


STAB = os.path.join(ROOT, "experiments/split_stability_result.json")
STAB_START = "// AUTOGEN:split_stability"
STAB_END = "// /AUTOGEN:split_stability"
STAB_NAMES = {"general": "General", "liver": "Liver", "lung": "Lung",
              "cancer_mortality": "Cancer mortality",
              "breast": "Breast, biopsy", "breast_screening": "Breast, mammogram",
              "pancreatic": "Pancreatic", "ovarian": "Ovarian", "prostate": "Prostate",
              "colorectal": "Bowel"}


def sync_stability(page):
    """The About page's split-stability table, written from the experiment.

    It used to be typed by hand and went stale twice over: it kept a row for a
    withdrawn panel, and its numbers came from a script that fitted the ensemble
    while most panels ship logistic regression. A split outside the 10th to 90th
    percentile of its own distribution is marked, because that held-out AUC is a
    lucky or unlucky draw rather than an estimate.
    """
    if STAB_START not in page or not os.path.exists(STAB):
        return page
    panels = json.load(open(STAB, encoding="utf-8")).get("panels") or {}
    shipped = json.load(open(METRICS, encoding="utf-8"))
    rows = []
    for k, v in sorted(panels.items(), key=lambda kv: -kv[1]["mean_auc"]):
        if shipped.get(k, {}).get("shipped", True) is False:
            continue
        pct = v["shipped_split_percentile"]
        ok = "true" if 10 < pct < 90 else "false"
        rows.append(f'  {{ panel: "{STAB_NAMES.get(k, k)}", mean: {v["mean_auc"]}, '
                    f'spread: "{v["min_auc"]} to {v["max_auc"]}", '
                    f'shipped: {v["shipped_split_auc"]}, pct: {round(pct)}, ok: {ok} }},')
    rows.append('  { panel: "Cervical, withdrawn", mean: 0.594, spread: "0.421 to 0.789", '
                'shipped: 0.725, pct: 97, ok: false },')
    s = page.index(STAB_START)
    e = page.index(STAB_END, s)
    head = page.index("const SPLIT_STABILITY = [", s)
    block = page[s:head] + "const SPLIT_STABILITY = [\n" + "\n".join(rows) + "\n];\n"
    return page[:s] + block + page[e:]


if __name__ == "__main__":
    sys.exit(main())
