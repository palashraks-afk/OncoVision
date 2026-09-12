"""
If age and sex alone triage as well as the panel, the bloodwork is not what pays.

The cost model found one panel whose triage pays once a missed cancer is priced
at a life: bowel, avoiding tens of thousands of colonoscopies per 100,000
people. That was read as bloodwork earning its keep. baseline_strength.py then
found that a logistic model on age and sex alone discriminates bowel cancer as
well as, or better than, the shipped panel.

Those two facts are only compatible if most of the saving comes from age. So the
question here is the counterfactual the cost model never ran: take the SAME
operating-point search, the same prices, the same incidence, and run it on an
age-and-sex model instead of the panel. If age and sex alone reach a similar
net benefit, then "triage on free bloodwork saves money" should read "triage on
age saves money", which is a much older finding and a much weaker product.

Inputs are the out-of-fold scores baseline_strength.py saves, so both arms are
scored on the same people from the same folds.

Run:  python experiments/baseline_strength.py   (first)
      python experiments/triage_on_age_alone.py
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as cm  # noqa: E402

OUT = "experiments/triage_on_age_alone_result.json"
# The same valuation cost_model.py applies inside main(): a missed case costs
# its life-years lost at the conventional US willingness-to-pay per QALY.
WTP_PER_QALY = 150_000


def best_point(y, p, cfg):
    fpr, tpr, _ = roc_curve(y, p)
    societal = WTP_PER_QALY * cfg["life_years_lost"]
    best = None
    for sens, spec in zip(tpr, 1.0 - fpr):
        tp, fp, fn = cm.outcomes(cm.COHORT, cfg["incidence"], float(sens), float(spec))
        net = (cm.COHORT - (tp + fp)) * cfg["procedure_cost"] - fn * societal
        if best is None or net > best["net_benefit"]:
            best = {"sensitivity": round(float(sens), 3),
                    "specificity": round(float(spec), 3),
                    "procedures_avoided": int(round(cm.COHORT - (tp + fp))),
                    "cancers_missed": round(float(fn), 1),
                    "net_benefit": round(float(net))}
    return best


def main():
    results = {}
    for panel, cfg in cm.SETTINGS.items():
        path = f"experiments/baseline_strength_oof_{panel}.npz"
        if not os.path.isfile(path):
            print(f"{panel}: {path} missing, run baseline_strength.py first")
            continue
        d = np.load(path)
        y = d["y"]
        arms = {k: best_point(y, d[k], cfg)
                for k in ("full_logistic", "full_ensemble", "base_logistic", "base_ensemble")}
        best_panel = max(("full_logistic", "full_ensemble"), key=lambda k: arms[k]["net_benefit"])
        best_base = max(("base_logistic", "base_ensemble"), key=lambda k: arms[k]["net_benefit"])
        diff = arms[best_panel]["net_benefit"] - arms[best_base]["net_benefit"]
        results[panel] = {"arms": arms, "best_panel_arm": best_panel,
                          "best_baseline_arm": best_base,
                          "panel_minus_age_and_sex": diff}

        print(f"=== {panel} ===  {cfg['procedure']}, per {cm.COHORT:,} people")
        for k, a in arms.items():
            print(f"    {k:<15} avoids {a['procedures_avoided']:>7,}  misses "
                  f"{a['cancers_missed']:>5}  net ${a['net_benefit']:>13,}")
        verdict = ("the panel earns more than age and sex alone" if diff > 0
                   else "AGE AND SEX ALONE EARN AS MUCH OR MORE")
        print(f"    panel minus age and sex: ${diff:+,}  -> {verdict}\n", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
