"""
What would it cost to stop flagging 800 healthy people per cancer found?

The problem
-----------
At real prevalence the pancreatic panel flags roughly 843 people per true case
and the bowel panel roughly 768. Both have good AUCs. Neither is usable as a
population screen at those numbers, and no better model fixes it, because the
arithmetic is driven by prevalence rather than by discrimination.

But part of it IS fixable, and that part has been ignored.

Every panel currently picks its threshold by Youden's J, which maximises
sensitivity plus specificity. That is the standard choice and it is the wrong
objective here, because it weights a false positive and a false negative
equally and takes no account of how rare the disease is. On a condition with a
prevalence of 0.04 percent, the false positives outnumber the true ones by
orders of magnitude no matter how the two error rates trade off.

So this measures the alternative rather than asserting one. For each panel it
sweeps the threshold and reports what precision is achievable and what
sensitivity it costs.

Three operating points per panel
--------------------------------
    youden      what ships today, maximum sensitivity plus specificity
    per_case    the tightest threshold that gets flagged-per-true-case to the
                target below, if one exists
    high_spec   fixed at 99 percent specificity, the conventional screening
                choice, shown because it is what a real programme would use

The honest question each panel has to answer is whether there EXISTS a threshold
that is simultaneously precise enough to act on and sensitive enough to be worth
running. For some of these panels the answer is no, and that is the finding.

Run:  python experiments/operating_point.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import SEER_INCIDENCE, projected_ppv_npv

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_PER_CASE = 20.0     # flagging 20 people to find 1 cancer is arguably actionable
MIN_USEFUL_SENS = 0.30     # below this the panel misses so much it is not worth running
OUT = "experiments/operating_point_result.json"


def stats_at(y, p, thr, prev):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv, npv, per_case = projected_ppv_npv(sens, spec, prev)
    return {"threshold": round(float(thr), 5), "sensitivity": round(float(sens), 3),
            "specificity": round(float(spec), 3),
            "ppv_at_prevalence": round(float(ppv), 5) if np.isfinite(ppv) else None,
            "flagged_per_true_case": round(float(per_case), 1) if np.isfinite(per_case) else None}


def main():
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN:
            continue
        prev = SEER_INCIDENCE[name][0] / 100_000.0

        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)

        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        p = cross_val_predict(
            CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                                   method="isotonic", cv=cv),
            X, y, cv=cv, method="predict_proba")[:, 1]

        # What ships today.
        shipped_thr = tm.choose_threshold(y, p)
        entry = {"prevalence_used": prev, "auc": round(float(roc_auc_score(y, p)), 3),
                 "points": {"youden (shipped)": stats_at(y, p, shipped_thr, prev)}}

        # Fixed 99 percent specificity.
        thr99 = float(np.quantile(p[y == 0], 0.99))
        entry["points"]["99% specificity"] = stats_at(y, p, thr99, prev)

        # Tightest threshold reaching the per-case target.
        best = None
        for q in np.linspace(0.5, 0.99999, 400):
            t = float(np.quantile(p, q))
            st = stats_at(y, p, t, prev)
            pc = st["flagged_per_true_case"]
            if pc is not None and pc <= TARGET_PER_CASE:
                best = st
                break
        entry["points"][f"<= {TARGET_PER_CASE:.0f} flagged per case"] = best

        print(f"\n=== {name} ===  prevalence {prev*100:.4f}%, AUC {entry['auc']}")
        for label, st in entry["points"].items():
            if st is None:
                print(f"  {label:<32} unreachable at any threshold")
                continue
            print(f"  {label:<32} thr {st['threshold']:<8} sens {st['sensitivity']:<6} "
                  f"spec {st['specificity']:<6} flagged/case "
                  f"{st['flagged_per_true_case']}")

        target = entry["points"][f"<= {TARGET_PER_CASE:.0f} flagged per case"]
        if target is None:
            verdict = "NO usable operating point exists"
        elif target["sensitivity"] < MIN_USEFUL_SENS:
            verdict = (f"precision reachable but only at {target['sensitivity']:.0%} "
                       f"sensitivity, so it would miss most cancers")
        else:
            verdict = (f"usable: {target['flagged_per_true_case']} flagged per case at "
                       f"{target['sensitivity']:.0%} sensitivity")
        entry["verdict"] = verdict
        print(f"  -> {verdict}")
        results[name] = entry

    print("\n" + "=" * 88)
    print(f"{'panel':<12}{'shipped/case':>14}{'best reachable':>17}{'sens there':>12}   verdict")
    print("=" * 88)
    for n, e in results.items():
        sh = e["points"]["youden (shipped)"]["flagged_per_true_case"]
        tg = e["points"][f"<= {TARGET_PER_CASE:.0f} flagged per case"]
        reach = str(tg["flagged_per_true_case"]) if tg else "none"
        sens = f"{tg['sensitivity']:.2f}" if tg else "-"
        print(f"{n:<12}{str(sh):>14}{reach:>17}{sens:>12}   {e['verdict'][:40]}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
