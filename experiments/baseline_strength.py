"""
Is "gain over age and sex" a gain, or a weak baseline?

What prompted this
------------------
demographic_gain.py was changed to fit both arms with the model kind that
actually ships, instead of always using the ensemble. The bowel panel's gain
went from +0.033, winning 5 of 5 repeats, to -0.011, losing all 5. The held-out
split says the same thing: the shipped logistic panel scores 0.821 and a
logistic age-and-sex model 0.843.

The likely reason is not the panel. It is the baseline. A calibrated tree
ensemble handed only two features -- age, and a binary sex flag -- ranks people
coarsely: trees cut age into steps, and isotonic calibration flattens it
further. It scored 0.765 on bowel. A logistic regression on the same two
features scores 0.822, because age is the single strongest predictor and a
smooth monotone curve is exactly the shape that suits it.

So every "gain over age and sex" in this project may have been measured
against a baseline hobbled by the wrong model. If so, the claim that routine
bloodwork adds signal needs to be re-earned against the STRONGEST age-and-sex
model available, not the one that happens to share the panel's model kind.

Design
------
For each population panel, four arms on identical folds, repeated:

    full panel,  logistic       full panel,  ensemble
    age and sex, logistic       age and sex, ensemble

and three gains per repeat:

    same kind      the shipped kind against the same kind's baseline
    vs ensemble    the shipped kind against the ensemble baseline (the old way)
    vs strongest   the shipped kind against whichever baseline scored higher
                   on THAT repeat -- the honest one

Only the last one answers whether the bloodwork is doing anything.

Run:  python experiments/baseline_strength.py
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

warnings.filterwarnings("ignore")

OUT = "experiments/baseline_strength_result.json"
PANELS = ["colorectal", "general", "lung", "liver"]
REPEATS = 5


def oof(X, y, seed, kind):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(
        CalibratedClassifierCV(tm.model_factory(kind, len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]


def main():
    results = {}
    if os.path.isfile(OUT):
        results = json.load(open(OUT))
    for name in PANELS:
        cfg = next(c for c in tm.DATASETS if c["name"] == name)
        X, y, _ = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int).reset_index(drop=True)
        base_feats = [f for f in ("age", "gender") if f in X.columns]
        shipped, _ = tm.select_model(X, y, float(y.mean()))

        auc = {k: [] for k in ("full_logistic", "full_ensemble",
                               "base_logistic", "base_ensemble")}
        oof_store = {}
        for s in range(REPEATS):
            for kind in ("logistic", "ensemble"):
                pf = oof(X, y, s, kind)
                pb = oof(X[base_feats], y, s, kind)
                auc[f"full_{kind}"].append(roc_auc_score(y, pf))
                auc[f"base_{kind}"].append(roc_auc_score(y, pb))
                if s == 0:
                    oof_store[f"full_{kind}"] = pf
                    oof_store[f"base_{kind}"] = pb

        full = np.array(auc[f"full_{shipped}"])
        same = full - np.array(auc[f"base_{shipped}"])
        vs_ens = full - np.array(auc["base_ensemble"])
        strongest = full - np.maximum(auc["base_logistic"], auc["base_ensemble"])
        # The best the panel could do with either kind, against the best
        # baseline: if even this is not positive, no model choice rescues it.
        best_full = np.maximum(auc["full_logistic"], auc["full_ensemble"])
        best_vs_best = best_full - np.maximum(auc["base_logistic"], auc["base_ensemble"])

        def summ(d):
            return {"mean": round(float(d.mean()), 3),
                    "range": [round(float(d.min()), 3), round(float(d.max()), 3)],
                    "wins": int((d > 0).sum()), "repeats": REPEATS}

        entry = {
            "shipped_kind": shipped,
            "n": int(len(y)), "events": int(y.sum()),
            "mean_auc": {k: round(float(np.mean(v)), 3) for k, v in auc.items()},
            "gain_same_kind": summ(same),
            "gain_vs_ensemble_baseline": summ(vs_ens),
            "gain_vs_strongest_baseline": summ(strongest),
            "best_panel_vs_best_baseline": summ(best_vs_best),
            "bloodwork_adds_signal": bool(best_vs_best.mean() > 0 and (best_vs_best > 0).sum() >= 4),
        }
        results[name] = entry
        print(f"=== {name} ===  n={entry['n']:,} events={entry['events']}  ships {shipped}")
        for k, v in entry["mean_auc"].items():
            print(f"    {k:<15} {v:.3f}")
        print(f"    gain, same kind            {entry['gain_same_kind']['mean']:+.3f}  "
              f"wins {entry['gain_same_kind']['wins']}/{REPEATS}")
        print(f"    gain, vs ensemble baseline {entry['gain_vs_ensemble_baseline']['mean']:+.3f}  "
              f"(the old measurement)")
        print(f"    gain, vs strongest baseline {entry['gain_vs_strongest_baseline']['mean']:+.3f}  "
              f"wins {entry['gain_vs_strongest_baseline']['wins']}/{REPEATS}")
        print(f"    best panel vs best baseline {entry['best_panel_vs_best_baseline']['mean']:+.3f}  "
              f"wins {entry['best_panel_vs_best_baseline']['wins']}/{REPEATS}"
              f"  -> {'bloodwork adds signal' if entry['bloodwork_adds_signal'] else 'NO SIGNAL BEYOND AGE AND SEX'}",
              flush=True)

        # Out-of-fold scores from repeat 0, kept so the cost model can ask
        # whether triage on age and sex ALONE pays as well as the panel does.
        np.savez_compressed(f"experiments/baseline_strength_oof_{name}.npz",
                            y=y.to_numpy(), **oof_store)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
