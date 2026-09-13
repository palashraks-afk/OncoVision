"""
Is the lung panel's gain over age and sex real? Every cycle held out in turn.

The problem
-----------
Inside its training cycles the lung panel beats the strongest age-and-sex model
by +0.026, winning 5 of 5 repeats. On the one withheld cycle, 2017-2018, it
measured -0.022 with an interval from -0.126 to +0.056 -- thirteen events, too
few to confirm or refute anything. No external cohort carries serum cotinine
and CRP, which the panel reads.

What this does
--------------
Leave-one-cycle-out across all ten NHANES cycles, 1999 to 2018. For each cycle,
the panel and both age-and-sex models are fitted on the other nine and score the
held-out one. Every prediction is therefore made on a cycle the model never saw,
and pooling them uses all 117 lung cancers instead of thirteen.

What it is and is not
---------------------
It is within one survey. Cycles share a protocol and a laboratory contract, and
section 4.2 of the paper records a leave-one-cycle-out result that looked like
transfer and was not. So a positive result here strengthens the in-survey claim;
it does not replace an external cohort, and it is written up that way. A negative
one would settle the question against the panel.

The baseline is the better of logistic regression and the ensemble on age and
sex, chosen on the pooled predictions -- which favours the baseline, not the
panel. The gain gets a paired bootstrap interval over people.

Run:  python experiments/lung_loco_gain.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/lung_loco_gain_result.json"
NAME = "lung"
BASE = ["age", "gender"]
BOOT = 2000


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    # All cycles, the withheld one included: every cycle is held out in turn, so
    # nothing is scored by a model that saw it.
    df = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    X = tm.build_features(cfg, df)
    y = cfg["target"](df)
    keep = y.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
    cycles = df.loc[keep.values, "cycle"].astype(str).reset_index(drop=True)
    base_cols = [c for c in BASE if c in X.columns]

    X_tr0, y_tr0, _ = tm.prepare(cfg)
    kind, _ = tm.select_model(X_tr0, y_tr0, float(y_tr0.mean()))

    p_panel = np.full(len(y), np.nan)
    p_base = {k: np.full(len(y), np.nan) for k in ("logistic", "ensemble")}
    for cyc in sorted(cycles.unique()):
        te = (cycles == cyc).to_numpy()
        tr = ~te
        med = X[tr].median()
        Xtr, Xte = X[tr].fillna(med), X[te].fillna(med)
        ytr = y[tr]
        folds = max(2, min(5, int(ytr.value_counts().min())))
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        rate = float(ytr.mean())
        m = CalibratedClassifierCV(tm.model_factory(kind, len(ytr), rate),
                                   method="isotonic", cv=cv).fit(Xtr, ytr)
        p_panel[te] = m.predict_proba(Xte)[:, 1]
        for k in p_base:
            b = CalibratedClassifierCV(tm.model_factory(k, len(ytr), rate),
                                       method="isotonic", cv=cv).fit(Xtr[base_cols], ytr)
            p_base[k][te] = b.predict_proba(Xte[base_cols])[:, 1]
        print(f"  held out {cyc}: n={int(te.sum()):,}  cancers={int(y[te].sum())}", flush=True)

    yv = y.to_numpy()
    base_kind = max(p_base, key=lambda k: roc_auc_score(yv, p_base[k]))
    pb = p_base[base_kind]
    auc_panel, auc_base = roc_auc_score(yv, p_panel), roc_auc_score(yv, pb)

    rng = np.random.default_rng(0)
    gains = []
    for _ in range(BOOT):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].sum() == 0:
            continue
        gains.append(roc_auc_score(yv[i], p_panel[i]) - roc_auc_score(yv[i], pb[i]))
    ci = [round(float(np.percentile(gains, 2.5)), 3), round(float(np.percentile(gains, 97.5)), 3)]
    gain = auc_panel - auc_base
    confirmed = bool(ci[0] > 0)

    print(f"\nlung panel ({kind}) vs age and sex ({base_kind}), pooled over "
          f"{cycles.nunique()} held-out cycles, {int(yv.sum())} cancers")
    print(f"  panel {auc_panel:.3f}  age and sex {auc_base:.3f}  gain {gain:+.3f} "
          f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f})")
    print(f"  -> {'the gain holds on every held-out cycle pooled' if confirmed else 'THE GAIN IS NOT CONFIRMED ACROSS HELD-OUT CYCLES'}")

    with open(OUT, "w") as f:
        json.dump({"panel_model": kind, "baseline_model": base_kind,
                   "cycles": int(cycles.nunique()), "n": int(len(yv)),
                   "events": int(yv.sum()),
                   "panel_auc": round(float(auc_panel), 3),
                   "age_sex_auc": round(float(auc_base), 3),
                   "gain": round(float(gain), 3), "gain_ci": ci,
                   "gain_confirmed_within_survey": confirmed}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
