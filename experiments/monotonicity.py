"""
Where does a panel's risk move the wrong way as a marker gets worse?

The failure
-----------
The liver panel scored a coherent acute-hepatitis pattern — ALT 300, AST 260,
GGT 200, bilirubin 2.5 — at 3.0 percent. A completely normal patient scored 3.3
percent, and a mild abnormality scored 14.1 percent. The worse the liver
looked, the safer the panel said it was.

It was not a coding error. Only 19 of 35,511 people in that cohort have an ALT
over 250, and among the 1,436 WITH liver disease the highest ALT is 232. Every
high-ALT person in the training data is a non-case, so the model learned that a
very high ALT means no liver disease. That is true of NHANES, whose label is
self-reported "were you ever told you had a liver condition", and false of
medicine: someone in acute hepatitis at the time of the survey has not been told
yet.

A tree has no splits beyond the data, so past that edge it returns whichever
leaf it lands in, at full confidence. The direction it happens to point is
arbitrary.

What this measures
------------------
For every panel and every feature a clinician would expect to move risk in a
known direction, sweep that feature from its 5th to its 99.9th percentile with
everything else held at the training median, and record where predicted risk
moves against expectation.

Two numbers per feature:

    reversal      the largest drop in risk while the marker is getting worse
    cliff_at      the value where risk peaks, beyond which the model stops
                  responding or turns around

A reversal inside the training range is a modelling problem. A reversal beyond
it is an extrapolation problem, and no retraining on the same cohort fixes it.
The two are reported separately because they have different remedies.

Run:  python experiments/monotonicity.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/monotonicity_result.json"

# A caution learned by getting this wrong first.
#
# The first version of this experiment swept one feature at a time with every
# other held at its median, and reported three "reversals inside the training
# range". At least two were artefacts of the method rather than faults in the
# models:
#
#   prostate  raising PSA while holding PSA DENSITY fixed implies a physically
#             larger prostate, which is benign hyperplasia. The model reading
#             that as lower risk is correct, not broken.
#   liver     raising AST while holding ALT at 20 describes an AST/ALT ratio
#             that means something different from hepatitis.
#   breast    sweeping radius_worst alone moved the prediction not at all,
#             because the thirty Wisconsin measurements are geometrically
#             collinear. Moved together from the benign median to the malignant
#             median, the same model goes from 0% to 100%.
#
# So features that move together in a patient are swept together here. A
# single-feature partial dependence plot answers "what does this model do to an
# impossible patient", which is rarely the question worth asking.
COHERENT = {
    # Hepatocellular injury: the transaminases, the bile duct enzymes and
    # bilirubin rise together, and albumin falls as synthetic function fails.
    "liver": {
        "hepatocellular injury": {"alt": "up", "ast": "up", "ggt": "up",
                                  "bilirubin": "up", "alkaline_phosphatase": "up",
                                  "albumin": "down"},
    },
    # Obstructive jaundice from a pancreatic head tumour.
    "pancreatic": {
        "obstruction and marker rise": {"plasma_ca19_9": "up", "bilirubin": "up"},
    },
    "ovarian": {
        "pelvic mass markers": {"ca125": "up", "he4": "up", "cea": "up"},
    },
    # PSA and density rise together when the cause is cancer rather than a
    # larger gland; PI-RADS rises with it.
    "prostate": {
        "suspicious MRI and rising PSA": {"psa": "up", "psa_density": "up",
                                          "pi_rads": "up"},
    },
    # Iron-deficiency anaemia from occult bleeding.
    "colorectal": {
        "iron deficiency": {"hemoglobin": "down", "mcv": "down", "rdw": "up"},
    },
    "breast": {
        "malignant morphology": {f"{b}_{s}": "up"
                                 for s in ("mean", "worst")
                                 for b in ("radius", "perimeter", "area",
                                           "concavity", "concave_points",
                                           "compactness")},
    },
}

# Retained for the single-feature sweep, which is still run and reported
# separately, clearly labelled as the weaker of the two.
#
# "up" means higher values should not lower predicted risk. Only listed where
# the expectation is uncontroversial; ambiguous markers are deliberately left
# out rather than guessed at.
EXPECTED_UP = {
    "liver": ["alt", "ast", "ggt", "bilirubin", "alkaline_phosphatase"],
    "pancreatic": ["plasma_ca19_9", "bilirubin", "glucose"],
    "ovarian": ["ca125", "he4", "cea"],
    "prostate": ["psa", "psa_density", "pi_rads"],
    "colorectal": ["rdw"],
    "general": [],
    "lung": [],
    "breast": ["radius_worst", "area_worst", "concave_points_worst"],
}
# Lower is worse for these.
EXPECTED_DOWN = {
    "liver": ["albumin", "platelets"],
    "colorectal": ["hemoglobin"],
}

STEPS = 40


def sweep(model, X, feat, lo, hi, medians):
    grid = np.linspace(lo, hi, STEPS)
    base = pd.DataFrame([medians[X.columns].to_dict()] * STEPS)
    base[feat] = grid
    return grid, model.predict_proba(base[X.columns])[:, 1]


def coherent_sweep(model, X, spec, medians):
    """
    Move a whole clinical pattern from its 5th percentile to its 99.9th at once.

    Every feature in the pattern travels the same distance through its own
    distribution, so the patient stays physiologically coherent at each step
    instead of acquiring an impossible combination halfway along.
    """
    frames = pd.DataFrame([medians[X.columns].to_dict()] * STEPS)
    for feat, direction in spec.items():
        if feat not in X.columns:
            continue
        lo = float(X[feat].quantile(0.05))
        hi = float(X[feat].quantile(0.999))
        if direction == "down":
            lo, hi = float(X[feat].quantile(0.999)), float(X[feat].quantile(0.05))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        frames[feat] = np.linspace(lo, hi, STEPS)
    return model.predict_proba(frames[X.columns])[:, 1]


def main():
    results = {}
    for cfg in tm.DATASETS:
        name = cfg["name"]
        if name in tm.WITHDRAWN:
            continue
        up = EXPECTED_UP.get(name, [])
        down = EXPECTED_DOWN.get(name, [])
        if not up and not down:
            continue

        X, y, medians = tm.prepare(cfg)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        y = pd.Series(y).astype(int)

        model = CalibratedClassifierCV(
            tm.build_ensemble(len(y), float(y.mean())), method="isotonic",
            cv=StratifiedKFold(5, shuffle=True, random_state=0))
        model.fit(X, y)

        panel = {}
        print(f"=== {name} ===", flush=True)

        # The coherent sweep first, because it is the one worth believing.
        for label, spec in (COHERENT.get(name) or {}).items():
            risk = coherent_sweep(model, X, spec, medians)
            peak_i = int(np.argmax(risk))
            drop = float(risk[peak_i] - risk[peak_i:].min())
            rise = float(risk[-1] - risk[0])
            panel[f"pattern: {label}"] = {
                "kind": "coherent pattern",
                "risk_at_healthiest": round(float(risk[0]) * 100, 1),
                "risk_at_worst": round(float(risk[-1]) * 100, 1),
                "peak_risk": round(float(risk[peak_i]) * 100, 1),
                "rise_pct_points": round(rise * 100, 1),
                "reversal_pct_points": round(drop * 100, 1),
                "reverses": bool(drop > 0.02),
            }
            verdict = ("REVERSES" if drop > 0.02
                       else "rises as expected" if rise > 0.02
                       else "barely responds")
            print(f"  [pattern] {label:<30} {risk[0]*100:>5.1f}% -> "
                  f"{risk[-1]*100:>5.1f}%   {verdict}", flush=True)
        for feat, direction in [(f, "up") for f in up] + [(f, "down") for f in down]:
            if feat not in X.columns:
                continue
            lo = float(X[feat].quantile(0.05))
            hi = float(X[feat].quantile(0.999))
            p99 = float(X[feat].quantile(0.99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            grid, risk = sweep(model, X, feat, lo, hi, medians)
            if direction == "down":
                grid, risk = grid[::-1], risk[::-1]

            peak_i = int(np.argmax(risk))
            after = risk[peak_i:]
            reversal = float(risk[peak_i] - after.min()) if len(after) else 0.0
            peak_at = float(grid[peak_i])

            # Was the reversal inside the data, or past its edge?
            edge = p99 if direction == "up" else float(X[feat].quantile(0.01))
            inside = (peak_at < edge) if direction == "up" else (peak_at > edge)

            entry = {
                "direction_expected": direction,
                "risk_at_worst": round(float(risk[-1]) * 100, 1),
                "peak_risk": round(float(risk[peak_i]) * 100, 1),
                "peak_at_value": round(peak_at, 2),
                "p99_of_training": round(p99, 2),
                "reversal_pct_points": round(reversal * 100, 1),
                "reversal_inside_training_range": bool(inside and reversal > 0.02),
            }
            panel[feat] = entry
            flag = ""
            if entry["reversal_pct_points"] >= 2.0:
                flag = ("   <-- REVERSES, inside the data"
                        if entry["reversal_inside_training_range"]
                        else "   <-- reverses past the edge of the data")
            print(f"  {feat:<24} peak {entry['peak_risk']:>5.1f}% at {peak_at:>8.2f}"
                  f"   worst-case {entry['risk_at_worst']:>5.1f}%"
                  f"   drop {entry['reversal_pct_points']:>5.1f}pp{flag}", flush=True)
        results[name] = panel

    worst = [(n, f, v) for n, p in results.items() for f, v in p.items()
             if v.get("kind") != "coherent pattern" and v["reversal_pct_points"] >= 2.0]
    patterns = [(n, f, v) for n, p in results.items() for f, v in p.items()
                if v.get("kind") == "coherent pattern"]
    bad_patterns = [p for p in patterns if p[2]["reverses"]]
    inside = [w for w in worst if w[2]["reversal_inside_training_range"]]

    print("\n" + "=" * 84)
    print(f"  {len(worst)} feature(s) where risk falls by 2 points or more as the marker worsens")
    print(f"  {len(inside)} of those reverse INSIDE the training range, which is a modelling problem")
    print(f"  {len(worst) - len(inside)} reverse past the edge of the data, which clipping addresses "
          f"and retraining on the same cohort does not")

    with open(OUT, "w") as f:
        json.dump({"panels": results,
                   "n_patterns": len(patterns),
                   "n_patterns_reversing": len(bad_patterns),
                   "n_reversals": len(worst),
                   "n_reversals_inside_training_range": len(inside)}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
