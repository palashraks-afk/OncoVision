"""
Does the mammogram-report breast panel's rule-out cut beat a cut on age alone?

Why this has to be asked
------------------------
The bowel panel's rule-out point appeared to pay: it took tens of thousands of
people per 100,000 out of the colonoscopy queue. It paid because age does. The
same search on age and sex alone took out a similar share and missed half as
many cancers, and the panel was withdrawn.

The breast panel ships a rule-out cut too, and it kept its promise on the
consortium's held-out split: promised to catch 95.1% and exclude 11.0%,
delivered 95.3% and 11.6%. That is necessary and not sufficient. A cut that
keeps its promise can still be one that age alone would have kept better.

Design
------
An age-only model is fitted on the SAME 400,000-mammogram training sample the
panel saw, its cut is chosen the same way the panel's was (the highest threshold
whose out-of-fold sensitivity stays at 95%), and both cuts are applied unchanged
to the 597,859-mammogram validation split, weighted by count.

Then the comparison that decides it: at the SAME share of cancers caught, which
cut excludes more women? That removes the possibility that one cut looks better
only because it sits at a slightly different sensitivity. The difference gets a
Poisson bootstrap interval over mammograms.

The panel earns its extra questions only if it excludes materially more women at
matched sensitivity.

Run:  python experiments/bcsc_rule_out_vs_age.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/bcsc_rule_out_vs_age_result.json"
NAME = "breast_screening"
VALID = "data/bcsc_breast_validation.csv"
BOOT = 300
# A panel that asks a woman seven extra questions has to exclude at least one
# more woman in a hundred than her age does, at the same sensitivity, to be
# worth the questions. Set before the numbers were known.
MATERIAL = 0.01


def exclusion_at_sensitivity(y, p, w, target):
    """Share of mammograms a cut can exclude while still catching `target`.

    Scores are pooled by unique value so ties move together, and the lowest
    scores are excluded for as long as the cancers among them stay within the
    allowed miss rate.
    """
    df = pd.DataFrame({"p": p, "pos": w * (y == 1), "all": w})
    g = df.groupby("p", sort=True)[["pos", "all"]].sum()
    allowed = (1.0 - target) * g["pos"].sum()
    ok = g["pos"].cumsum() <= allowed + 1e-9
    if not ok.any():
        return 0.0
    return float(g["all"].cumsum()[ok].iloc[-1] / g["all"].sum())


def delivered(y, p, w, cut):
    out = p < cut
    return (float(w[(y == 1) & ~out].sum() / w[y == 1].sum()),
            float(w[out].sum() / w.sum()))


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    ro = (bundle.get("metrics") or {}).get("rule_out")
    if not ro:
        print("the breast panel ships no rule-out cut; nothing to compare")
        return

    tr = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    y_tr = tr["cancer"].astype(int).to_numpy()
    age_tr = tm.build_features(cfg, tr)[["age"]]
    age_med = age_tr["age"].median()
    age_tr = age_tr.fillna(age_med)

    def age_model():
        return CalibratedClassifierCV(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")),
            method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=0))

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = cross_val_predict(age_model(), age_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    age_ro = tm.rule_out_threshold(y_tr, oof)
    age_fit = age_model().fit(age_tr, y_tr)

    va = pd.read_csv(VALID)
    y = va["cancer"].astype(int).to_numpy()
    w = va["count"].to_numpy(dtype=float)

    med = pd.Series(bundle["feature_medians"])
    X = tm.build_features(cfg, va).fillna(med)
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    p_panel = bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]
    p_age = age_fit.predict_proba(tm.build_features(cfg, va)[["age"]].fillna(age_med))[:, 1]

    panel_caught, panel_excl = delivered(y, p_panel, w, ro["threshold"])
    age_caught, age_excl = (delivered(y, p_age, w, age_ro["threshold"])
                            if age_ro else (float("nan"), float("nan")))

    target = panel_caught
    m_panel = exclusion_at_sensitivity(y, p_panel, w, target)
    m_age = exclusion_at_sensitivity(y, p_age, w, target)

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(BOOT):
        wb = rng.poisson(w).astype(float)
        diffs.append(exclusion_at_sensitivity(y, p_panel, wb, target)
                     - exclusion_at_sensitivity(y, p_age, wb, target))
    ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    diff = m_panel - m_age
    earns = bool(ci[0] > 0 and diff >= MATERIAL)

    n, cancers = int(w.sum()), int(w[y == 1].sum())
    print(f"BCSC validation split: {n:,} mammograms, {cancers:,} cancers\n")
    print(f"  shipped cuts, applied unchanged")
    print(f"    panel      catches {panel_caught:.1%}  excludes {panel_excl:.1%}")
    print(f"    age alone  catches {age_caught:.1%}  excludes {age_excl:.1%}")
    print(f"\n  at the same share of cancers caught ({target:.1%})")
    print(f"    panel      excludes {m_panel:.1%}")
    print(f"    age alone  excludes {m_age:.1%}")
    print(f"    difference {diff:+.1%}  (95% CI {ci[0]:+.1%} to {ci[1]:+.1%})")
    print(f"\n  -> {'the panel earns its questions: it excludes materially more women than age' if earns else 'THE PANEL DOES NOT MATERIALLY BEAT AGE ALONE AT THIS CUT'}")

    with open(OUT, "w") as f:
        json.dump({
            "n_mammograms": n, "n_cancers": cancers,
            "panel_cut": {"threshold": ro["threshold"], "caught": round(panel_caught, 3),
                          "excluded": round(panel_excl, 3)},
            "age_cut": ({"threshold": age_ro["threshold"], "caught": round(age_caught, 3),
                         "excluded": round(age_excl, 3)} if age_ro else None),
            "matched_sensitivity": round(target, 3),
            "panel_excluded_at_matched": round(m_panel, 4),
            "age_excluded_at_matched": round(m_age, 4),
            "difference": round(diff, 4), "difference_ci": ci,
            "material_threshold": MATERIAL,
            "panel_beats_age": earns,
        }, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
