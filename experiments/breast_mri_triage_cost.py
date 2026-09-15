"""
Can the breast panel decide which women with dense breasts get a supplemental MRI?

Why this decision
-----------------
The cost model never priced the breast panel, because a woman whose mammogram
has just come back normal is not normally sent anywhere. There is one exception
that is real, expensive and argued about: women with dense breasts, where a
mammogram misses more cancers and supplemental MRI finds them. The DENSE trial
(NEJM 2019) offered MRI to women with extremely dense breasts and a normal
mammogram; MRI found 16.5 cancers per 1,000 women screened and halved the
interval-cancer rate, from 5.0 to 2.5 per 1,000.

So the question is concrete: among women with heterogeneously or extremely
dense breasts, per 100,000 of them, is it better to

    send none for MRI,
    send all of them,
    send only those the panel ranks highest, or
    send only those an age-only model ranks highest?

The last comparison is the one that withdrew the bowel panel. A triage rule has
to beat age, not just beat sending everyone.

Inputs, and where they come from
--------------------------------
    MRI price           $366, Medicare HCPCS C8906; swept up to commercial prices
    treatment cost      first-year allowed cost, Blumen et al. 2016 (MarketScan):
                        stage I/II $82,121, stage III $129,387. A cancer found
                        late is charged the difference.
    benefit fraction    the share of these cancers whose detection MRI brings
                        earlier. DENSE halved interval cancers, so 0.5 is the
                        default; swept 0.25 to 0.75.
    life-years lost     for a cancer found late rather than early. Not measured
                        here; swept 1 to 5, default 3.
    willingness to pay  $150,000 per life-year, as in cost_model.py.
    cancer rate         measured: cancer within one year in dense-breast women on
                        BCSC's 597,859-mammogram validation split.

What this is not
----------------
An illustrative model, the same as cost_model.py. It ignores false-positive MRI
work-ups and biopsies (which would count against sending more women), discounting,
and the difference between MRI access in the Netherlands and the US. Every
assumption is swept, and the verdict is only reported if it holds across the
sweep.

Run:  python experiments/breast_mri_triage_cost.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/breast_mri_triage_cost_result.json"
NAME = "breast_screening"
VALID = "data/bcsc_breast_validation.csv"
COHORT = 100_000
WTP = 150_000

BASE = {"mri_cost": 366.0, "early_cost": 82_121.0, "late_cost": 129_387.0,
        "benefit_fraction": 0.5, "life_years_lost": 3.0}
SWEEPS = {
    "mri_cost": [366.0, 700.0, 1_000.0, 1_500.0],
    "benefit_fraction": [0.25, 0.5, 0.75],
    "life_years_lost": [1.0, 3.0, 5.0],
}


def policy_costs(y, p, w, prevalence, cfg):
    """Cost per 100,000 dense-breast women at every threshold, and the best one."""
    cancers = COHORT * prevalence
    loss = cfg["benefit_fraction"] * ((cfg["late_cost"] - cfg["early_cost"])
                                      + cfg["life_years_lost"] * WTP)
    df = pd.DataFrame({"p": p, "pos": w * (y == 1), "all": w})
    g = df.groupby("p", sort=True)[["pos", "all"]].sum().iloc[::-1]
    pos_tot, all_tot = g["pos"].sum(), g["all"].sum()
    sent = np.concatenate([[0.0], (g["all"].cumsum() / all_tot).to_numpy()])
    caught = np.concatenate([[0.0], (g["pos"].cumsum() / pos_tot).to_numpy()])
    cost = COHORT * sent * cfg["mri_cost"] + cancers * (1 - caught) * loss
    i = int(np.argmin(cost))
    return {"cost_none": float(cost[0]), "cost_all": float(cost[-1]),
            "best_cost": float(cost[i]), "best_share_sent": float(sent[i]),
            "best_share_caught": float(caught[i])}


def main():
    cfg_p = next(c for c in tm.DATASETS if c["name"] == NAME)
    bundle = joblib.load(f"backend/models/model_{NAME}.joblib")
    va = pd.read_csv(VALID)

    med = pd.Series(bundle["feature_medians"])
    X = tm.build_features(cfg_p, va).fillna(med)
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    p_panel = bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]

    tr = pd.read_csv(os.path.join(tm.DATA_DIR, cfg_p["file"]))
    age_tr = tm.build_features(cfg_p, tr)[["age"]]
    age_med = age_tr["age"].median()
    age_model = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=0))
    age_model.fit(age_tr.fillna(age_med), tr["cancer"].astype(int))
    p_age = age_model.predict_proba(tm.build_features(cfg_p, va)[["age"]].fillna(age_med))[:, 1]

    dense = va["density"].isin([3, 4]).to_numpy()
    y = va["cancer"].astype(int).to_numpy()[dense]
    w = va["count"].to_numpy(dtype=float)[dense]
    pp, pa = p_panel[dense], p_age[dense]
    prevalence = float(w[y == 1].sum() / w.sum())

    print(f"dense-breast women on the validation split: {int(w.sum()):,} mammograms, "
          f"{int(w[y == 1].sum()):,} cancers within a year ({prevalence:.2%})\n")

    def run(cfg):
        panel = policy_costs(y, pp, w, prevalence, cfg)
        age = policy_costs(y, pa, w, prevalence, cfg)
        simple = min(panel["cost_none"], panel["cost_all"])
        return {"panel": panel, "age": age,
                "best_simple_policy": "none" if panel["cost_none"] <= panel["cost_all"] else "all",
                "panel_saves_vs_simple": round(simple - panel["best_cost"]),
                "panel_saves_vs_age": round(age["best_cost"] - panel["best_cost"])}

    base = run(BASE)
    print(f"base case (MRI ${BASE['mri_cost']:.0f}, benefit {BASE['benefit_fraction']}, "
          f"{BASE['life_years_lost']:.0f} life-years), per {COHORT:,} dense-breast women:")
    print(f"  MRI for none   ${base['panel']['cost_none']:>14,.0f}")
    print(f"  MRI for all    ${base['panel']['cost_all']:>14,.0f}")
    print(f"  panel triage   ${base['panel']['best_cost']:>14,.0f}   sends "
          f"{base['panel']['best_share_sent']:.0%}, catches {base['panel']['best_share_caught']:.0%}")
    print(f"  age triage     ${base['age']['best_cost']:>14,.0f}   sends "
          f"{base['age']['best_share_sent']:.0%}, catches {base['age']['best_share_caught']:.0%}")
    print(f"  panel saves ${base['panel_saves_vs_simple']:+,} vs the better simple policy "
          f"({base['best_simple_policy']}), and ${base['panel_saves_vs_age']:+,} vs age triage\n")

    sweeps, beats_simple, beats_age = {}, [], []
    for key, values in SWEEPS.items():
        row = {}
        for v in values:
            r = run(dict(BASE, **{key: v}))
            row[str(v)] = {"vs_simple": r["panel_saves_vs_simple"], "vs_age": r["panel_saves_vs_age"],
                           "simple": r["best_simple_policy"]}
            beats_simple.append(r["panel_saves_vs_simple"] > 0)
            beats_age.append(r["panel_saves_vs_age"] > 0)
        sweeps[key] = row
        print(f"  sweep {key:<17} " + "  ".join(
            f"{v}: {row[str(v)]['vs_simple']:+,.0f} / {row[str(v)]['vs_age']:+,.0f}" for v in values))

    robust_simple, robust_age = all(beats_simple), all(beats_age)
    print(f"\n  (each cell: panel saving vs the better simple policy / vs age triage)")
    print(f"  -> panel triage beats sending all or none in every sweep: {robust_simple}")
    print(f"  -> panel triage beats age triage in every sweep:          {robust_age}")

    with open(OUT, "w") as f:
        json.dump({"dense_mammograms": int(w.sum()), "dense_cancers": int(w[y == 1].sum()),
                   "prevalence": round(prevalence, 5), "base_inputs": BASE, "base": base,
                   "sweeps": sweeps, "beats_simple_in_every_sweep": robust_simple,
                   "beats_age_in_every_sweep": robust_age,
                   "sources": {"mri_cost": "Medicare HCPCS C8906",
                               "treatment_cost": "Blumen et al., Am Health Drug Benefits 2016",
                               "benefit_fraction": "DENSE trial, Bakker et al., NEJM 2019"}},
                  f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
