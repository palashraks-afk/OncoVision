"""
What happens to a panel when the lab that ran the sample is not the lab it was
trained on, and can training fix it?

Why this is worth measuring
---------------------------
Two things in this project already look like lab-to-lab problems. The liver
panel scored 0.442 on a German cohort, below chance, and the explanation offered
was a different disease stage; the experiments that decide how much of it was
analyser calibration were never run. And every panel here is trained on NHANES,
whose laboratory contract and instruments differ from the hospital a real user's
report comes from. Reference intervals and method biases differ between labs by
a few percent for most analytes and by more for some (albumin depends on the dye
method; enzyme activities depend on temperature and reagent).

Part 1, fragility. The shipped models are scored with every analyte multiplied
by a random bias, one bias per analyte for the whole test cohort, which is what
one different laboratory looks like. Biases are lognormal with a standard
deviation of 0, 5, 10 and 20 percent. These are STRESS levels, not measured
inter-laboratory biases, and they are labelled that way.

Part 2, a fix. Train on data where each batch of patients has its own random
bias, so the model cannot lean on an analyte's exact scale. The natural batches
are the survey cycles, which really do differ in instruments. Same model kind,
same features, only the training data changes.

Pre-registered, written before running
--------------------------------------
Augmented training is adopted for a panel only if, at a 10 percent bias,
  - its worst-decile AUC over 50 random laboratories improves by at least 0.010
    over the plain model, and
  - its AUC with no bias falls by no more than 0.005.
Both cohorts must meet both conditions. A fix that costs accuracy where there is
no problem, or that does nothing where there is one, is not worth the added
training complexity.

Run:  python experiments/lab_shift_robustness.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm

warnings.filterwarnings("ignore")

OUT = "experiments/lab_shift_robustness_result.json"
LEVELS = [0.0, 0.05, 0.10, 0.20]
DRAWS = 50
AUG_SD = 0.10
REPLICAS = 4
ADOPT_GAIN = 0.010
ADOPT_COST = 0.005


def score(bundle, X):
    X = X.copy()
    for col, (lo, hi) in (bundle.get("feature_ranges") or {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    return bundle["model"].predict_proba(X[bundle["feature_names"]])[:, 1]


def lab_bias(X, cols, sd, rng):
    """One random multiplicative bias per analyte, shared by every patient."""
    X = X.copy()
    if sd > 0:
        X[cols] = X[cols] * np.exp(rng.normal(0.0, sd, len(cols)))
    return X


def fragility(predict, X, y, cols, seed=0):
    out = {}
    rng = np.random.default_rng(seed)
    for sd in LEVELS:
        aucs = []
        for _ in range(1 if sd == 0 else DRAWS):
            aucs.append(roc_auc_score(y, predict(lab_bias(X, cols, sd, rng))))
        a = np.array(aucs)
        out[f"{int(sd * 100)}%"] = {"mean": round(float(a.mean()), 4),
                                   "worst_decile": round(float(np.percentile(a, 10)), 4),
                                   "worst": round(float(a.min()), 4)}
    return out


def augmented_training_set(X, y, cols, batches, rng):
    """Replicas of the training data, each batch given its own lab bias."""
    parts_X, parts_y = [X], [y]
    for _ in range(REPLICAS - 1):
        Xr = X.copy()
        for b in np.unique(batches):
            idx = batches == b
            Xr.loc[idx, cols] = (Xr.loc[idx, cols].to_numpy()
                                 * np.exp(rng.normal(0.0, AUG_SD, len(cols))))
        parts_X.append(Xr)
        parts_y.append(y)
    return pd.concat(parts_X, ignore_index=True), pd.concat(parts_y, ignore_index=True)


def cohorts():
    """(name, X_train, y_train, batch labels, X_test, y_test, lab columns, kind, bundle)."""
    out = []

    # The shipped bloodwork panel, tested on the decade-earlier survey.
    tr = pd.read_csv("data/nhanes_cancer_mortality.csv")
    te = pd.read_csv("data/nhanes3_cancer_mortality.csv")
    skip = {"cancer_death", "followup_months", "cycle", "race_ethnicity"}
    feats = [c for c in tr.columns if c in te.columns and c not in skip
             and pd.api.types.is_numeric_dtype(tr[c])]
    med = tr[feats].median()
    labs = [c for c in feats if c not in ("age", "gender")]
    out.append(("cancer_mortality", tr[feats].fillna(med), tr["cancer_death"].astype(int),
                tr["cycle"].to_numpy(), te[feats].fillna(med), te["cancer_death"].astype(int),
                labs, "logistic", "backend/models/model_cancer_mortality.joblib"))

    # The liver panel, tested on the 2021-2023 survey that nothing had seen.
    cfg = next(c for c in tm.DATASETS if c["name"] == "liver")
    full = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    train_df, _ = tm.split_temporal(full, "liver")
    fresh = pd.read_csv("data/nhanes_liver_2021.csv")
    Xl = tm.build_features(cfg, train_df)
    lmed = Xl.median()
    Xt = tm.build_features(cfg, fresh)
    lab_cols = [c for c in ("bilirubin", "alkaline_phosphatase", "ggt", "alt", "ast",
                            "protein_total", "albumin") if c in Xl.columns]
    out.append(("liver", Xl.fillna(lmed), cfg["target"](train_df).astype(int),
                train_df["cycle"].to_numpy(), Xt.fillna(lmed),
                cfg["target"](fresh).astype(int), lab_cols, "ensemble",
                "backend/models/model_liver.joblib"))
    return out


def main():
    results = {"levels": LEVELS, "draws": DRAWS, "aug_sd": AUG_SD, "cohorts": {}}
    verdicts = []
    for (name, Xtr, ytr, batches, Xte, yte, cols, kind, bundle_path) in cohorts():
        print(f"\n{name}: train {len(ytr):,}, test {len(yte):,} ({int(yte.sum())} events), "
              f"{len(cols)} analytes perturbed")
        yte_a = yte.to_numpy()
        entry = {"kind": kind, "n_test": int(len(yte)), "events": int(yte.sum())}

        shipped = joblib.load(bundle_path)
        entry["shipped"] = fragility(lambda X: score(shipped, X), Xte, yte_a, cols)

        # Per analyte: which single value moves the AUC most when it is off by 20%.
        rng = np.random.default_rng(1)
        base = float(roc_auc_score(yte_a, score(shipped, Xte)))
        per = {}
        for c in cols:
            drops = []
            for sign in (-0.2, 0.2):
                Xs = Xte.copy()
                Xs[c] = Xs[c] * (1 + sign)
                drops.append(base - float(roc_auc_score(yte_a, score(shipped, Xs))))
            per[c] = round(float(max(drops)), 4)
        entry["single_analyte_20pct_worst_drop"] = dict(sorted(per.items(),
                                                              key=lambda kv: -kv[1]))
        top = list(entry["single_analyte_20pct_worst_drop"].items())[:3]
        print(f"  shipped model, clean AUC {base:.4f}; most fragile single analytes at 20%: "
              + ", ".join(f"{k} {v:+.4f}" for k, v in top))
        for lvl, v in entry["shipped"].items():
            print(f"    bias {lvl:>4}: mean {v['mean']:.4f}  worst-decile {v['worst_decile']:.4f}")

        # Plain vs augmented, same kind and features, fitted here.
        rate = float(ytr.mean())
        plain = tm.model_factory(kind, len(ytr), rate).fit(Xtr, ytr)
        Xa, ya = augmented_training_set(Xtr, ytr, cols, batches, np.random.default_rng(7))
        aug = tm.model_factory(kind, len(ya), float(ya.mean())).fit(Xa, ya)
        fp = fragility(lambda X: plain.predict_proba(X)[:, 1], Xte, yte_a, cols, seed=2)
        fa = fragility(lambda X: aug.predict_proba(X)[:, 1], Xte, yte_a, cols, seed=2)
        entry["plain"], entry["augmented"] = fp, fa
        print(f"  refitted {kind}, plain vs trained across simulated labs:")
        for lvl in fp:
            print(f"    bias {lvl:>4}: plain {fp[lvl]['mean']:.4f} (worst-decile "
                  f"{fp[lvl]['worst_decile']:.4f})   augmented {fa[lvl]['mean']:.4f} "
                  f"(worst-decile {fa[lvl]['worst_decile']:.4f})")

        gain = fa["10%"]["worst_decile"] - fp["10%"]["worst_decile"]
        cost = fp["0%"]["mean"] - fa["0%"]["mean"]
        ok = bool(gain >= ADOPT_GAIN and cost <= ADOPT_COST)
        entry.update({"worst_decile_gain_at_10pct": round(float(gain), 4),
                      "clean_auc_cost": round(float(cost), 4), "meets_bar": ok})
        verdicts.append(ok)
        print(f"  worst-decile gain at 10% bias {gain:+.4f} (need >= {ADOPT_GAIN}); "
              f"clean AUC cost {cost:+.4f} (allowed <= {ADOPT_COST}) -> "
              f"{'meets the bar' if ok else 'does not meet the bar'}")
        results["cohorts"][name] = entry

    results["adopt_augmented_training"] = bool(all(verdicts))
    print(f"\n  VERDICT: augmented training is "
          + ("adopted." if all(verdicts) else
             "not adopted: it does not clear the bar on both cohorts."))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
