"""
Does training across simulated laboratories hold for liver on its OTHER unseen cycle?

Why this exists
---------------
lab_shift_robustness.py tested augmented training on two cohorts. On liver, scored
on NHANES 2021-2023, it improved the worst case at 10 percent lab bias by +0.021
and also raised the clean AUC by +0.011. On the cancer-mortality cohort it did
nothing. The pre-set rule needed both, so it was not adopted. The mortality panel
has since been withdrawn, which leaves a rule that can no longer be met, and
deciding for liver alone after seeing its result would be moving the bar.

The honest way to settle it without moving anything is to test liver on a second
cohort it has never seen, with the same thresholds, written down before this run:
the 2017-2018 cycle that train_models withholds from liver training.

    adopt only if, at a 10 percent bias, the worst-decile AUC over 50 random
    laboratories improves by at least 0.010 over the plain model, and the AUC
    with no bias falls by no more than 0.005

A control is added because the augmented set is four copies of the training rows,
each with its own bias, and the clean-AUC gain could come from the extra copies
rather than the bias. So there is also a PLAIN model trained on four unbiased
copies. If that matches the augmented model's clean AUC, the gain belongs to the
duplication, and what augmentation adds is only the robustness.

Run:  python experiments/lab_shift_liver_confirm.py
"""

import importlib.util
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_models as tm

spec = importlib.util.spec_from_file_location(
    "lab_shift_robustness", os.path.join(ROOT, "experiments", "lab_shift_robustness.py"))
lsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lsr)

warnings.filterwarnings("ignore")
OUT = "experiments/lab_shift_liver_confirm_result.json"


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == "liver")
    full = pd.read_csv(os.path.join(tm.DATA_DIR, cfg["file"]))
    train_df, held_df = tm.split_temporal(full, "liver")
    Xtr = tm.build_features(cfg, train_df)
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    ytr = cfg["target"](train_df).astype(int)
    Xte = tm.build_features(cfg, held_df).fillna(med)
    yte = cfg["target"](held_df).astype(int).to_numpy()
    cols = [c for c in ("bilirubin", "alkaline_phosphatase", "ggt", "alt", "ast",
                        "protein_total", "albumin") if c in Xtr.columns]
    batches = train_df["cycle"].to_numpy()
    rate = float(ytr.mean())
    print(f"liver: train {len(ytr):,}, held-out 2017-2018 {len(yte):,} "
          f"({int(yte.sum())} events), {len(cols)} analytes perturbed\n")

    plain = tm.model_factory("ensemble", len(ytr), rate).fit(Xtr, ytr)

    Xa, ya = lsr.augmented_training_set(Xtr, ytr, cols, batches, np.random.default_rng(7))
    aug = tm.model_factory("ensemble", len(ya), float(ya.mean())).fit(Xa, ya)

    # The control: the same number of rows, none of them biased.
    X4 = pd.concat([Xtr] * lsr.REPLICAS, ignore_index=True)
    y4 = pd.concat([ytr] * lsr.REPLICAS, ignore_index=True)
    plain4 = tm.model_factory("ensemble", len(y4), float(y4.mean())).fit(X4, y4)

    arms = {"plain": plain, "plain x4 copies (control)": plain4, "augmented": aug}
    frag = {name: lsr.fragility(lambda X, m=m: m.predict_proba(X)[:, 1], Xte, yte, cols, seed=2)
            for name, m in arms.items()}

    print(f"  {'bias':>5}  " + "  ".join(f"{n:>26}" for n in arms))
    for lvl in frag["plain"]:
        print(f"  {lvl:>5}  " + "  ".join(
            f"{frag[n][lvl]['mean']:.4f} (worst {frag[n][lvl]['worst_decile']:.4f})".rjust(26)
            for n in arms))

    p, a, p4 = frag["plain"], frag["augmented"], frag["plain x4 copies (control)"]
    gain = a["10%"]["worst_decile"] - p["10%"]["worst_decile"]
    gain_vs_control = a["10%"]["worst_decile"] - p4["10%"]["worst_decile"]
    cost = p["0%"]["mean"] - a["0%"]["mean"]
    clean_vs_control = a["0%"]["mean"] - p4["0%"]["mean"]
    meets = bool(gain >= lsr.ADOPT_GAIN and cost <= lsr.ADOPT_COST)
    print(f"\n  worst-decile gain at 10% bias vs plain {gain:+.4f} (need >= {lsr.ADOPT_GAIN}); "
          f"vs the 4-copy control {gain_vs_control:+.4f}")
    print(f"  clean AUC cost {cost:+.4f} (allowed <= {lsr.ADOPT_COST}); clean AUC vs control "
          f"{clean_vs_control:+.4f}")
    print(f"  -> {'MEETS the bar on the second unseen cycle' if meets else 'does NOT meet the bar'}")

    out = {"cohort": "NHANES 2017-2018, withheld from liver training",
           "n_test": int(len(yte)), "events": int(yte.sum()),
           "arms": frag, "worst_decile_gain_at_10pct": round(float(gain), 4),
           "worst_decile_gain_vs_control": round(float(gain_vs_control), 4),
           "clean_auc_cost": round(float(cost), 4),
           "clean_auc_vs_control": round(float(clean_vs_control), 4),
           "meets_bar": meets}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
