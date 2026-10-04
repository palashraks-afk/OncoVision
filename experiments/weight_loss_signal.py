"""
Does unintended weight loss, which no lab contains, carry a cancer signal?

Why this one by itself
----------------------
Unexplained weight loss is among the best-established early warnings of cancer,
and it is the clearest case of the thing a lab report cannot hold: information
about the person. NHANES asks for current weight and weight one year ago. The
whole-picture models saw those as two separate columns, and a regularised model
has no particular reason to subtract them, so the signal could have been present
and invisible. This builds the change explicitly and tests it alone.

    percent change    (current - a year ago) / a year ago
    lost 5%+          a binary flag, the usual clinical threshold for "significant"

Tested on both outcomes, against age and sex, by repeated paired cross-validation:
    honest       cancer death within ten years against everyone else
    decedents    among those who died within ten years, cancer against another cause

Pre-registered, written before running: weight loss is a cancer signal only if it
beats age and sex with a 95% interval excluding zero on the decedent outcome AND
the honest one. A flag that predicts any death (weight loss goes with every serious
illness) would pass the honest outcome alone, which is why the decedent outcome has
to hold too.

Run:  python experiments/weight_loss_signal.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = "experiments/weight_loss_signal_result.json"
DATA = "data/nhanes_wide.csv.gz"
CANCER, HORIZON = 2, 120
REPEATS, FOLDS, N_BOOT = 5, 5, 2000


def oof(X, y):
    p = np.zeros(len(y))
    for r in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=r)
        for tr, te in cv.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            m.fit(X.iloc[tr], y.iloc[tr])
            p[te] += m.predict_proba(X.iloc[te])[:, 1]
    return p / REPEATS


def paired(y, a, b):
    rng = np.random.default_rng(0)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    g = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        g.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]


def main():
    df = pd.read_csv(DATA)
    now, ago = df["WHD020"], df["WHD050"]
    # Refused and don't-know codes for weight are 7777 and 9999, which are not
    # weights. Anything outside a human range is missing.
    ok = now.between(60, 600) & ago.between(60, 600)
    df["pct_change"] = np.where(ok, (now - ago) / ago * 100.0, np.nan)
    df["lost5"] = np.where(ok, (df["pct_change"] <= -5).astype(float), np.nan)
    # The clinical warning is UNINTENDED loss. WHQ070 asks whether they tried to lose
    # weight in the past year (1 yes, 2 no); anyone who was dieting is not a signal.
    tried = df["WHQ070"].where(df["WHQ070"].isin([1, 2]))
    df["unintended"] = np.where(ok & tried.notna(),
                                ((df["pct_change"] <= -5) & (tried == 2)).astype(float), np.nan)
    have = df["pct_change"].notna()
    print(f"{int(have.sum()):,} of {len(df):,} adults report both weights "
          f"({have.mean():.0%}); {int(df.loc[have, 'lost5'].sum()):,} lost 5% or more\n")

    months, died = df["followup_months"], df["died"] == 1
    cancer_in = died & (df["ucod_leading"] == CANCER) & (months <= HORIZON)
    died_in = died & (months <= HORIZON)
    pops = {"honest": (died_in | (months >= HORIZON)) & have, "decedents": died_in & have}

    result = {"reporting": int(have.sum()), "lost5": int(df.loc[have, "lost5"].sum()),
              "outcomes": {}}
    passes = []
    for name, mask in pops.items():
        d = df[mask.values].reset_index(drop=True)
        y = cancer_in[mask.values].astype(int).reset_index(drop=True)
        base = oof(d[["age", "gender"]], y)
        withw = oof(d[["age", "gender", "pct_change", "lost5", "unintended"]].fillna(0.0), y)
        a0, a1 = roc_auc_score(y, base), roc_auc_score(y, withw)
        ci = paired(y, withw, base)
        # The raw rate, because it is what a person would be told.
        lost = d["unintended"] == 1
        rate_lost, rate_ok = float(y[lost].mean()), float(y[~lost].mean())
        result["outcomes"][name] = {
            "n": int(len(y)), "events": int(y.sum()),
            "age_sex_auc": round(float(a0), 4), "with_weight_change_auc": round(float(a1), 4),
            "gain": round(float(a1 - a0), 4), "gain_ci": ci, "beats_age_sex": bool(ci[0] > 0),
            "cancer_share_if_unintended_loss": round(rate_lost, 3),
            "cancer_share_otherwise": round(rate_ok, 3), "n_lost": int(lost.sum())}
        passes.append(ci[0] > 0)
        print(f"{name.upper()}: {len(y):,} adults, {int(y.sum())} cancer deaths")
        print(f"   age+sex {a0:.4f}  +weight change {a1:.4f}  gain {a1 - a0:+.4f} {ci}  "
              f"-> {'beats age+sex' if ci[0] > 0 else 'not shown'}")
        print(f"   share who died of cancer: {rate_lost:.1%} among {int(lost.sum())} with UNINTENDED loss of 5%+, "
              f"{rate_ok:.1%} otherwise\n")

    result["is_a_cancer_signal"] = bool(all(passes))
    print("VERDICT: " + ("weight loss is a cancer signal here." if all(passes) else
          "weight loss does not clear both outcomes."))
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
