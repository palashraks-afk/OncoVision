"""
Breast panel accuracy by race and ethnicity, on every mammogram rather than a quarter of them.

The problem
-----------
bcsc_validation.py measures subgroups on BCSC's validation split: 597,859
mammograms. Two groups had too few cancers there to measure at all -- Native
American women (19) and other or mixed race (22) -- so for them the panel's
accuracy was unknown rather than acceptable.

The consortium's full file holds 2,392,998 mammograms, four times as many. The
validation split cannot simply be enlarged, because the shipped model was
trained on the rest. Cross-fitting can: the full table is split into five folds,
a model of the shipped kind is fitted on four and scores the fifth, and every
mammogram ends up scored by a model that never saw it. The subgroup AUCs are
then computed on all 2.39M, with roughly four times the cancers per group.

What it is and is not
---------------------
It measures the method -- the same features, the same model kind -- on data each
prediction did not train on. It does not re-measure the exact shipped model,
which saw the training split; the validation-split figures remain the ones for
that. A group is reported only once it has at least MIN_EVENTS cancers.

The file is a frequency table, so rows are covariate patterns with a count. A
pattern and all its mammograms sit in one fold, and every statistic is weighted
by count, with a Poisson bootstrap over mammograms for the intervals.

Run:  python experiments/bcsc_subgroups_full.py
"""

import io
import json
import os
import ssl
import sys
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_bcsc_breast as fb  # noqa: E402
import train_models as tm  # noqa: E402

warnings.filterwarnings("ignore")

OUT = "experiments/bcsc_subgroups_full_result.json"
NAME = "breast_screening"
MIN_EVENTS = 50
BOOT = 300
RACE = {1: "White", 2: "Asian or Pacific Islander", 3: "Black",
        4: "Native American", 5: "Other or mixed"}


def load_full():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    raw = urllib.request.urlopen(
        urllib.request.Request(fb.URL, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=600, context=ctx).read()
    df = pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw)).read(fb.MEMBER)),
                     sep=r"\s+", header=None, names=fb.COLUMNS)
    for col in fb.FEATURES:
        df[col] = df[col].replace(9, np.nan)
    return df


def wauc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))


def main():
    cfg = next(c for c in tm.DATASETS if c["name"] == NAME)
    df = load_full().reset_index(drop=True)
    y = df["cancer"].astype(int).to_numpy()
    w = df["count"].to_numpy(dtype=float)
    X = tm.build_features(cfg, df)
    X = X.fillna(X.median())

    # Shipped kind for this panel is logistic regression, and weighted logistic
    # regression on a frequency table is exactly the fit on the expanded rows.
    p = np.zeros(len(y))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced"))
        m.fit(X.iloc[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        print(f"  fold {k + 1} of 5 scored", flush=True)

    overall = wauc(y, p, w)
    print(f"\ncross-fitted over {int(w.sum()):,} mammograms, {int(w[y == 1].sum()):,} "
          f"cancers: AUC {overall:.3f}\n")

    rng = np.random.default_rng(0)
    groups = {}
    race = df["race"].to_numpy()
    for code, label in RACE.items():
        m = race == code
        ev = int(w[m & (y == 1)].sum())
        if ev < MIN_EVENTS or len(np.unique(y[m])) < 2:
            groups[label] = {"events": ev, "auc": None}
            print(f"  {label:<28} {ev:>6,} cancers  still too few to measure")
            continue
        a = wauc(y[m], p[m], w[m])
        boots = []
        for _ in range(BOOT):
            wb = rng.poisson(w[m]).astype(float)
            if wb[y[m] == 1].sum() == 0:
                continue
            boots.append(wauc(y[m], p[m], wb))
        ci = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
        worse = bool(a < overall - 0.05)
        groups[label] = {"events": ev, "auc": round(a, 3), "auc_ci": ci,
                         "materially_worse": worse}
        print(f"  {label:<28} {ev:>6,} cancers  AUC {a:.3f} (95% CI {ci[0]} to {ci[1]})"
              f"{'   <-- more than 0.05 below overall' if worse else ''}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"n_mammograms": int(w.sum()), "n_cancers": int(w[y == 1].sum()),
                   "overall_auc": round(overall, 3), "groups": groups,
                   "min_events": MIN_EVENTS,
                   "method": "5-fold cross-fitting over the full BCSC risk estimation table"},
                  f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
