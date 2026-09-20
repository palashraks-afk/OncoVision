"""
Cross-checking a report against the patterns of many cancer patients.

The idea being tested
---------------------
Instead of fitting a model, hold a library of what cancer patients' bloodwork
looked like and ask how closely a new report matches it. Three forms of it, all
of them things a person might actually mean by "cross-check against patterns":

    NEIGHBOURS    of the 50 people in the library whose bloodwork most resembles
                  yours, how many went on to have cancer
    CENTROID      how much closer your report sits to the average cancer profile
                  than to the average healthy one, in Mahalanobis distance, which
                  accounts for values moving together
    UNUSUALNESS   how odd your COMBINATION of values is against healthy people
                  alone, by Mahalanobis distance and by isolation forest. This is
                  the project's founding sentence tested directly: signal lives in
                  a combination being slightly unusual together, not in one value
                  crossing a limit. It never looks at a cancer patient at all.

Why it is worth running rather than reasoning about
---------------------------------------------------
These are genuinely different instruments. A classifier draws one boundary;
neighbour matching can follow a lumpy, many-shaped region; an unusualness score
can flag a person who resembles nobody. If the signal in routine bloodwork is
scattered across several patterns rather than one direction, matching should
beat regression, and the whole project would be built the wrong way round.

The cohort is the only honest one for this question: NHANES linked to the
National Death Index, where blood was drawn years before the outcome, with
NHANES III as an external test. Doing this on a case-control cohort would
guarantee a flattering answer, because there the "patterns" belong to people
already diagnosed and often treated.

Pre-registered, written before running
--------------------------------------
Every method is fitted on the training cohort only and scored on NHANES III.
A method REPLACES the shipped logistic panel only if its paired gain over that
panel has a 95% interval excluding zero. Adding it as an extra feature counts
as a win on the same terms. Anything else is reported and not adopted.

Run:  python experiments/pattern_matching.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

OUT = "experiments/pattern_matching_result.json"
TRAIN = "data/nhanes_cancer_mortality.csv"
EXTERNAL = "data/nhanes3_cancer_mortality.csv"
TARGET = "cancer_death"
K = 50
N_BOOT = 2000


def paired_ci(y, a, b, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5 or y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return (round(float(np.mean(out)), 4),
            [round(float(np.percentile(out, 2.5)), 4),
             round(float(np.percentile(out, 97.5)), 4)])


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(EXTERNAL)
    skip = {TARGET, "followup_months", "cycle", "race_ethnicity"}
    feats = [c for c in tr.columns if c in te.columns and c not in skip
             and pd.api.types.is_numeric_dtype(tr[c])]
    labs = [f for f in feats if f not in ("age", "gender")]

    y_tr = tr[TARGET].astype(int).to_numpy()
    y_te = te[TARGET].astype(int).to_numpy()
    med = tr[feats].median()
    X_tr = tr[feats].fillna(med)
    X_te = te[feats].fillna(med)

    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    lab_idx = [feats.index(c) for c in labs]
    L_tr, L_te = Z_tr[:, lab_idx], Z_te[:, lab_idx]

    print(f"Train {len(tr):,} adults, {int(y_tr.sum())} cancer deaths | "
          f"Test {len(te):,}, {int(y_te.sum())}")
    print(f"{len(feats)} values, of which {len(labs)} are lab values\n")

    scores = {}

    # The panel that ships: one boundary, fitted the ordinary way.
    shipped = tm.model_factory("logistic", len(y_tr), float(y_tr.mean()))
    shipped.fit(X_tr, y_tr)
    scores["shipped panel"] = shipped.predict_proba(X_te)[:, 1]

    # NEIGHBOURS: of the K most similar reports in the library, how many were
    # people who went on to die of cancer. Similarity is on lab values only, so
    # it cannot quietly rediscover age.
    nn = NearestNeighbors(n_neighbors=K).fit(L_tr)
    _, idx = nn.kneighbors(L_te)
    scores[f"neighbours, {K} nearest"] = y_tr[idx].mean(axis=1)

    # The same two matchers given EVERYTHING the shipped panel gets, age and sex
    # included. Matching on lab values alone is the more faithful reading of the
    # idea, but it hands the panel a two-value head start, and a comparison that
    # unfair would prove nothing either way.
    nn_all = NearestNeighbors(n_neighbors=K).fit(Z_tr)
    _, idx_all = nn_all.kneighbors(Z_te)
    scores[f"neighbours with age and sex, {K}"] = y_tr[idx_all].mean(axis=1)

    cov_all = EmpiricalCovariance().fit(Z_tr)
    prec_all = cov_all.precision_
    ca_case, ca_well = Z_tr[y_tr == 1].mean(axis=0), Z_tr[y_tr == 0].mean(axis=0)

    def maha_all(Z, centre):
        d = Z - centre
        return np.einsum("ij,jk,ik->i", d, prec_all, d)

    scores["centroid match with age and sex"] = maha_all(Z_te, ca_well) - maha_all(Z_te, ca_case)

    # CENTROID: closer to the cancer profile than to the healthy one, measured
    # with the covariance so that values which move together are not counted
    # twice.
    cov = EmpiricalCovariance().fit(L_tr)
    prec = cov.precision_
    c_case = L_tr[y_tr == 1].mean(axis=0)
    c_well = L_tr[y_tr == 0].mean(axis=0)

    def maha(Z, centre):
        d = Z - centre
        return np.einsum("ij,jk,ik->i", d, prec, d)

    scores["centroid match"] = maha(L_te, c_well) - maha(L_te, c_case)

    # UNUSUALNESS: fitted on healthy people only, never shown a cancer patient.
    well = L_tr[y_tr == 0]
    cov_well = EmpiricalCovariance().fit(well)
    prec_well = cov_well.precision_
    centre_well = well.mean(axis=0)
    d = L_te - centre_well
    scores["unusualness, distance"] = np.einsum("ij,jk,ik->i", d, prec_well, d)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(well)
    scores["unusualness, isolation forest"] = -iso.score_samples(L_te)

    # Each pattern score offered to the shipped panel as one extra input, which
    # is the fair way to ask whether it knows anything the panel does not.
    d_tr = L_tr - centre_well
    extras_tr = {
        "unusualness, distance": np.einsum("ij,jk,ik->i", d_tr, prec_well, d_tr),
        "unusualness, isolation forest": -iso.score_samples(L_tr),
        "centroid match": maha(L_tr, c_well) - maha(L_tr, c_case),
    }
    # The training-side value of each pattern score has to exclude the person
    # themselves, or every row would be matched against its own answer and the
    # model would learn to trust a number it can never have at prediction time.
    _, idx_tr = nn.kneighbors(L_tr, n_neighbors=K + 1)
    extras_tr[f"neighbours, {K} nearest"] = y_tr[idx_tr[:, 1:]].mean(axis=1)
    _, idx_tr_all = nn_all.kneighbors(Z_tr, n_neighbors=K + 1)
    extras_tr[f"neighbours with age and sex, {K}"] = y_tr[idx_tr_all[:, 1:]].mean(axis=1)
    extras_tr["centroid match with age and sex"] = (maha_all(Z_tr, ca_well)
                                                    - maha_all(Z_tr, ca_case))

    added = {}
    for name, col_tr in extras_tr.items():
        Xa_tr = X_tr.copy()
        Xa_tr["pattern"] = col_tr
        Xa_te = X_te.copy()
        Xa_te["pattern"] = scores[name]
        m = tm.model_factory("logistic", len(y_tr), float(y_tr.mean()))
        m.fit(Xa_tr, y_tr)
        added[name] = m.predict_proba(Xa_te)[:, 1]

    base = scores["shipped panel"]
    base_auc = float(roc_auc_score(y_te, base))
    result = {"n_train": int(len(tr)), "n_test": int(len(te)),
              "events_test": int(y_te.sum()), "features": feats,
              "shipped_auc": round(base_auc, 4), "methods": {}}

    print(f"  {'method':<34} {'AUC alone':>9}   {'vs shipped':>10}   {'AUC as extra input':>18}")
    print(f"  {'shipped panel':<34} {base_auc:>9.4f}   {'--':>10}")
    for name, s in scores.items():
        if name == "shipped panel":
            continue
        auc = float(roc_auc_score(y_te, s))
        gain, ci = paired_ci(y_te, s, base)
        a_auc = float(roc_auc_score(y_te, added[name]))
        a_gain, a_ci = paired_ci(y_te, added[name], base, seed=1)
        beats = bool(ci[0] > 0)
        adds = bool(a_ci[0] > 0)
        result["methods"][name] = {
            "auc_alone": round(auc, 4), "gain_alone": gain, "gain_ci": ci,
            "replaces_shipped": beats,
            "auc_as_extra_input": round(a_auc, 4), "gain_as_extra": a_gain,
            "gain_as_extra_ci": a_ci, "adds_to_shipped": adds,
            "auc_ci_alone": bootstrap_ci(y_te, s, roc_auc_score)}
        print(f"  {name:<34} {auc:>9.4f}   {gain:>+10.4f}   {a_auc:>18.4f}"
              f"   {'REPLACES' if beats else ''}{'  ADDS' if adds else ''}")

    # Clearing zero is not the same as being worth shipping. This project already
    # refuses a more complicated model that leads by less than SIMPLER_MODEL_MARGIN,
    # which is how the panels choose between logistic regression and the ensemble,
    # and the same margin applies here rather than a new one invented afterwards.
    margin = tm.SIMPLER_MODEL_MARGIN
    result["materiality_margin"] = margin
    winners = []
    for n, v in result["methods"].items():
        significant = v["replaces_shipped"] or v["adds_to_shipped"]
        best = max(v["gain_alone"], v["gain_as_extra"])
        v["material"] = bool(best >= margin)
        v["adopted"] = bool(significant and v["material"])
        if v["adopted"]:
            winners.append(n)
        elif significant:
            print(f"\n  {n}: beats zero ({best:+.4f}) but not the {margin} margin this "
                  f"project requires before taking on extra machinery, so it is measured "
                  f"and not adopted.")
    result["adopt"] = winners
    print("\n  " + ("ADOPT: " + ", ".join(winners) if winners else
                    "Nothing is adopted. Matching a report against the patterns of cancer "
                    "patients reads the same signal the model already reads, and reads it "
                    "less well: the closest any version comes on its own is "
                    f"{max(v['auc_alone'] for v in result['methods'].values()):.3f} against "
                    f"{base_auc:.3f}."))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
