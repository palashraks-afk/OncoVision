"""
NEGATIVE RESULT, kept deliberately.

Question
--------
The breast panel contradicts this project's own schema rule. Its four inputs are
nuclear morphology from a fine needle aspirate, which means a biopsy has already
been taken, so the panel interprets a biopsy rather than screening for one. The
obvious fix is to rebuild it on blood markers instead, so it needs only a blood
draw like every other panel here.

Breast Cancer Coimbra (UCI 451) is exactly that dataset: 116 real women from
Portugal, blood markers only, no biopsy. Five of its features are ones a patient
could plausibly have measured and that NHANES also carries, which makes an
external test possible:

    age, bmi, glucose, insulin, HOMA

So: train on Coimbra, test on NHANES 2017-2018 women aged 20 and over who have
fasting glucose and insulin, using MCQ230 code 14 as the breast cancer label.

Answer
------
It does not work. The model reaches 0.80 on a held-out slice of Coimbra and
falls to chance on NHANES. Logistic regression does no better. The panel is
therefore NOT built, and the schema rule is amended instead to acknowledge that
this project contains two kinds of panel: ones that screen from routine blood,
and one that interprets a diagnostic test already performed.

This file exists so the negative result is reproducible rather than merely
asserted. Running the experiment before building the feature is what stopped a
non-working panel from shipping.

Caveat stated plainly: NHANES contributes 33 cases in the fasting subsample,
which is below the roughly 96 events needed to estimate a proportion tightly, so
the interval is wide. That makes this evidence of "not shown to work" rather
than proof of "cannot work". Either way it does not support shipping.

Run:  python experiments/blood_breast_panel.py
"""

import io
import json
import os
import ssl
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm
from evaluate import bootstrap_ci

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES = ["age", "bmi", "glucose", "insulin", "homa"]
NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"


def grab(fname):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(NHANES + fname, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=300, context=ctx).read()
    return pd.read_sas(io.BytesIO(raw), format="xport")


def nhanes_women():
    """
    NHANES women 20+ with fasting glucose and insulin.

    Units were checked rather than assumed: LBXGLU is mg/dL and LBXIN is uU/mL,
    which match the Coimbra columns directly, so no conversion is needed here.
    HOMA is derived the standard way, glucose times insulin over 405.
    """
    demo = grab("DEMO_J.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    bmx = grab("BMX_J.XPT")[["SEQN", "BMXBMI"]]
    glu = grab("GLU_J.XPT")[["SEQN", "LBXGLU"]]
    ins = grab("INS_J.XPT")[["SEQN", "LBXIN"]]
    mcq = grab("MCQ_J.XPT")

    kind = [c for c in mcq.columns if c.startswith("MCQ230")]
    mcq["breast_cancer"] = mcq[kind].isin([14.0]).any(axis=1).astype(int)

    d = (demo.merge(bmx, on="SEQN")
              .merge(glu, on="SEQN")
              .merge(ins, on="SEQN")
              .merge(mcq[["SEQN", "breast_cancer"]], on="SEQN"))
    d = d[(d["RIAGENDR"] == 2) & (d["RIDAGEYR"] >= 20)]
    d = d.dropna(subset=["BMXBMI", "LBXGLU", "LBXIN"])

    return pd.DataFrame({
        "age": d["RIDAGEYR"].astype(float),
        "bmi": d["BMXBMI"].astype(float),
        "glucose": d["LBXGLU"].astype(float),
        "insulin": d["LBXIN"].astype(float),
        "homa": (d["LBXGLU"] * d["LBXIN"] / 405.0).astype(float),
        "breast_cancer": d["breast_cancer"].astype(int),
    })


def fit(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(
        tm.build_ensemble(len(y), float(np.mean(y))), method="isotonic", cv=cv
    )
    return model.fit(X, y)


def main():
    co = pd.read_csv("data/breast_coimbra_portugal.csv")
    Xc, yc = co[FEATURES], co["breast_cancer"].astype(int)
    print(f"Coimbra, Portugal : {len(co)} women, {int(yc.sum())} with breast cancer "
          f"({yc.mean():.1%}), case-control")

    nh = nhanes_women()
    yn = nh["breast_cancer"]
    print(f"NHANES, USA       : {len(nh)} women 20+ with fasting labs, "
          f"{int(yn.sum())} with breast cancer ({yn.mean():.2%}), population based")

    Xa, Xb, ya, yb = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc
    )
    p_int = fit(Xa, ya).predict_proba(Xb)[:, 1]
    internal = roc_auc_score(yb, p_int)

    model = fit(Xc, yc)
    p_ext = model.predict_proba(nh[FEATURES])[:, 1]
    external = roc_auc_score(yn, p_ext)
    ext_ci = bootstrap_ci(np.asarray(yn), p_ext, roc_auc_score)

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(Xc, yc)
    p_lr = lr.predict_proba(nh[FEATURES])[:, 1]
    ext_lr = roc_auc_score(yn, p_lr)
    lr_ci = bootstrap_ci(np.asarray(yn), p_lr, roc_auc_score)

    print()
    print(f"  Coimbra internal held-out AUC   {internal:.3f}")
    print(f"  EXTERNAL on NHANES women        {external:.3f}   "
          f"(95% CI {ext_ci[0]} to {ext_ci[1]})")
    print(f"  logistic regression, external   {ext_lr:.3f}   "
          f"(95% CI {lr_ci[0]} to {lr_ci[1]})")
    print(f"  drop                            {internal - external:.3f}")

    chance_covered = ext_ci[0] <= 0.5 <= ext_ci[1]
    print()
    print("VERDICT: " + (
        "the external interval contains 0.5, so the panel cannot be shown to beat "
        "chance on an independent population. Not built."
        if chance_covered else
        "the external interval excludes chance, so this is worth building."
    ))

    out = {
        "question": "Can the breast panel be rebuilt on blood markers instead of biopsy morphology?",
        "answer": "No, not on the evidence available.",
        "features": FEATURES,
        "train_cohort": {"name": "Breast Cancer Coimbra, UCI 451", "n": int(len(co)),
                         "cases": int(yc.sum()), "prevalence": round(float(yc.mean()), 3)},
        "test_cohort": {"name": "NHANES 2017-2018 women 20+, fasting subsample", "n": int(len(nh)),
                        "cases": int(yn.sum()), "prevalence": round(float(yn.mean()), 4)},
        "internal_auc": round(float(internal), 3),
        "external_auc": round(float(external), 3),
        "external_auc_ci": ext_ci,
        "external_logistic_auc": round(float(ext_lr), 3),
        "external_logistic_ci": lr_ci,
        "auc_drop": round(float(internal - external), 3),
        "interval_contains_chance": bool(chance_covered),
        "caveat": ("NHANES contributes only 33 cases in the fasting subsample, below the "
                   "roughly 96 events needed for a tight estimate, so this shows the panel "
                   "is not demonstrated to work rather than proving it cannot."),
        "decision": ("Blood-based breast panel not built. The schema rule is amended instead "
                     "to acknowledge two kinds of panel: those that screen from routine blood, "
                     "and one that interprets a diagnostic test already performed."),
    }
    with open("experiments/blood_breast_panel_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/blood_breast_panel_result.json")


if __name__ == "__main__":
    main()
