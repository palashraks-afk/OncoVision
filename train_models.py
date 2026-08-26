"""
Oncovision model training pipeline.

Trains one soft-voting ensemble (XGBoost + Extra Trees) per cancer domain and
writes it to /models and /backend/models as a joblib bundle.

Design rule: a model may only be trained on features the application can
actually collect from a patient. Training on columns the app never supplies
would produce an AUC that does not describe how the model performs in use, so
every dataset column is either mapped onto the app's canonical input schema or
dropped.

That rule has two classes of panel under it, and pretending otherwise was a real
inconsistency in this project:

  SCREENING panels read values from routine bloodwork that anyone can obtain.
  General and liver are these.

  INTERPRETATION panels read a diagnostic test that has already been performed.
  Breast is this one, because nuclear morphology comes from a fine needle
  aspirate. Rebuilding it on blood markers was attempted and failed at chance on
  an independent cohort, which is recorded in experiments/blood_breast_panel.py.

The distinction is now stated on the panel rather than glossed, because a
screening claim and an interpretation claim are not the same claim. Each bundle therefore carries its own feature list, the training
median for every feature (used to impute values the patient did not provide)
and its measured cross-validated performance.

Run:  python train_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = "data"
MODEL_DIRS = ["models", os.path.join("backend", "models")]
RANDOM_STATE = 42

# Panels that are trained and measured but deliberately not shipped.
# evaluate.py is the evidence for each decision.
WITHDRAWN = {
    "cervical": (
        "Withdrawn after the published 0.725 turned out to be a lucky split. "
        "Repeating the identical 80/20 protocol 30 times with different seeds gives a mean "
        "AUC of 0.594 with a standard deviation of 0.096 and a spread from 0.421 to 0.789. "
        "The 0.725 that was being published sits at the 97th percentile of that "
        "distribution, so it described one favourable shuffle rather than the model. Every "
        "other panel's shipped split lands between the 20th and 73rd percentile of its own "
        "distribution and is representative; this one was not. "
        "At a mean of 0.594 the spread includes chance, which is the same reason prostate "
        "was withdrawn, and the standard has to apply to both. "
        "The cause is 55 positive biopsies against the roughly 96 events needed, so a single "
        "test split carries about 11 cases and cannot estimate anything stably. "
        "An external cohort was built rather than assumed missing: 11,100 women from NHANES "
        "2005 to 2014 with the same risk history and 184 cervical cancers, in "
        "fetch_nhanes_cervical.py. It remains available if the panel is ever rebuilt on a "
        "cohort with enough events. See experiments/split_stability.py. "
        "The training config was removed alongside the 14 sexual-history fields it needed, "
        "because the schema guard correctly refuses features the app no longer collects. "
        "fetch_cervical.py, fetch_nhanes_cervical.py and experiments/cervical_panel.py are "
        "kept, so restoring the panel means re-adding those fields and the DATASETS entry."
    ),
}

# Cohort design, stated on every panel because it bounds what the numbers mean.
COHORT_DESIGN = {
    "general": "23,923 US adults from NHANES 2005 to 2014, nationally representative. The "
               "target is a cancer diagnosis within four years of the blood draw, with "
               "long-ago survivors excluded, so this is a screening question rather than a "
               "lifetime one. It adds 0.019 over knowing age and sex alone, which is small "
               "and is stated rather than hidden. Adding routine bloodwork was measured and "
               "made it worse, so this panel reads risk factors, not the lab report.",
    "breast": "Case-control and post-biopsy. This panel reads an aspirate that has already "
              "been taken, so it interprets a diagnostic test rather than screening for one. "
              "Rebuilding it on blood markers was tried and failed: see "
              "experiments/blood_breast_panel.py, external AUC 0.495 with a 95% CI of 0.377 "
              "to 0.607, which contains chance.",
    "liver": "35,511 US adults from NHANES 2005 to 2018, chemistry plus diabetes and "
             "hepatitis serology. Externally validated against 583 patients in India and "
             "589 in Germany, and temporally against later cycles. Detects liver disease, "
             "not liver cancer.",
    "pancreatic": "Case-control across three independent tissue banks. Cases are confirmed "
                  "adenocarcinoma, controls include benign hepatobiliary disease. Validated by "
                  "leave-one-site-out: mean AUC 0.962 with every site's interval excluding "
                  "chance, and a drop of only 0.007 from the internal random split, so the "
                  "panel transfers between institutions. It has still never met a screening "
                  "population, where it would flag roughly 525 people per true case.",
    "colorectal": "23,794 US adults from NHANES 2005 to 2014, nationally representative. Age, "
                  "sex and a complete blood count, which is the ColonFlag feature set, built "
                  "from open data. 96 cases diagnosed within eight years of the draw, the "
                  "smallest event count of any shipped panel, so its interval is correspondingly "
                  "wide. Bloodwork adds 0.030 over age and sex. Lung and prostate were tested "
                  "the same way and failed: lung on events, prostate because bloodwork adds "
                  "nothing without PSA.",
    "lung": "21,916 US adults with tobacco exposure from NHANES 1999 to 2018, 104 with lung "
            "cancer. Controls are restricted to smokers on purpose: on the unrestricted cohort "
            "the smoking question alone reaches 0.836 and any lab model inherits the credit. "
            "On the restricted comparison the baseline is 0.792 and the lab report adds 0.029, "
            "winning every repeat. Serum cotinine alone adds 0.016 while self-reported "
            "pack-years adds 0.001, so the measured value beats the questionnaire it replaces. "
            "The target is a lifetime diagnosis, and lung cancer survival is poor, so the cases "
            "are survivor-biased. Spirometry helps far more but only 13 cases have it.",
    "prostate": "212 men with suspected prostate cancer, all biopsied transperineally at one "
                "centre in 2022 to 2023, 121 with adenocarcinoma. Controls are men whose biopsy "
                "came back benign, mostly BPH. This is an INTERPRETATION panel: it needs a "
                "PI-RADS score from an MRI, because blood and ultrasound alone tie PSA rather "
                "than beat it (0.668 against 0.670), while adding PI-RADS reaches 0.826 and "
                "wins every repeat. Single centre, no external cohort. Precision is projected "
                "onto the roughly 40 percent malignancy rate among men taken to biopsy, not "
                "onto SEER, because this runs after referral.",
    "ovarian": "349 women operated on at one Chinese hospital between 2011 and 2018. The "
               "controls are women whose ovarian mass proved benign, not healthy volunteers, "
               "so this separates malignant from benign rather than sick from well. It is a "
               "triage panel for a mass that has already been found, which is why precision "
               "is projected onto the roughly 20 percent malignancy rate among women taken "
               "to surgery rather than onto SEER incidence. Single centre and single country, "
               "so it has no external cohort yet and that is its main open limitation.",
    "cervical": "858 women assessed for colposcopy at one hospital in Caracas, 55 with a "
                "positive biopsy. Prior-diagnosis columns are dropped as leakage. 55 events "
                "is below the roughly 96 needed for a tight estimate, so the interval runs "
                "0.539 to 0.888: it excludes chance and beats age alone, 0.725 against 0.458, "
                "but it is the least certain panel here. Precision is projected onto the 6.4 "
                "percent prevalence among referred women, because at population incidence it "
                "would flag about 7,383 women per true case and would not be shippable.",
    "prostate": "Case-control and post-prostatectomy. Gleason grade comes from the "
                "surgical specimen, not from screening.",
}

# ---------------------------------------------------------------------------
# Canonical application input schema.
# Every model feature must be one of these keys.
# ---------------------------------------------------------------------------
# Lab values, entered directly or parsed out of a PDF report.
LAB_FIELDS = [
    "age", "bmi", "wbc", "rbc", "hemoglobin", "platelets", "glucose", "calcium",
    "bun", "creatinine", "protein_total", "albumin", "ast", "alt", "bilirubin",
    "alkaline_phosphatase", "alpha_fetoprotein_level", "psa", "plasma_ca19_9",
    # Red cell and platelet indices, and GGT. Printed on every CBC and liver
    # panel already, and the ovarian panel reads them.
    "hematocrit", "mcv", "mch", "rdw", "mpv", "neutrophil_pct", "ggt",
    # Tumour markers ordered when a pelvic mass is being worked up.
    "ca125", "he4", "cea",
    # Prostate work-up. Volume comes off the ultrasound report and PSA density
    # is PSA divided by it; PI-RADS is the radiologist's score from an MRI.
    "prostate_volume", "psa_density", "pi_rads",
    # Serum cotinine is the nicotine metabolite and the objective measure of
    # tobacco exposure. It is a lab value, which is the point: it beats the
    # smoking question it replaces. CRP captures chronic inflammation.
    "cotinine", "crp",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
]

# Patient history, answered by the user rather than read off a lab report.
# gender             0 female, 1 male
# smoking            0 never, 1 former, 2 current
# alcohol_intake     0 none through 5 heavy
# physical_activity  hours of exercise per week, 0 to 10
# genetic_risk       0 low, 1 medium, 2 high
# remaining flags    0 no, 1 yes
# Only fields a shipped panel actually consumes. Inherited risk, prior cancer
# diagnosis, family history and cirrhosis were dropped: no panel uses them now
# that general and liver train on NHANES, and asking a patient for information
# nothing reads is just friction.
HISTORY_FIELDS = [
    "gender", "smoking", "alcohol_intake", "physical_activity",
    "hepatitis_b", "hepatitis_c", "diabetes",
    # Menopausal status, read by the ovarian panel. The rest of the
    # reproductive and sexual history went with the cervical panel when it was
    # withdrawn: no shipped model reads it, and asking a patient for fourteen
    # intimate questions that nothing consumes is pure friction.
    "menopause",
    # Tobacco dose. Kept because the lung panel is offered to people with
    # tobacco exposure and dose is the thing a clinician would ask for, even
    # though the measurement below beats it: pack-years added 0.001 over the
    # smoking question while serum cotinine added 0.016.
    "smoking_packyears",
]

APP_FIELDS = LAB_FIELDS + HISTORY_FIELDS


# ---------------------------------------------------------------------------
# Per dataset column translation into the canonical schema
# ---------------------------------------------------------------------------
def _map(series, table, default=np.nan):
    return series.astype(str).str.strip().map(table).fillna(default)


DATASETS = [
    {
        "name": "general",
        # A screening target, not a lifetime one.
        #
        # The previous version predicted "have you EVER been told you had
        # cancer". A person cured thirty years ago counted as positive, so the
        # model was largely predicting age: it reached 0.781 while age and sex
        # alone reached 0.777, a gain of 0.004.
        #
        # NHANES 2005 to 2014 records age at diagnosis, so the cohort can be cut
        # properly. Positives are people diagnosed within four years of the exam,
        # meaning the disease was present or recent when the blood was drawn.
        # Long-ago survivors are EXCLUDED rather than relabelled, because their
        # bloodwork reflects treatment and elapsed time rather than detection.
        # 23,923 adults, 750 recent diagnoses, 3.14 percent.
        #
        # On that target the gain over age and sex rises to 0.019. Small, and
        # real, and honestly reported on the panel.
        #
        # Bloodwork is deliberately NOT in this feature set, and that was
        # measured rather than assumed. Adding all 14 routine blood values made
        # the panel worse, 0.737 against 0.748, and bloodwork alone reached only
        # 0.663, below age and sex. Routine chemistry does not detect cancer in
        # a general population, which is precisely why the specific panels here
        # use disease specific markers and why commercial multi-cancer tests use
        # cell-free DNA. See experiments/screening_vs_age.py.
        "file": "nhanes_screening_general.csv",
        "label": "General Cancer Risk",
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: d["gender"],
            "bmi": lambda d: d["bmi"],
            "smoking": lambda d: d["smoking"],
            "alcohol_intake": lambda d: d["alcohol_intake"],
        },
        "target": lambda d: d["recent_cancer"].astype(int),
        "positive_means": ("a cancer diagnosis within four years of the blood draw, so recent "
                           "or current disease rather than a lifetime history"),
    },
    {
        "name": "breast",
        "file": "data.csv",
        "label": "Breast Malignancy, from biopsy imaging",
        # Only the four nuclear morphology means the app collects. The other 26
        # columns in the Wisconsin set are deliberately left out.
        "features": {
            "radius_mean": lambda d: d["radius_mean"],
            "texture_mean": lambda d: d["texture_mean"],
            "perimeter_mean": lambda d: d["perimeter_mean"],
            "area_mean": lambda d: d["area_mean"],
        },
        "target": lambda d: (d["diagnosis"].astype(str).str.upper() == "M").astype(int),
        "positive_means": "a malignant fine needle aspirate",
    },
    {
        "name": "liver",
        # 35,511 NHANES adults across seven cycles, 1,436 with a liver
        # condition. Chemistry plus risk history, so the panel reads the lab
        # report and the patient together rather than chemistry alone.
        #
        # Hepatitis here is serology rather than self-report, which is a better
        # measurement of the same question the app asks a patient to answer.
        #
        # India (UCI 225) and Germany (UCI 571) are no longer training data.
        # They are kept as independent external cohorts, which is worth more:
        # see external_validation.py.
        "file": "nhanes_liver_multicycle.csv",
        "label": "Liver Disease Risk",
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: d["gender"],
            "bilirubin": lambda d: d["bilirubin"],
            "alkaline_phosphatase": lambda d: d["alkaline_phosphatase"],
            "alt": lambda d: d["alt"],
            "ast": lambda d: d["ast"],
            "protein_total": lambda d: d["protein_total"],
            "albumin": lambda d: d["albumin"],
            "diabetes": lambda d: d["diabetes"],
            "hepatitis_b": lambda d: d["hepatitis_b"],
            "hepatitis_c": lambda d: d["hepatitis_c"],
        },
        "target": lambda d: d["liver_disease"].astype(int),
        "positive_means": "a clinical diagnosis of liver disease, not liver cancer",
    },
    {
        "name": "pancreatic",
        "file": "pancreatic_600_risk123.csv",
        "label": "Pancreatic Cancer Risk",
        # diagnosis is three class: 1 control, 2 benign hepatobiliary disease,
        # 3 pancreatic ductal adenocarcinoma. Only class 3 is cancer.
        # The urinary panel (LYVE1, REG1B, TFF1, REG1A), CEA and the urine
        # chemistry are dropped because the app does not collect them, and
        # 'stage' is dropped because it is only known after diagnosis.
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: _map(d["sex"], {"M": 1, "F": 0}),
            "creatinine": lambda d: d["creatinine"],
            "plasma_ca19_9": lambda d: d["CA19_9"],
            "bilirubin": lambda d: d["bilirubin"],
            "glucose": lambda d: d["glucose"],
        },
        "target": lambda d: (d["diagnosis"].astype(int) == 3).astype(int),
        "positive_means": "pancreatic ductal adenocarcinoma, separated from both healthy controls and benign hepatobiliary disease",
    },
    {
        "name": "colorectal",
        # 23,794 NHANES adults, 96 diagnosed with colon or rectal cancer within
        # eight years of the blood draw.
        #
        # This is the ColonFlag idea built from open data: age, sex and a
        # complete blood count. ColonFlag is the best known colorectal
        # early-detection model, is exactly that feature set, reports AUCs in
        # the low 0.80s, and is proprietary with no public data behind it. This
        # reaches 0.804 against 0.774 for age and sex alone.
        #
        # The eight year window was swept, not chosen. Four years is the cleaner
        # target but leaves 60 events, below the roughly 96 this project treats
        # as a floor. Eight is the tightest window that clears it, at exactly
        # 96, which makes this the smallest shipped panel by event count and
        # the interval says so.
        #
        # Anyone diagnosed longer ago than the window is excluded rather than
        # counted as a control, because labelling a survivor healthy is worse
        # than the problem it would solve.
        #
        # Lung and prostate were tested the same way and are not shipped. Lung
        # has 57 events at any window. Prostate has 373 and bloodwork adds
        # +0.002, because prostate needs PSA and NHANES excludes diagnosed men
        # from PSA testing. See experiments/site_window_sweep.py.
        "file": "nhanes_colorectal.csv",
        "label": "Bowel Cancer Risk",
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: d["gender"],
            "wbc": lambda d: d["wbc"],
            "rbc": lambda d: d["rbc"],
            "hemoglobin": lambda d: d["hemoglobin"],
            "platelets": lambda d: d["platelets"],
            "glucose": lambda d: d["glucose"],
            "calcium": lambda d: d["calcium"],
            "bun": lambda d: d["bun"],
            "creatinine": lambda d: d["creatinine"],
            "protein_total": lambda d: d["protein_total"],
            "albumin": lambda d: d["albumin"],
            "ast": lambda d: d["ast"],
            "alt": lambda d: d["alt"],
            "bilirubin": lambda d: d["bilirubin"],
            "alkaline_phosphatase": lambda d: d["alkaline_phosphatase"],
        },
        "target": lambda d: d["colorectal_cancer"].astype(int),
        "positive_means": ("colon or rectal cancer diagnosed within eight years of the "
                           "blood draw, with longer-ago survivors excluded rather than "
                           "counted as healthy"),
    },
    {
        "name": "ovarian",
        # 349 women operated on at the Third Affiliated Hospital of Soochow
        # University, 171 with ovarian cancer and 178 with a benign ovarian
        # tumour, every label taken from the resected specimen.
        #
        # The controls are the point. They are not healthy volunteers, they are
        # women whose ovarian mass turned out to be benign, so the panel has to
        # separate malignant from benign rather than sick from well. That is
        # the decision a clinician actually faces once a mass has been found,
        # and it is the same decision the ROMA and RMI indices are used for.
        #
        # Age had to be beaten before this could ship: cases run seventeen years
        # older than controls, 53 against 36, which is exactly the trap the
        # general panel fell into. Age and menopausal status alone reach 0.792.
        # Routine bloodwork with no age and no tumour markers reaches 0.938, and
        # the full set reaches 0.941. So the lab report is doing the work here,
        # which is not true of the general panel and is worth stating.
        #
        # Source values are SI and are converted to conventional units in
        # fetch_ovarian.py. Mixing the two is what broke the German liver
        # transfer, so it is done once, explicitly, at load time.
        "file": "ovarian_soochow.csv",
        "label": "Ovarian Malignancy, in a known ovarian mass",
        "features": {
            "age": lambda d: d["age"],
            "menopause": lambda d: d["menopause"],
            "albumin": lambda d: d["albumin"],
            "protein_total": lambda d: d["protein_total"],
            "glucose": lambda d: d["glucose"],
            "calcium": lambda d: d["calcium"],
            "creatinine": lambda d: d["creatinine"],
            "bun": lambda d: d["bun"],
            "bilirubin": lambda d: d["bilirubin"],
            "alt": lambda d: d["alt"],
            "ast": lambda d: d["ast"],
            "alkaline_phosphatase": lambda d: d["alkaline_phosphatase"],
            "ggt": lambda d: d["ggt"],
            "hemoglobin": lambda d: d["hemoglobin"],
            "rbc": lambda d: d["rbc"],
            "platelets": lambda d: d["platelets"],
            "hematocrit": lambda d: d["hematocrit"],
            "mcv": lambda d: d["mcv"],
            "mch": lambda d: d["mch"],
            "rdw": lambda d: d["rdw"],
            "mpv": lambda d: d["mpv"],
            "neutrophil_pct": lambda d: d["neutrophil_pct"],
            "ca125": lambda d: d["ca125"],
            "he4": lambda d: d["he4"],
            "cea": lambda d: d["cea"],
            "alpha_fetoprotein_level": lambda d: d["alpha_fetoprotein_level"],
            "plasma_ca19_9": lambda d: d["plasma_ca19_9"],
        },
        "target": lambda d: d["ovarian_cancer"].astype(int),
        "positive_means": ("ovarian cancer on histology, separated from a benign ovarian "
                           "tumour rather than from a healthy woman"),
    },
    {
        "name": "lung",
        # 21,916 NHANES adults WITH TOBACCO EXPOSURE, 104 of them with lung
        # cancer, pooled across all ten cycles 1999 to 2018.
        #
        # The restriction is the whole design. On the full 45,396-adult cohort
        # the questionnaire alone scores 0.836, because almost every case smoked
        # and most controls did not, so "do you smoke" separates the groups by
        # itself and any lab model inherits the credit. That is not a lung
        # panel, it is a smoking detector.
        #
        # Restricting controls to people with tobacco exposure, by self-report
        # or serum cotinine at or above 3 ng/mL, drops the baseline to 0.792 and
        # asks the real question: among people who smoke, whose blood looks like
        # cancer. It is also the population that actually gets offered lung
        # screening.
        #
        # On that harder comparison, measured over 10 paired repeats:
        #
        #     questionnaire alone            0.792
        #     plus pack-years                0.793   +0.001, wins 5/10
        #     plus serum cotinine            0.808   +0.016, wins 10/10
        #     plus the whole lab report      0.821   +0.029, wins 10/10
        #
        # The middle two lines are the finding worth keeping: a measured lab
        # value beat the self-reported dose it replaces. That is the claim this
        # whole application makes, tested and upheld.
        #
        # Spirometry adds far more, 0.807 to 0.914, but only 13 lung cases have
        # FEV1 and FVC recorded, so it is nowhere near the event floor and is
        # left out. See experiments/lung_panel.py.
        #
        # Limitation carried onto the panel: pooling all ten cycles means giving
        # up age at diagnosis, so the target is a LIFETIME diagnosis. Lung cancer
        # five year survival is about a quarter, so the cases who survive to be
        # interviewed are a survivor-biased minority.
        "file": "nhanes_lung_smokers.csv",
        "label": "Lung Cancer Risk, with tobacco exposure",
        "features": {
            "age": lambda d: d["age"],
            "gender": lambda d: d["gender"],
            "smoking": lambda d: d["smoking"],
            "smoking_packyears": lambda d: d["smoking_packyears"],
            "cotinine": lambda d: d["cotinine"],
            "crp": lambda d: d["crp"],
            "wbc": lambda d: d["wbc"],
            "rbc": lambda d: d["rbc"],
            "hemoglobin": lambda d: d["hemoglobin"],
            "platelets": lambda d: d["platelets"],
            "hematocrit": lambda d: d["hematocrit"],
            "mcv": lambda d: d["mcv"],
            "rdw": lambda d: d["rdw"],
            "mpv": lambda d: d["mpv"],
            "glucose": lambda d: d["glucose"],
            "calcium": lambda d: d["calcium"],
            "bun": lambda d: d["bun"],
            "creatinine": lambda d: d["creatinine"],
            "protein_total": lambda d: d["protein_total"],
            "albumin": lambda d: d["albumin"],
            "ast": lambda d: d["ast"],
            "alt": lambda d: d["alt"],
            "bilirubin": lambda d: d["bilirubin"],
            "alkaline_phosphatase": lambda d: d["alkaline_phosphatase"],
        },
        "target": lambda d: d["lung_cancer"].astype(int),
        "positive_means": ("a lung cancer diagnosis, among people who smoke or have "
                           "measurable tobacco exposure"),
    },
    {
        "name": "prostate",
        # 212 men with suspected prostate cancer, all of whom went to
        # transperineal biopsy at one centre between May 2022 and November 2023.
        # 121 came back adenocarcinoma. Zenodo 10.5281/zenodo.17623285, CC-BY 4.0.
        #
        # This replaces the 97-record Stanford cohort that was withdrawn. The
        # controls here are men whose biopsy came back benign, 67 of them BPH,
        # which is the decision a urologist actually faces once a man has been
        # referred. Separating a biopsied man from the general population is not
        # a decision at all.
        #
        # THIS IS AN INTERPRETATION PANEL, not a screening one, and the
        # distinction is load bearing. Measured against PSA alone, which is the
        # only baseline that counts because every one of these men already had a
        # PSA drawn:
        #
        #     age alone                            0.568
        #     PSA alone                            0.670
        #     age, PSA, volume, PSA density, BMI   0.668   -0.003, wins 10/20
        #     the above plus PI-RADS               0.826   +0.156, wins 20/20
        #
        # A panel built only from blood and ultrasound does NOT beat reading the
        # PSA number. It ties it. What carries this panel is PI-RADS, the
        # radiologist's score off an MRI, exactly as nuclear morphology carries
        # the breast panel. So it ships requiring an MRI score and says so,
        # rather than pretending a lab report is enough.
        #
        # NHANES independently confirms the negative: with 738 lifetime prostate
        # cases and no PSA available, bloodwork added -0.000 over age.
        # See experiments/prostate_biopsy.py and experiments/pooled_sites.py.
        "file": "prostate_biopsy_zenodo.csv",
        "label": "Prostate Cancer Risk, with an MRI score",
        "features": {
            "age": lambda d: d["Age"],
            "psa": lambda d: d["PSA"],
            "prostate_volume": lambda d: d["Prostate_Volume"],
            "psa_density": lambda d: d["PSAD"],
            "bmi": lambda d: d["BMI"],
            "pi_rads": lambda d: d["PI_RADS"],
        },
        "target": lambda d: (d["Pathology"].astype(str).str.strip() == "Adenocarcinoma").astype(int),
        "positive_means": ("adenocarcinoma on biopsy, separated from a benign biopsy result "
                           "rather than from an unbiopsied man"),
    },
]


def build_ensemble(n_samples: int, pos_rate: float) -> VotingClassifier:
    """Soft-voting ensemble of XGBoost and Extra Trees, sized to the dataset."""
    small = n_samples < 300
    scale_pos = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0

    # Tree counts and depth are deliberately modest. The service runs on a
    # 512 MB instance and SHAP's TreeExplainer holds a second copy of every
    # tree structure, so an oversized forest is paid for twice. Measured on the
    # held-out split, shrinking these costs a fraction of a point of AUC and
    # roughly halves the resident footprint.
    xgb = XGBClassifier(
        n_estimators=120 if small else 200,
        max_depth=3 if small else 4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        min_child_weight=2,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ext = ExtraTreesClassifier(
        n_estimators=120 if small else 200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgboost", xgb), ("extra_trees", ext)],
        voting="soft",
        weights=[1, 1],
    )



def choose_threshold(y_true, p) -> float:
    """
    Pick the operating point, instead of assuming 0.5.

    0.5 is only the right cut when the classes are balanced. Once the panels
    moved to real population data at 4 to 9 percent prevalence, a well
    calibrated model almost never crosses 0.5, and measured sensitivity fell to
    0.008. That is not a broken model, it is a broken threshold: the model was
    correctly reporting that almost nobody is more likely than not to have the
    disease.

    Youden's J, sensitivity plus specificity minus one, picks the point that
    best separates the two groups. It is computed on cross validated
    predictions inside the training data only, so the held-out split never
    influences it, and the chosen value is stored in the bundle so the API,
    the evaluation and the prospective analysis all use the same number.
    """
    fpr, tpr, cuts = roc_curve(y_true, p)
    j = tpr - fpr
    best = cuts[int(np.argmax(j))]
    # roc_curve can return an infinite first cut point.
    if not np.isfinite(best):
        best = 0.5
    return float(np.clip(best, 0.01, 0.99))


def evaluate(clf_factory, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Cross validated AUC, a held-out confusion matrix, and the operating point.

    Everything here runs on the CALIBRATED estimator, which matters. Choosing a
    threshold on uncalibrated out-of-fold scores and then applying it to
    calibrated probabilities compares two different scales, and the symptom is a
    threshold that looks reasonable while catching almost nobody.
    """
    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=RANDOM_STATE)

    def calibrated():
        inner = StratifiedKFold(n_splits=max(2, min(3, folds)), shuffle=True,
                                random_state=RANDOM_STATE)
        return CalibratedClassifierCV(clf_factory(), method="isotonic", cv=inner)

    # Out-of-fold predictions on the calibrated pipeline. These set both the
    # reported CV AUC and the threshold, and neither sees the held-out split.
    oof = cross_val_predict(calibrated(), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    cv_auc = roc_auc_score(y, oof)
    threshold = choose_threshold(y, oof)

    fold_aucs = []
    for tr, te in cv.split(X, y):
        m = calibrated().fit(X.iloc[tr], y.iloc[tr])
        fold_aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    holdout = calibrated().fit(X_tr, y_tr)
    proba = holdout.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    return {
        "auc": round(float(cv_auc), 3),
        "auc_std": round(float(np.std(fold_aucs)), 3),
        "accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 3),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 3),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_features": int(X.shape[1]),
        "cv_folds": int(cv.get_n_splits()),
        "threshold": round(float(threshold), 4),
    }


def prepare(config: dict):
    path = os.path.join(DATA_DIR, config["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    X = pd.DataFrame({key: fn(df) for key, fn in config["features"].items()})
    X = X.apply(pd.to_numeric, errors="coerce")

    unknown = [c for c in X.columns if c not in APP_FIELDS]
    if unknown:
        raise ValueError(f"{config['name']}: features outside the app schema: {unknown}")

    y = config["target"](df)

    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    return X, y, medians


def load_held_out() -> dict:
    """
    Held-out test results from evaluate.py, keyed by domain.

    These are the numbers that belong in front of a user: measured on a split
    that was cut before anything was fitted, with bootstrap intervals and with
    precision projected onto real SEER incidence. Run evaluate.py first.
    """
    if not os.path.isfile("evaluation.json"):
        print("NOTE: evaluation.json not found, run evaluate.py for held-out metrics.")
        return {}
    with open("evaluation.json") as f:
        raw = json.load(f)
    out = {}
    for name, r in raw.items():
        c = r["calibrated"]
        out[name] = {
            "n_test": r["n_test"],
            "auc": c["auc"],
            "auc_ci": c["auc_ci"],
            "sensitivity": c["sensitivity"],
            "sensitivity_ci": c["sensitivity_ci"],
            "specificity": c["specificity"],
            "specificity_ci": c["specificity_ci"],
            "brier": c["brier"],
            "calibration_slope": c["calibration_slope"],
            "ppv_at_population_prevalence": c["ppv_at_population_prevalence"],
            "people_flagged_per_true_case": c["people_flagged_per_true_case"],
            "population_prevalence": c["population_prevalence"],
            "prevalence_source": c["prevalence_source"],
            "cohort_prevalence": r["cohort_prevalence"],
            "baseline_logistic_auc": r["baselines"]["logistic_regression"]["auc"],
            "baseline_age_sex_auc": r["baselines"].get("age_sex_only", {}).get("auc"),
            "subgroups": r["subgroups"],
        }
    return out


HELD_OUT = {}


def main():
    global HELD_OUT
    HELD_OUT = load_held_out()

    for d in MODEL_DIRS:
        os.makedirs(d, exist_ok=True)

    summary = {}

    for config in DATASETS:
        name = config["name"]
        print(f"\n=== {name} ===")
        try:
            X, y, medians = prepare(config)
        except FileNotFoundError as e:
            print(f"  skipped, data file missing: {e}")
            continue

        pos_rate = float(y.mean())
        print(f"  {len(y)} rows, {int(y.sum())} positive ({pos_rate:.1%}), "
              f"{X.shape[1]} features: {', '.join(X.columns)}")

        if name in WITHDRAWN:
            print(f"  WITHDRAWN, not shipped: {WITHDRAWN[name][:70]}...")
            for d in MODEL_DIRS:
                stale = os.path.join(d, f"model_{name}.joblib")
                if os.path.isfile(stale):
                    os.remove(stale)
                    print(f"  removed stale bundle {stale}")
            summary[name] = {
                "label": config["label"],
                "shipped": False,
                "withdrawn_reason": WITHDRAWN[name],
                "cohort_design": COHORT_DESIGN[name],
            }
            continue

        # Pick the model that actually wins rather than assuming the ensemble
        # does. Selection is by cross validated AUC inside the training data,
        # never on the held-out split, so this does not leak. On the liver
        # panel plain logistic regression beats the ensemble outright, and it
        # also generalises better across cohorts in external_validation.py,
        # which is the sort of thing an unexamined "use the fancy model"
        # default would hide.
        ensemble_factory = lambda: build_ensemble(len(y), pos_rate)
        logistic_factory = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        )

        folds_sel = max(2, min(5, int(y.value_counts().min())))
        cv_sel = StratifiedKFold(n_splits=folds_sel, shuffle=True, random_state=RANDOM_STATE)
        candidates = {}
        for cand_name, cand in [("ensemble", ensemble_factory), ("logistic", logistic_factory)]:
            p = cross_val_predict(cand(), X, y, cv=cv_sel, method="predict_proba")[:, 1]
            candidates[cand_name] = round(float(roc_auc_score(y, p)), 3)
        chosen = max(candidates, key=lambda k: candidates[k])
        print(f"  model selection by CV AUC: {candidates}  ->  {chosen}")

        factory = ensemble_factory if chosen == "ensemble" else logistic_factory
        algorithm = (
            "Soft-voting ensemble: XGBoost + Extra Trees, isotonic calibrated"
            if chosen == "ensemble"
            else "Logistic regression, isotonic calibrated, selected over the ensemble on CV AUC"
        )
        metrics = evaluate(factory, X, y)
        metrics["model_selection"] = candidates
        metrics["chosen_model"] = chosen
        print(f"  AUC {metrics['auc']} +/- {metrics['auc_std']}   "
              f"acc {metrics['accuracy']}   sens {metrics['sensitivity']}   "
              f"spec {metrics['specificity']}")

        # Isotonic calibration so the percentage the interface shows a person
        # corresponds to an observed frequency rather than a raw forest vote.
        # CalibratedClassifierCV does its own internal cross validation, so the
        # calibrator never sees the data it is scoring.
        folds = max(2, min(5, int(y.value_counts().min())))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        model = CalibratedClassifierCV(factory(), method="isotonic", cv=cv)
        model.fit(X, y)
        print("  calibrated (isotonic)")

        # Held-out evidence from evaluate.py, attached so the API can report it.
        held_out = HELD_OUT.get(name, {})

        bundle = {
            "model": model,
            "feature_names": list(X.columns),
            "feature_medians": {k: float(v) for k, v in medians.items()},
            "config_name": name,
            "label": config["label"],
            "positive_means": config["positive_means"],
            "metrics": metrics,
            "threshold": metrics.get("threshold", 0.5),
            "held_out": held_out,
            "cohort_design": COHORT_DESIGN[name],
            "algorithm": algorithm,
        }
        # compress=3 keeps the bundles small enough to commit and deploy.
        for d in MODEL_DIRS:
            joblib.dump(bundle, os.path.join(d, f"model_{name}.joblib"), compress=3)
        print(f"  saved to {' and '.join(MODEL_DIRS)}")

        summary[name] = {
            "label": config["label"],
            "shipped": True,
            "features": list(X.columns),
            "cohort_design": COHORT_DESIGN[name],
            **metrics,
            "held_out": held_out,
        }

    with open(os.path.join("backend", "model_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote backend/model_metrics.json")

    print("\n" + "=" * 92)
    print(f"{'panel':<13}{'shipped':<10}{'test AUC':<12}{'95% CI':<20}{'PPV@pop':<11}{'flagged/case'}")
    print("=" * 92)
    for name, m in summary.items():
        if not m.get("shipped"):
            print(f"{name:<13}{'NO':<10}withdrawn")
            continue
        h = m.get("held_out", {})
        ci = h.get("auc_ci", ["", ""])
        ci_text = f"{ci[0]} to {ci[1]}"
        ppv_text = f"{h.get('ppv_at_population_prevalence', 0) * 100:.2f}%"
        print(f"{name:<13}{'yes':<10}{str(h.get('auc', '')):<12}{ci_text:<20}"
              f"{ppv_text:<11}{h.get('people_flagged_per_true_case', '')}")


if __name__ == "__main__":
    main()
