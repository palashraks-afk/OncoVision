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
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
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

# What kind of question each panel answers, and what the user must already have.
#
# This distinction existed in the documentation and nowhere the user could see
# it, which made the whole application read as "upload your lab report and get
# eight cancer risks". Four of the eight are not that:
#
#   SCREENING       reads routine bloodwork. Anyone can run it.
#   TRIAGE          runs AFTER something has been found, and its precision is
#                   quoted against prevalence in the referred group, not the
#                   street. Offering it to an unselected person is meaningless.
#   INTERPRETATION  reads a diagnostic test that has already been performed:
#                   a biopsy, or an MRI. It cannot be run from a lab report at
#                   all, and the inputs simply will not exist for most people.
#
# Shown on every card, because a demo that implies all eight work the same way
# is overselling four of them.
PANEL_KIND = {
    "general":    ("screening", "Reads routine bloodwork and your history. Anyone can run it."),
    "liver":      ("screening", "Reads routine bloodwork and your history. Anyone can run it."),
    "colorectal": ("screening", "Reads a routine blood count. Anyone can run it."),
    "lung":       ("screening", "Reads routine bloodwork. Offered to people with tobacco exposure, "
                                "which is who lung screening is for."),
    "ovarian":    ("triage", "For a woman who ALREADY has an ovarian mass found on imaging. It "
                             "asks whether that mass is malignant, not whether one exists."),
    "prostate":   ("interpretation", "Needs a PI-RADS score from a prostate MRI. Without it this "
                                     "panel only matches reading the PSA number, so it is not a "
                                     "lab-report test."),
    "breast":     ("interpretation", "Needs nuclear measurements from a breast biopsy that has "
                                     "already been taken and imaged. It interprets that biopsy; "
                                     "it does not screen for one."),
    "pancreatic": ("triage", "For someone ALREADY being investigated for pancreatic or biliary "
                             "disease. It reads CA 19-9, which is a tumour marker ordered when "
                             "cancer is suspected rather than part of routine bloodwork, and its "
                             "controls were people with benign disease of the same organs. It "
                             "asks which of those two a sick person has."),
}

# Above this many people flagged per true case, a panel is not a screening
# instrument whatever its AUC says, because acting on it would mean investigating
# hundreds of healthy people to find one cancer. Measured in
# experiments/operating_point.py, which also shows that for bowel, lung and
# pancreatic NO threshold fixes it.
NOT_SCREENING_ABOVE = 50.0

# Panels that can still say something useful from a lab report alone, when the
# test that defines their full version has not been done.
#
# The point of this application is to read the bloodwork somebody already has.
# A panel that refuses until you have had an MRI is not serving that person, and
# the honest response is not silence but a weaker answer clearly labelled as
# weaker. Only listed where a reduced model was measured and found to beat
# chance by a useful margin.
REDUCED_INPUTS = {
    # 0.676 against 0.825 with the MRI, on the same 212 biopsied men. Weak, and
    # available to a man holding an ordinary lab report.
    "prostate": ["age", "psa", "bmi"],
}
# {auc} is filled with the measured cross-validated AUC of the reduced model at
# training time, so this note cannot drift away from the number it quotes.
REDUCED_NOTE = {
    "prostate": ("Scored from your PSA and age alone, without a prostate MRI. "
                 "This is the weaker version of this panel: it reaches {auc} "
                 "here against 0.825 when a PI-RADS score from an MRI is "
                 "available. Treat it as a reason to ask about an MRI, not as a "
                 "substitute for one."),
}
# What to call a panel when only its reduced tier could run. The full label
# names a test the user has not had, which reads as a mistake on their card.
REDUCED_LABEL = {
    "prostate": "Prostate Cancer Risk, from PSA alone",
}

# A panel that beats knowing someone's age and sex by less than this is close to
# a demographic lookup, and the card should say so rather than letting a good
# looking AUC imply the lab values did the work. The general panel gains 0.006,
# and adding serum cotinine, CRP and the full blood count moved it to 0.009,
# so it is not for want of trying: see experiments/general_rescue.py.
BARELY_BEATS_DEMOGRAPHICS = 0.02


# Cohort design, stated on every panel because it bounds what the numbers mean.
COHORT_DESIGN = {
    "general": "23,923 US adults from NHANES 2005 to 2014, nationally representative. The "
               "target is a cancer diagnosis within four years of the blood draw, with "
               "long-ago survivors excluded, so this is a screening question rather than a "
               "lifetime one. It adds 0.006 over knowing age and sex alone, measured on "
               "repeated paired folds rather than one split, which is small and is stated "
               "rather than hidden. Adding routine bloodwork was measured and made it worse, "
               "so this panel reads risk factors and not the lab report. Treat even that "
               "0.006 with suspicion: the same question asked prospectively, on 33,834 people "
               "with death-certificate outcomes, gained 0.013 from bloodwork inside its own "
               "survey and LOST 0.013 when tested on a cohort from a different decade. No "
               "external cohort has confirmed that routine bloodwork predicts undifferentiated "
               "cancer risk, and one careful attempt to confirm it failed.",
    "breast": "Case-control and post-biopsy. Precision is projected onto the roughly 25 percent "
              "malignancy rate among breast lesions taken to biopsy, not onto SEER population "
              "incidence, because nobody gets an aspirate without a lesion being found first. "
              "This panel reads an aspirate that has already "
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
    # Breast aspirate morphology, all thirty Wisconsin measurements.
    # This used to be four. The restriction came from a time when the
    # application asked for four numbers, but breast is an INTERPRETATION
    # panel: whoever runs it is holding a pathology report that carries all
    # thirty. Measured, the four-feature version scores 0.680 on the smallest
    # third of lesions and the full set scores 0.952, which is the difference
    # between useless and useful exactly where an early answer matters.
    # See experiments/breast_small_lesions.py.
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Patient history, answered by the user rather than read off a lab report.
# gender             0 female, 1 male
# smoking            0 never, 1 former, 2 current
# alcohol_intake     0 none through 5 heavy
# genetic_risk       0 low, 1 medium, 2 high
# remaining flags    0 no, 1 yes
# Only fields a shipped panel actually consumes. Inherited risk, prior cancer
# diagnosis, family history and cirrhosis were dropped: no panel uses them now
# that general and liver train on NHANES, and asking a patient for information
# nothing reads is just friction.
#
# Exercise went the same way, and was given a fair hearing first. NHANES records
# it from 2007 under the Global Physical Activity Questionnaire, so it was pulled
# and offered to the general panel. It did not help: the panel's gain over age
# and sex went from +0.005 to +0.003, a loss, and the same on the full pooled
# cohort. A real cancer risk factor at the population level is not the same thing
# as one that adds information once you already know a person's age, sex, BMI,
# smoking and drinking. See experiments/general_body_activity.py.
HISTORY_FIELDS = [
    "gender", "smoking", "alcohol_intake",
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
        # All thirty nuclear morphology measurements. Restricting to four means
        # cost 0.272 of AUC on the smallest third of lesions, which is the group
        # where an earlier answer changes anything.
        "features": {
            "radius_mean": lambda d, c="radius_mean": d[c],
            "texture_mean": lambda d, c="texture_mean": d[c],
            "perimeter_mean": lambda d, c="perimeter_mean": d[c],
            "area_mean": lambda d, c="area_mean": d[c],
            "smoothness_mean": lambda d, c="smoothness_mean": d[c],
            "compactness_mean": lambda d, c="compactness_mean": d[c],
            "concavity_mean": lambda d, c="concavity_mean": d[c],
            "concave_points_mean": lambda d, c="concave points_mean": d[c],
            "symmetry_mean": lambda d, c="symmetry_mean": d[c],
            "fractal_dimension_mean": lambda d, c="fractal_dimension_mean": d[c],
            "radius_se": lambda d, c="radius_se": d[c],
            "texture_se": lambda d, c="texture_se": d[c],
            "perimeter_se": lambda d, c="perimeter_se": d[c],
            "area_se": lambda d, c="area_se": d[c],
            "smoothness_se": lambda d, c="smoothness_se": d[c],
            "compactness_se": lambda d, c="compactness_se": d[c],
            "concavity_se": lambda d, c="concavity_se": d[c],
            "concave_points_se": lambda d, c="concave points_se": d[c],
            "symmetry_se": lambda d, c="symmetry_se": d[c],
            "fractal_dimension_se": lambda d, c="fractal_dimension_se": d[c],
            "radius_worst": lambda d, c="radius_worst": d[c],
            "texture_worst": lambda d, c="texture_worst": d[c],
            "perimeter_worst": lambda d, c="perimeter_worst": d[c],
            "area_worst": lambda d, c="area_worst": d[c],
            "smoothness_worst": lambda d, c="smoothness_worst": d[c],
            "compactness_worst": lambda d, c="compactness_worst": d[c],
            "concavity_worst": lambda d, c="concavity_worst": d[c],
            "concave_points_worst": lambda d, c="concave points_worst": d[c],
            "symmetry_worst": lambda d, c="symmetry_worst": d[c],
            "fractal_dimension_worst": lambda d, c="fractal_dimension_worst": d[c],
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
            # GGT sits on the same comprehensive metabolic panel and was simply
            # never pulled. It is what tells you a raised alkaline phosphatase
            # came from the liver and not from bone, so the panel was reading
            # ALP without the value that disambiguates it. Worth +0.009 AUC,
            # winning 5 of 5 paired repeats with no overlap between the two
            # ranges. Globulin, LDH and uric acid were tested alongside it and
            # added nothing once GGT was in, so only GGT is taken:
            # see experiments/liver_extra_analytes.py.
            "ggt": lambda d: d["ggt"],
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



# A panel flagging more than this many people per true case is not a screening
# instrument, it is an anxiety generator, and the threshold should be tightened
# if tightening helps.
PRECISION_TROUBLE = 50.0
# How much sensitivity a panel is allowed to give up chasing precision. Below
# this it starts missing most of the cancers it exists to find, which is a worse
# failure than a false alarm.
SENSITIVITY_FLOOR = 0.70


# The sensitivity a rule-out point has to hit before it is worth offering.
#
# The shipped threshold is Youden's J, which balances a false positive against a
# false negative as though the two cost the same. Before an expensive
# diagnostic they do not: a false positive costs a colonoscopy and a false
# negative costs a life. experiments/cost_model.py finds that at Youden the
# bowel panel misses 190 cancers in 400 and stops saving money the moment a
# missed cancer is priced at a life rather than at a treatment bill, while at
# high sensitivity it avoids 36,052 colonoscopies per 100,000 people and misses
# 8.
#
# So every panel also carries a rule-out point: the least aggressive threshold
# that still catches this share of cases. It answers a different and more useful
# question than "are you flagged", namely "is there enough here to leave you
# out".
RULE_OUT_SENSITIVITY = 0.95


def rule_out_threshold(y_true, p):
    """
    The highest threshold that still keeps sensitivity at RULE_OUT_SENSITIVITY.

    Returns the threshold and what it buys, so the interface can say how many
    people it would exclude and how many cases it would miss doing so, rather
    than presenting a cut with no consequences attached.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if y_true.sum() == 0:
        return None
    order = np.unique(p)
    best = None
    for thr in order:
        flagged = p >= thr
        sens = float(flagged[y_true == 1].mean())
        if sens < RULE_OUT_SENSITIVITY:
            continue
        spec = float((~flagged)[y_true == 0].mean())
        best = {
            "threshold": float(thr),
            "sensitivity": round(sens, 3),
            "specificity": round(spec, 3),
            "share_ruled_out": round(float((~flagged).mean()), 3),
            "cases_missed_per_100": round((1 - sens) * 100, 1),
            "target_sensitivity": RULE_OUT_SENSITIVITY,
        }
    return best


def choose_threshold(y_true, p, prevalence: float | None = None) -> float:
    """
    Pick the operating point, instead of assuming 0.5.

    0.5 is only the right cut when the classes are balanced. Once the panels
    moved to real population data at 4 to 9 percent prevalence, a well
    calibrated model almost never crosses 0.5, and measured sensitivity fell to
    0.008. That is not a broken model, it is a broken threshold: the model was
    correctly reporting that almost nobody is more likely than not to have the
    disease.

    Youden's J, sensitivity plus specificity minus one, is the default. It picks
    the point that best separates the two groups.

    BUT Youden weights a false positive and a false negative equally, and takes
    no account of how rare the disease is. On a cancer with a prevalence of
    0.13 percent that is the wrong objective: it chose a breast threshold that
    flags 65 people for every true case, where tightening to 99 percent
    specificity flags 13 and costs 16 points of sensitivity. Five times the
    precision for that is a trade worth making, and Youden cannot see it because
    prevalence is not in its formula.

    So when a prevalence is supplied and Youden lands somewhere with poor
    precision, the threshold is re-chosen to maximise precision subject to
    keeping sensitivity at or above SENSITIVITY_FLOOR. If no such point improves
    matters, Youden stands.

    Everything is computed on cross validated predictions inside the training
    data only, so the held-out split never influences it, and the chosen value
    is stored in the bundle so the API, the evaluation and the prospective
    analysis all use the same number.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)

    fpr, tpr, cuts = roc_curve(y_true, p)
    j = tpr - fpr
    best = cuts[int(np.argmax(j))]
    # roc_curve can return an infinite first cut point.
    if not np.isfinite(best):
        best = 0.5
    best = float(np.clip(best, 0.01, 0.99))

    if prevalence is None or not (0 < prevalence < 1):
        return best

    def flagged_per_case(thr):
        pred = (p >= thr).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        if (tp + fn) == 0 or (tn + fp) == 0:
            return float("inf"), 0.0
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        num = sens * prevalence
        den = num + (1 - spec) * (1 - prevalence)
        if den <= 0 or num <= 0:
            return float("inf"), sens
        return 1.0 / (num / den), sens

    per_case_at_youden, _ = flagged_per_case(best)
    if not np.isfinite(per_case_at_youden) or per_case_at_youden <= PRECISION_TROUBLE:
        return best

    # Sweep upward from Youden and keep the most precise point that still
    # retains enough sensitivity to be worth running.
    candidates = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 300)))
    chosen, chosen_per_case = best, per_case_at_youden
    for thr in candidates:
        if thr <= best:
            continue
        per_case, sens = flagged_per_case(float(thr))
        if sens < SENSITIVITY_FLOOR:
            break          # sweep is monotone in sensitivity, so stop here
        if per_case < chosen_per_case:
            chosen, chosen_per_case = float(thr), per_case
    return float(np.clip(chosen, 0.01, 0.99))


def evaluate(clf_factory, X: pd.DataFrame, y: pd.Series,
             prevalence: float | None = None) -> dict:
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
    threshold = choose_threshold(y, oof, prevalence)
    rule_out = rule_out_threshold(y, oof)

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
        # The rule-out point, alongside the balanced one. See
        # RULE_OUT_SENSITIVITY for why a panel used before an expensive
        # procedure needs both.
        "rule_out": rule_out,
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


def load_demographic_gain() -> dict:
    """
    How much each panel adds over age and sex, measured across repeated folds.

    Not from the held-out split. The bowel panel's held-out split says it LOSES
    to age and sex by 0.024 while twenty paired repeats say it WINS by 0.039,
    and that split has already been shown to be an unrepresentative draw.
    Driving a user-facing "barely beats demographics" warning off it would
    repeat the mistake that got the cervical panel withdrawn.
    """
    path = "experiments/demographic_gain_result.json"
    if not os.path.isfile(path):
        print("NOTE: demographic_gain_result.json not found, "
              "run experiments/demographic_gain.py.")
        return {}
    with open(path) as f:
        return json.load(f)


def load_fairness() -> dict:
    """
    Per-group AUC from experiments/fairness.py, keyed by domain.

    Carried on the bundle so the interface can show it next to the headline
    number. A panel that works measurably worse for one group and does not say
    so is making a claim it has not earned, and for bowel and lung the group
    that comes out worst is the one with the higher mortality from that cancer,
    which is the opposite of a harmless gap.
    """
    path = "experiments/fairness_result.json"
    if not os.path.isfile(path):
        print("NOTE: fairness_result.json not found, run experiments/fairness.py.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for name, r in raw.items():
        if name.startswith("_") or not isinstance(r, dict):
            continue
        worse = [g for g, v in r.get("groups", {}).items() if v.get("materially_worse")]
        out[name] = {
            "overall_auc": r.get("overall_auc"),
            "groups": r.get("groups", {}),
            "unmeasurable_groups": r.get("unmeasurable_groups", []),
            "worst_group": r.get("worst_group"),
            "worst_auc": r.get("worst_auc"),
            "spread": r.get("spread"),
            "materially_worse_groups": worse,
        }
    return out


def load_split_stability() -> dict:
    """
    Repeated-split results from experiments/split_stability.py, keyed by domain.

    A single held-out split is one draw from a distribution, and on a small
    cohort that distribution is wide. Two panels have already been caught by
    this: cervical published a 0.725 that sat at the 97th percentile of its own
    splits and was withdrawn, and colorectal drew a split where it scores 0.793
    against an 0.817 age-and-sex baseline while over twenty repeats it beats
    that baseline by 0.038.

    So every panel now also carries the mean across repeated splits and the
    spread, and the interface shows both. Where the shipped split is
    unrepresentative, the user can see that rather than taking one draw as the
    answer.
    """
    path = "experiments/split_stability_result.json"
    if not os.path.isfile(path):
        print("NOTE: split_stability_result.json not found, "
              "run experiments/split_stability.py for the stable estimate.")
        return {}
    with open(path) as f:
        raw = json.load(f).get("panels", {})
    return {
        name: {
            "stable_auc": r["mean_auc"],
            "split_spread": [r["min_auc"], r["max_auc"]],
            "split_sd": r["std_auc"],
            "shipped_split_percentile": r["shipped_split_percentile"],
            "n_splits": r["n_splits"],
        }
        for name, r in raw.items()
    }


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
STABILITY = {}
FAIRNESS = {}
DEMO_GAIN = {}


def main():
    global HELD_OUT, STABILITY, FAIRNESS, DEMO_GAIN
    HELD_OUT = load_held_out()
    STABILITY = load_split_stability()
    FAIRNESS = load_fairness()
    DEMO_GAIN = load_demographic_gain()

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
        # The prevalence this panel will actually meet, so the operating point
        # can weigh a false alarm against a miss the way the real world does.
        prev = None
        try:
            from evaluate import SEER_INCIDENCE
            if name in SEER_INCIDENCE:
                prev = SEER_INCIDENCE[name][0] / 100_000.0
        except Exception:
            prev = None
        metrics = evaluate(factory, X, y, prev)
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
        stability = STABILITY.get(name, {})
        fairness = FAIRNESS.get(name, {})
        # No default. A missing entry used to fall back to ("screening", ""),
        # which meant a new panel silently claimed to be a screening test with no
        # note explaining itself. That is the strongest claim a panel can make
        # and the one most likely to be wrong, so it has to be written down
        # deliberately. The pancreatic panel sat in that default for a while.
        if name not in PANEL_KIND:
            raise KeyError(
                f"panel {name!r} has no PANEL_KIND entry. Decide whether it is a "
                f"screening, triage or interpretation panel and say so in "
                f"PANEL_KIND before it can ship."
            )
        kind, kind_note = PANEL_KIND[name]

        # A weaker model that runs on what a lab report actually contains.
        #
        # This exists because gating a panel behind a test most people have not
        # had turns the application off for the person it was built for. The
        # prostate panel needs a PI-RADS score from an MRI to reach 0.825. A man
        # with a PSA on his annual bloodwork — the exact user this project is
        # for — was getting nothing at all.
        #
        # Measured, age and PSA and BMI reach 0.676 on the same cohort. That is
        # weak, and it is a great deal better than silence, and it is the
        # difference between a tool that answers the question it was built to
        # answer and one that refuses until you have already been to a
        # radiologist. The reduced model ships INSIDE the same bundle, so it is
        # one panel with two tiers rather than two panels competing for the same
        # card, and the interface says which tier answered.
        reduced = None
        if name in REDUCED_INPUTS:
            cols = [c for c in REDUCED_INPUTS[name] if c in X.columns]
            Xr = X[cols]
            r_factory = lambda: build_ensemble(len(y), float(y.mean()))  # noqa: E731
            r_folds = max(2, min(5, int(y.value_counts().min())))
            r_cv = StratifiedKFold(n_splits=r_folds, shuffle=True,
                                   random_state=RANDOM_STATE)
            r_model = CalibratedClassifierCV(r_factory(), method="isotonic", cv=r_cv)
            r_auc = float(np.mean(cross_val_score(
                CalibratedClassifierCV(r_factory(), method="isotonic", cv=r_cv),
                Xr, y, cv=r_cv, scoring="roc_auc")))
            r_model.fit(Xr, y)
            reduced = {
                "model": r_model,
                "feature_names": cols,
                "auc": round(r_auc, 3),
                "note": REDUCED_NOTE.get(name, "").format(auc=f"{r_auc:.3f}"),
                "label": REDUCED_LABEL.get(name, ""),
            }
            print(f"  reduced tier on {len(cols)} lab-report features: AUC {r_auc:.3f}")
        per_case = (held_out or {}).get("people_flagged_per_true_case")
        # A screening panel that flags hundreds per case is not one.
        screening_viable = not (
            kind == "screening" and per_case is not None and per_case > NOT_SCREENING_ABOVE
        )
        # How much this panel adds over simply knowing age and sex.
        dg = DEMO_GAIN.get(name, {})
        demo_gain = dg.get("gain")
        barely_beats_demographics = bool(dg.get("barely_beats_demographics", False))

        bundle = {
            "model": model,
            "feature_names": list(X.columns),
            # The lab-report-only tier, when this panel has one. See
            # REDUCED_INPUTS for why a weaker answer beats a refusal here.
            "reduced": reduced,
            "feature_medians": {k: float(v) for k, v in medians.items()},
            # The range each feature was actually observed over, so the service
            # can refuse to extrapolate.
            #
            # This exists because of a genuinely dangerous failure. A tree model
            # has no splits beyond its training data, so past that edge it
            # returns whatever leaf it happens to land in, with full confidence.
            # The liver panel scored a coherent acute-hepatitis pattern — ALT
            # 300, AST 260, GGT 200, bilirubin 2.5 — at 3.0 percent, LOWER than
            # a completely normal patient at 3.3 percent and far below a mild
            # abnormality at 14.1 percent.
            #
            # It was not a bug in the model. Only 19 of 35,511 people in this
            # cohort have an ALT above 250, and among the 1,436 WITH liver
            # disease the highest ALT is 232. So every high-ALT person in the
            # training data is a non-case, and the model learned that a very
            # high ALT means no liver disease. That is true of NHANES, whose
            # label is self-reported "were you ever told you had a liver
            # condition", and false of medicine: someone in acute hepatitis at
            # the time of the survey has not been told yet.
            #
            # The model is right about its data and wrong about the world, and
            # no amount of retraining on this cohort fixes that. What the
            # service can do is decline to rank values it has no evidence about,
            # which is what these percentiles are for.
            "feature_ranges": {
                k: [float(X[k].quantile(0.01)), float(X[k].quantile(0.99))]
                for k in X.columns if pd.api.types.is_numeric_dtype(X[k])
            },
            "config_name": name,
            "label": config["label"],
            "positive_means": config["positive_means"],
            "metrics": metrics,
            "threshold": metrics.get("threshold", 0.5),
            "held_out": held_out,
            "stability": stability,
            "fairness": fairness,
            "panel_kind": kind,
            "panel_kind_note": kind_note,
            "screening_viable": screening_viable,
            "gain_over_age_sex": demo_gain,
            "barely_beats_demographics": barely_beats_demographics,
            "flagged_per_true_case": per_case,
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
            "stability": stability,
            "fairness": fairness,
            "panel_kind": kind,
            "panel_kind_note": kind_note,
            "screening_viable": screening_viable,
            "gain_over_age_sex": demo_gain,
            "barely_beats_demographics": barely_beats_demographics,
            "flagged_per_true_case": per_case,
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
