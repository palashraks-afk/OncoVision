"""
Tests for the symptom-and-lab navigator.

The cases are taken from the recommendation boxes of the NICE NG12 full guideline
(June 2015), which is what navigator_rules.json encodes, so a failing test means the
engine and the guideline disagree, not that someone's opinion changed. The second half
tests properties that must hold for any input and matter more than any one rule: it
must not alarm a person with nothing to report, must not crash on empty or impossible
input, must never state that someone has cancer, and must ask for missing facts instead
of assuming them.

Run:  python -m pytest tests/test_navigator.py -q
"""

import json
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
import navigator as nav  # noqa: E402


def S(*keys, **freq):
    """Symptom list. A key may be given a frequency: S('bloating', bloating=15)."""
    return [{"key": k, "times_per_month": freq.get(k)} for k in keys]


def run(**kw):
    return nav.evaluate(kw)


def ids(result, tier=None):
    return {m["id"] for m in result["matches"] if tier is None or m["tier"] == tier}


# ------------------------------------------------------------ the rule file
def test_knowledge_base_is_internally_consistent():
    assert nav.validate_knowledge_base() == []


def test_every_rule_names_its_source_and_the_file_admits_it_is_unreviewed():
    kb = nav.load()
    assert all(r["source"] for r in kb["rules"])
    assert kb["meta"]["review_status"].startswith("NOT YET REVIEWED")
    assert kb["meta"]["reviewed_by"] is None


# ------------------------------------------------------------- lung and chest
def test_haemoptysis_at_40_is_a_prompt_check_but_not_at_39():
    assert "lung_haemoptysis" in ids(run(age=40, symptoms=S("haemoptysis")), "talk_soon")
    assert "lung_haemoptysis" not in ids(run(age=39, symptoms=S("haemoptysis")))


def test_two_chest_symptoms_trigger_an_xray_and_one_does_not_for_a_never_smoker():
    two = run(age=55, symptoms=S("cough", "fatigue"))
    assert "lung_xray_symptoms" in ids(two, "talk_soon")
    normal = {"hemoglobin": 14.5, "ferritin": 80, "platelets": 250, "wbc": 7.0, "mcv": 90}
    one_never = run(age=55, sex="male", ever_smoked=False, asbestos=False, symptoms=S("cough"),
                    labs=normal)
    assert "lung_xray_symptoms" not in ids(one_never)
    assert one_never["state"] == "nothing_meets"
    # Without a platelet count a cough at this age could still reach the 'consider' rule,
    # so the engine asks for it.
    no_labs = run(age=55, ever_smoked=False, asbestos=False, symptoms=S("cough"))
    assert no_labs["state"] == "need_info"
    assert any("platelet" in n for c in no_labs["could_apply_if"] for n in c["needs"])
    # With asbestos exposure unanswered the mesothelioma rule could still apply, so the
    # engine asks instead of assuming. That is the behaviour, not an oversight.
    unasked = run(age=55, ever_smoked=False, symptoms=S("cough"))
    assert unasked["state"] == "need_info"
    assert any("asbestos" in n for c in unasked["could_apply_if"] for n in c["needs"])


def test_one_chest_symptom_in_a_smoker_triggers_and_unknown_smoking_is_asked_for():
    assert "lung_xray_symptoms" in ids(run(age=55, ever_smoked=True, symptoms=S("cough")))
    unknown = run(age=55, symptoms=S("cough"))
    assert "lung_xray_symptoms" not in ids(unknown)
    assert unknown["state"] == "need_info"
    assert any("smoked" in n for c in unknown["could_apply_if"] for n in c["needs"])


def test_a_high_platelet_count_alone_does_not_prompt_a_chest_xray_in_a_symptom_free_adult():
    # Gated on purpose: 3.7% of symptom-free adults over 40 have platelets above 400, and
    # the guideline's lung recommendation is written for people being assessed for symptoms.
    # See design_decisions_for_clinician_review in the rule file.
    r = run(age=55, labs={"platelets": 450})
    assert "lung_xray_consider" not in ids(r)
    assert r["state"] in ("nothing_meets", "need_info")


def test_a_high_platelet_count_with_a_symptom_is_worth_raising_not_urgent():
    r = run(age=55, ever_smoked=False, asbestos=False, symptoms=S("cough"), labs={"platelets": 450})
    assert "lung_xray_consider" in ids(r, "worth_raising")
    assert r["state"] == "worth_raising"


def test_the_rule_file_records_the_deviation_it_made_from_the_guideline_text():
    notes = " ".join(nav.load()["meta"]["design_decisions_for_clinician_review"])
    assert "platelet" in notes and "deviation" in notes


def test_asbestos_exposure_counts_for_mesothelioma():
    assert "meso_xray_symptoms" in ids(run(age=60, asbestos=True, symptoms=S("shortness_of_breath")))
    assert "meso_xray_symptoms" not in ids(run(age=60, asbestos=False, ever_smoked=False,
                                                symptoms=S("shortness_of_breath")))


# ------------------------------------------------------------- bowel and gut
def test_iron_deficiency_anaemia_at_60_is_a_prompt_check_via_ferritin():
    r = run(age=62, sex="male", labs={"hemoglobin": 11.0, "ferritin": 8})
    assert "colorectal_urgent" in ids(r, "talk_soon")
    assert any("iron-deficiency anaemia" in b for m in r["matches"] for b in m["because"])


def test_anaemia_without_iron_deficiency_at_60_asks_for_a_stool_test_not_urgent_referral():
    r = run(age=62, sex="male", labs={"hemoglobin": 11.0, "ferritin": 60})
    assert "colorectal_urgent" not in ids(r)
    assert "colorectal_stool_test" in ids(r)


def test_small_red_cells_stand_in_for_ferritin_and_the_assumption_is_disclosed():
    r = run(age=62, sex="female", labs={"hemoglobin": 10.0, "mcv": 70})
    assert "colorectal_urgent" in ids(r)
    assert r["lab_notes"] and "assumed" in r["lab_notes"][0]


def test_anaemia_with_no_ferritin_or_cell_size_is_asked_about_not_assumed():
    r = run(age=62, sex="male", labs={"hemoglobin": 11.0})
    assert "colorectal_urgent" not in ids(r)
    assert any("ferritin" in n for c in r["could_apply_if"] for n in c["needs"])


def test_rectal_bleeding_thresholds_differ_by_age():
    assert "colorectal_urgent" in ids(run(age=50, symptoms=S("rectal_bleeding")))
    assert "colorectal_urgent" not in ids(run(age=49, symptoms=S("rectal_bleeding")))
    assert "colorectal_young_bleeding" in ids(run(age=45, symptoms=S("rectal_bleeding", "weight_loss")))


def test_dysphagia_at_any_age_and_weight_loss_with_indigestion_at_55():
    assert "upper_gi_urgent" in ids(run(age=30, symptoms=S("dysphagia")), "talk_soon")
    assert "upper_gi_urgent" in ids(run(age=55, symptoms=S("weight_loss", "dyspepsia")), "talk_soon")
    assert "upper_gi_urgent" not in ids(run(age=54, symptoms=S("weight_loss", "dyspepsia")))


def test_jaundice_at_40_and_weight_loss_with_diabetes_at_60_for_pancreas():
    assert "pancreas_jaundice" in ids(run(age=40, symptoms=S("jaundice")), "talk_soon")
    assert "pancreas_scan" in ids(run(age=60, symptoms=S("weight_loss", "new_onset_diabetes")))


# ----------------------------------------------------------- breast, ovary, prostate
def test_breast_lump_is_urgent_from_30_and_non_urgent_below():
    assert "breast_urgent" in ids(run(age=30, symptoms=S("breast_lump")), "talk_soon")
    young = run(age=29, symptoms=S("breast_lump"))
    assert "breast_young_lump" in ids(young, "mention")
    assert "breast_urgent" not in ids(young)


def test_ovarian_symptoms_need_a_frequency_of_12_days_a_month():
    freq = run(age=55, sex="female", symptoms=S("bloating", bloating=15))
    assert "ovarian_symptoms" in ids(freq, "worth_raising")
    rare = run(age=55, sex="female", symptoms=S("bloating", bloating=5))
    assert "ovarian_symptoms" not in ids(rare)
    unsaid = run(age=55, sex="female", symptoms=S("bloating"))
    assert "ovarian_symptoms" not in ids(unsaid)
    assert any("days a month" in n for c in unsaid["could_apply_if"] for n in c["needs"])


def test_ovarian_rules_do_not_apply_to_men():
    assert "ovarian_symptoms" not in ids(run(age=55, sex="male", symptoms=S("bloating", bloating=20)))


def test_a_raised_ca125_with_symptoms_asks_for_an_ultrasound():
    r = run(age=55, sex="female", symptoms=S("bloating", bloating=20), labs={"ca125": 60})
    assert "ovarian_ca125" in ids(r)


def test_prostate_rules_need_a_man_and_a_flagged_psa():
    assert "prostate_psa" in ids(run(age=65, sex="male", findings=["psa_flagged_high"]), "talk_soon")
    assert "prostate_psa" not in ids(run(age=65, sex="female", findings=["psa_flagged_high"]))
    assert "prostate_symptoms" in ids(run(age=65, sex="male", symptoms=S("lower_urinary_tract_symptoms")))


# --------------------------------------------------------- urine, blood, skin
def test_visible_blood_in_urine_is_urgent_from_45_for_bladder_and_kidney():
    r = run(age=45, symptoms=S("visible_haematuria"))
    assert {"bladder_urgent", "renal_haematuria"} <= ids(r, "talk_soon")
    assert "renal_haematuria" not in ids(run(age=44, symptoms=S("visible_haematuria")))


def test_children_with_pallor_get_a_two_day_blood_count_and_adults_only_a_consideration():
    child = run(age=8, symptoms=S("pallor", "fatigue"))
    assert "leukaemia_children_blood_count" in ids(child, "talk_soon")
    adult = run(age=30, symptoms=S("pallor"))
    assert "leukaemia_adult_blood_count" in ids(adult, "worth_raising")
    assert "leukaemia_adult_blood_count" not in ids(adult, "talk_soon")


def test_petechiae_in_a_child_is_immediate():
    r = run(age=8, symptoms=S("petechiae"))
    m = next(m for m in r["matches"] if m["id"] == "leukaemia_children_petechiae")
    assert m["timeframe"] == "immediate" and m["tier"] == "talk_soon"
    assert "seek urgent care now" in m["action"].lower()


def test_unexplained_weight_loss_is_prompt_at_any_age():
    assert "weight_loss" in ids(run(age=25, symptoms=S("weight_loss")), "talk_soon")


def test_swollen_glands_are_a_stronger_prompt_in_the_young():
    assert "lymphoma_children" in ids(run(age=15, symptoms=S("swollen_glands")))
    assert "lymphoma_adult" in ids(run(age=40, symptoms=S("swollen_glands")))
    assert "lymphoma_adult" not in ids(run(age=15, symptoms=S("swollen_glands")))


def test_melanoma_checklist_threshold_is_three_points():
    hi = run(age=40, skin_lesion={"change_in_size": True, "irregular_colour": True})
    assert "melanoma_7point" in ids(hi, "talk_soon") and hi["melanoma_score"] == 4
    lo = run(age=40, skin_lesion={"diameter_7mm": True})
    assert "melanoma_7point" not in ids(lo) and lo["melanoma_score"] == 1


# ------------------------------------------------------------- explained symptoms
def test_a_symptom_marked_as_explained_is_ignored():
    r = run(age=60, symptoms=[{"key": "weight_loss", "explained": True}])
    assert r["state"] == "nothing_meets"


# --------------------------------------------------------------- properties
def test_empty_input_is_handled_and_alarms_nobody():
    for kw in ({}, {"age": 50}, {"age": 50, "sex": "female"}, {"symptoms": []}, {"labs": {}}):
        r = nav.evaluate(kw)
        assert r["state"] in ("nothing_meets", "need_info")
        assert not r["matches"]


def test_a_symptom_free_person_with_normal_labs_is_never_told_to_talk_soon():
    rng = random.Random(7)
    for _ in range(300):
        sex = rng.choice(["male", "female"])
        age = rng.randint(1, 95)
        hb = rng.uniform(13.5, 16.5) if sex == "male" else rng.uniform(12.2, 15.5)
        r = nav.evaluate({"age": age, "sex": sex, "ever_smoked": rng.choice([True, False, None]),
                          "labs": {"hemoglobin": hb, "mcv": rng.uniform(82, 98), "ferritin": rng.uniform(30, 200),
                                   "platelets": rng.uniform(160, 380), "wbc": rng.uniform(4.5, 10.5),
                                   "ca125": rng.uniform(3, 30)}})
        assert not r["matches"], (age, sex, r["matches"])


def test_impossible_values_are_refused_with_a_readable_message():
    for bad in ({"age": -3}, {"age": 200}, {"labs": {"hemoglobin": 99}}, {"labs": {"platelets": -4}},
                {"symptoms": [{"key": "bloating", "times_per_month": 99}]}):
        with pytest.raises(ValueError):
            nav.evaluate(bad)


def test_unknown_symptom_keys_are_reported_not_silently_dropped():
    r = nav.evaluate({"age": 50, "symptoms": [{"key": "not_a_symptom"}]})
    assert r["ignored_symptoms"] == ["not_a_symptom"]


def test_the_output_never_claims_a_person_has_cancer_or_is_fine():
    banned = [r"you have cancer", r"you do not have cancer", r"you don't have cancer", r"you are fine",
              r"you're fine", r"diagnos(?:is|ed|e) of cancer", r"nothing is wrong\b(?!\.)"]
    rng = random.Random(11)
    keys = [s["key"] for s in nav.load()["symptoms"]]
    for _ in range(200):
        picked = rng.sample(keys, rng.randint(0, 6))
        r = nav.evaluate({"age": rng.randint(1, 90), "sex": rng.choice(["male", "female", None]),
                          "ever_smoked": rng.choice([True, False, None]),
                          "symptoms": [{"key": k, "times_per_month": rng.choice([None, 3, 20])} for k in picked]})
        text = json.dumps(r).lower()
        for pat in banned:
            assert not re.search(pat, text), (pat, picked)


def test_every_response_carries_the_safety_and_prototype_statements():
    r = nav.evaluate({"age": 50})
    assert any("seek urgent care" in s.lower() for s in r["safety"])
    assert "not medical advice" in r["disclaimer"].lower()
    assert r["rules"]["review_status"].startswith("NOT YET REVIEWED")


def test_tiers_are_ordered_most_actionable_first():
    r = run(age=62, sex="male", symptoms=S("weight_loss", "abdominal_pain"),
            labs={"hemoglobin": 11.0, "ferritin": 60, "platelets": 450})
    tiers = [m["tier"] for m in r["matches"]]
    assert tiers == sorted(tiers, key=lambda t: nav.TIER_ORDER[t])
    assert r["state"] == tiers[0] == "talk_soon"


# ---------------------------------------------------------------- context layer
def test_context_rises_with_age_and_smoking_and_is_a_group_figure():
    young = nav.evaluate({"age": 45, "sex": "female", "ever_smoked": False})["context"]
    old = nav.evaluate({"age": 75, "sex": "female", "ever_smoked": False})["context"]
    smoker = nav.evaluate({"age": 75, "sex": "female", "ever_smoked": True})["context"]
    assert young["per_1000"] < old["per_1000"] < smoker["per_1000"]
    assert "not your own risk" in old["text"]


def test_context_absent_without_age_sex_or_outside_range():
    assert nav.evaluate({"age": 60})["context"] is None
    assert nav.evaluate({"sex": "male"})["context"] is None
    assert nav.evaluate({"age": 30, "sex": "male"})["context"] is None
    assert nav.evaluate({"age": 95, "sex": "male"})["context"] is None


def test_context_says_when_smoking_is_assumed():
    assert nav.evaluate({"age": 60, "sex": "male"})["context"]["smoking_assumed_average"] is True


def test_mesothelioma_rule_no_longer_fires_for_a_smoker_without_asbestos():
    out = nav.evaluate({"age": 60, "sex": "male", "ever_smoked": True, "asbestos": False,
                        "symptoms": [{"key": "cough"}]})
    ids = {m["id"] for m in out["matches"]}
    assert "lung_xray_symptoms" in ids and "meso_xray_symptoms" not in ids
    out = nav.evaluate({"age": 60, "sex": "male", "ever_smoked": False, "asbestos": True,
                        "symptoms": [{"key": "cough"}]})
    assert "meso_xray_symptoms" in {m["id"] for m in out["matches"]}


def test_lab_only_alert_carries_the_survey_caveat_and_symptoms_remove_it():
    labs = {"hemoglobin": 10.5, "mcv": 72, "platelets": 250, "wbc": 6}
    out = nav.evaluate({"age": 65, "sex": "male", "labs": labs})
    assert out["matches"] and any("lab results alone" in n for n in out["lab_notes"])
    out = nav.evaluate({"age": 65, "sex": "male", "labs": labs, "symptoms": [{"key": "weight_loss"}]})
    assert not any("lab results alone" in n for n in out["lab_notes"])


def test_one_site_shows_only_its_highest_tier():
    out = nav.evaluate({"age": 66, "sex": "male", "labs": {"hemoglobin": 10.8, "mcv": 74, "ferritin": 9, "platelets": 310}})
    bowel = [m for m in out["matches"] if m["site"] == "bowel cancer"]
    assert bowel and all(m["tier"] == "talk_soon" for m in bowel)


def test_young_woman_with_low_iron_gets_a_periods_note_and_bowel_matches_get_a_version_note():
    out = nav.evaluate({"age": 44, "sex": "female", "labs": {"hemoglobin": 10.5, "ferritin": 6, "mcv": 76, "platelets": 300}})
    assert any("periods" in n for n in out["lab_notes"])
    assert any("2015" in n and "FIT" in n for n in out["guideline_notes"])
    out = nav.evaluate({"age": 44, "sex": "female", "symptoms": [{"key": "cough"}]})
    assert out["guideline_notes"] == []


def test_context_for_people_in_their_fifties_says_it_runs_low():
    out = nav.evaluate({"age": 55, "sex": "male", "ever_smoked": True})["context"]
    assert "ran low" in out["text"] and "minimum" in out["text"]
    assert "ran low" not in nav.evaluate({"age": 45, "sex": "male", "ever_smoked": True})["context"]["text"]
    assert "ran low" not in nav.evaluate({"age": 65, "sex": "male", "ever_smoked": True})["context"]["text"]
