"""
Which routine cancer screenings do US guidelines say a person is due for?

This is a lookup, not a prediction. It applies the US Preventive Services Task Force
recommendations for breast, cervical, colorectal, lung and prostate cancer to a person's
age, sex at birth and (for lung cancer) smoking history, and says plainly what it does not
know. The wording of every recommendation comes from the USPSTF pages named in
screening_rules.json; this file only decides which recommendation applies.

Three-valued, like the navigator: a fact that is not given is asked for, never assumed.
Status: research prototype. Not reviewed by a clinician. Average-risk adults only.
"""

import json
import os
from typing import Any, Dict, List, Optional

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screening_rules.json")
_KB: Optional[dict] = None

DISCLAIMER = ("This is a research prototype and not medical advice. It lists general recommendations for people at average "
              "risk. A doctor who knows your history, including any family history of cancer, may advise something different.")
FAMILY_NOTE = ("If a close relative had cancer young, you carry a known gene change, you have had cancer before, or you have had "
               "radiation to the chest, ask your doctor. You may need to start sooner or be checked more often.")

ORDER = {"due": 0, "individual": 1, "need_info": 2, "no_recommendation": 3, "not_yet": 4, "not_recommended": 5, "not_needed": 5}


def load() -> dict:
    global _KB
    if _KB is None:
        with open(PATH, encoding="utf-8") as f:
            _KB = json.load(f)
        _KB["_by_id"] = {i["id"]: i for i in _KB["items"]}
    return _KB


def _b(v) -> Optional[bool]:
    return v if isinstance(v, bool) else None


def _num(v, name, lo, hi) -> Optional[float]:
    if v is None or v == "":
        return None
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def evaluate(req: dict) -> dict:
    """Raises ValueError on impossible input."""
    kb = load()
    age = _num(req.get("age"), "age", 0, 120)
    if age is None:
        raise ValueError("age is needed")
    sex = req.get("sex_at_birth")
    if sex not in ("female", "male"):
        sex = None
    cervix = _b(req.get("has_cervix"))
    ever = _b(req.get("ever_smoked"))
    now = _b(req.get("smokes_now"))
    quit_years = _num(req.get("years_since_quit"), "years since quitting", 0, 90)
    pack_years = _num(req.get("pack_years"), "pack-years", 0, 400)

    out: List[Dict[str, Any]] = []
    need: List[str] = []
    by = kb["_by_id"]

    def add(item_id, status, headline, detail, needs=None, options=None):
        i = by[item_id]
        out.append({"id": item_id, "site": i["site"], "test": i["test"], "status": status, "headline": headline,
                    "detail": detail, "source": i["source"], "url": i["url"], "needs": needs or [], "options": options or []})

    # ---- breast
    if sex is None:
        add("breast", "need_info", "Breast cancer", "Mammogram advice depends on sex at birth.", ["sex at birth"])
        need.append("sex at birth")
    elif sex == "female":
        if 40 <= age <= 74:
            add("breast", "due", "Breast cancer: a mammogram every 2 years",
                "The Task Force recommends a mammogram every 2 years for women aged 40 to 74. It also applies to anyone assigned female at birth.")
        elif age >= 75:
            add("breast", "no_recommendation", "Breast cancer: no recommendation at 75 and older",
                "The Task Force said there is not enough evidence to recommend for or against mammograms at 75 and older. Ask your doctor.")
        else:
            add("breast", "not_yet", "Breast cancer: not yet",
                "The Task Force recommends starting at age 40 for people at average risk. Ask your doctor sooner if you have a family history or a gene change.")
    # ---- cervical
    if sex == "female":
        if cervix is False:
            add("cervical", "not_needed", "Cervical cancer: not needed after the cervix was removed",
                "The Task Force recommends against screening after a hysterectomy that removed the cervix, if there was no history of a serious cervical lesion.")
        elif age < 21:
            add("cervical", "not_yet", "Cervical cancer: not before 21", "The Task Force recommends against screening before age 21.")
        elif age > 65:
            add("cervical", "not_needed", "Cervical cancer: usually not needed over 65",
                "The Task Force recommends against screening over 65 for women who have had enough normal screening in the past. Ask your doctor whether yours counts.")
        elif cervix is None:
            add("cervical", "need_info", "Cervical cancer", "Advice depends on whether you still have a cervix.", ["whether you still have a cervix"])
            need.append("whether you still have a cervix")
        elif age <= 29:
            add("cervical", "due", "Cervical cancer: a Pap test every 3 years",
                "For ages 21 to 29 the Task Force recommends a Pap test (cervical cytology) every 3 years.")
        else:
            add("cervical", "due", "Cervical cancer: one of three options",
                "For ages 30 to 65 the Task Force recommends one of these: a Pap test every 3 years, an HPV test every 5 years, or both together every 5 years.")
    elif sex is None:
        add("cervical", "need_info", "Cervical cancer", "Advice depends on sex at birth.", ["sex at birth"])
    # ---- colorectal
    if age < 45:
        add("colorectal", "not_yet", "Bowel cancer: not yet",
            "The Task Force recommends starting at age 45 for people at average risk. Ask your doctor sooner if you have bleeding, a family history, or bowel disease.")
    elif age < 50:
        add("colorectal", "due", "Bowel cancer: screening is recommended from 45",
            "For ages 45 to 49 the Task Force recommends screening. Choose one of the tests below.", options=kb["colorectal_tests"])
    elif age <= 75:
        add("colorectal", "due", "Bowel cancer: screening is recommended",
            "For ages 50 to 75 the Task Force strongly recommends screening. Choose one of the tests below.", options=kb["colorectal_tests"])
    elif age <= 85:
        add("colorectal", "individual", "Bowel cancer: your choice with your doctor",
            "For ages 76 to 85 the Task Force says screening should be offered selectively, depending on your overall health, past screening and wishes.")
    else:
        add("colorectal", "not_needed", "Bowel cancer: not recommended over 85", "The Task Force found not enough benefit to recommend screening after 85.")
    # ---- lung
    if age < 50:
        add("lung", "not_yet", "Lung cancer: not before 50", "The Task Force recommends lung screening from 50 for people with a heavy smoking history.")
    elif age > 80:
        add("lung", "not_needed", "Lung cancer: not recommended over 80", "The Task Force recommends lung screening only up to age 80.")
    elif ever is False:
        add("lung", "not_needed", "Lung cancer: not recommended if you have never smoked",
            "Lung screening is recommended only for people with a heavy smoking history, so it is not recommended for you.")
    elif ever is None:
        add("lung", "need_info", "Lung cancer", "Advice depends on your smoking history.", ["whether you have ever smoked"])
        need.append("whether you have ever smoked")
    elif pack_years is not None and pack_years < 20:
        add("lung", "not_needed", "Lung cancer: not recommended under 20 pack-years",
            "The Task Force recommends lung screening for a smoking history of 20 pack-years or more. One pack a day for 20 years is 20 pack-years.")
    elif now is False and quit_years is not None and quit_years > 15:
        add("lung", "not_needed", "Lung cancer: not recommended after 15 years without smoking",
            "The Task Force stops lung screening once you have not smoked for 15 years.")
    else:
        missing = []
        if now is None:
            missing.append("whether you smoke now")
        if now is False and quit_years is None:
            missing.append("how many years ago you quit")
        if pack_years is None:
            missing.append("pack-years (packs a day times years smoked)")
        if missing:
            add("lung", "need_info", "Lung cancer", "Advice depends on your smoking history.", missing)
            need.extend(missing)
        else:
            add("lung", "due", "Lung cancer: a low-dose CT scan every year",
                "You are in the group the Task Force recommends: aged 50 to 80, at least 20 pack-years, and either smoking now or quit in the last 15 years. "
                "Screening stops if your health limits your life expectancy or your ability to have lung surgery.")
    # ---- prostate
    if sex == "male":
        if age < 55:
            add("prostate", "not_yet", "Prostate cancer: no recommendation under 55",
                "The Task Force did not recommend routine PSA testing under 55. Talk to your doctor if you are at higher risk.")
        elif age <= 69:
            add("prostate", "individual", "Prostate cancer: a decision for you and your doctor",
                "For ages 55 to 69 the Task Force says the choice to have a PSA blood test should be yours, after talking about the small possible "
                "benefit and the harms, such as false alarms and unnecessary treatment.")
        else:
            add("prostate", "not_recommended", "Prostate cancer: routine PSA testing is not recommended at 70 and older",
                "The Task Force recommends against routine PSA testing at 70 and older.")
    elif sex is None:
        add("prostate", "need_info", "Prostate cancer", "Advice depends on sex at birth.", ["sex at birth"])

    out.sort(key=lambda r: ORDER.get(r["status"], 9))
    due = [r for r in out if r["status"] == "due"]
    if due:
        headline = f"By US guidelines you may be due for {len(due)} screening{'s' if len(due) != 1 else ''}. See below."
    elif any(r["status"] == "need_info" for r in out):
        headline = "A few more facts are needed to say."
    else:
        headline = "On what you entered, these guidelines do not recommend a routine screening for you right now."
    return {"headline": headline, "items": out, "needs": sorted(set(need)), "family_note": FAMILY_NOTE, "disclaimer": DISCLAIMER,
            "meta": {"review_status": kb["meta"]["review_status"], "checked_against": kb["meta"]["checked_against"],
                     "scope": kb["meta"]["scope"], "not_covered": kb["meta"]["not_covered"]}}
