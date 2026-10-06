"""
The symptom-and-lab pattern navigator.

What this is
------------
A rule engine that reads what a person has noticed (symptoms, how long, how often),
a few facts about them, and the lab values on a report they already hold, and says
whether the COMBINATION meets a published guideline threshold for a prompt check.

It is deliberately not a model. Every rule it applies is in navigator_rules.json,
transcribed from the NICE suspected-cancer guideline (NG12) with its source page, so
that a clinician can read, correct and sign off the rules without touching code, and
so that no medical content exists in this file that was not in the guideline. The
engine's job is only to apply them faithfully and to be honest about what it does
not know.

What it never does
------------------
It never says a person has cancer, never tells anyone they are fine, and never
produces clinical wording of its own: the action text and the criteria shown to the
user come from the rule file. A pattern that meets no threshold is reported as "no
guideline threshold met", which is a different statement from "nothing is wrong".

Three-valued logic
------------------
Every condition evaluates to True, False or None (unknown). A rule that could apply if
one more fact were known is reported as "could apply if", with exactly what is needed,
and is NOT treated as a match. This is how a missing smoking history or an unentered
haemoglobin is handled: asked for, never assumed.

Status: research prototype. The rules are the 2015 edition of UK guidance and have not
been reviewed by a clinician. See meta in navigator_rules.json.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navigator_rules.json")

_KB: Optional[dict] = None

# Display order. A person is shown the most actionable thing first.
TIER_ORDER = {"talk_soon": 0, "worth_raising": 1, "mention": 2}
TIMEFRAME_ORDER = {"immediate": 0, "48h": 1, "2w": 2, "primary_care": 3, "routine": 4}
VALID_TIMEFRAMES = set(TIMEFRAME_ORDER)
VALID_STRENGTHS = {"recommended", "consider"}

SAFETY = [
    "Seek urgent care now if you cough up or vomit a lot of blood. Do the same for severe chest "
    "pain, trouble breathing, or black or bloody stools with faintness. Also seek urgent care if a "
    "symptom is getting worse quickly. Do not wait for an appointment.",
    "Meeting no threshold here does not mean nothing is wrong. If a symptom lasts, returns, or "
    "worries you, see a doctor whatever this says.",
]

DISCLAIMER = (
    "This is a research prototype, not medical advice and not a diagnosis. It applies published "
    "UK guideline thresholds (the 2015 edition) that a clinician has not yet reviewed for this "
    "tool, and it cannot examine you. Most people who meet one of these patterns do not have "
    "cancer. Bring the summary to a doctor and let them decide."
)


def load() -> dict:
    global _KB
    if _KB is None:
        with open(RULES_PATH, encoding="utf-8") as f:
            _KB = json.load(f)
        _KB["_sym_labels"] = {s["key"]: s["label"] for s in _KB["symptoms"]}
        _KB["_finding_labels"] = {f_["key"]: f_["label"] for f_ in _KB["findings"]}
    return _KB


CONTEXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navigator_context.json")
_CTX: Optional[dict] = None


def context_for(age: Optional[float], sex: Optional[str], ever_smoked: Optional[bool]) -> Optional[dict]:
    """How common cancer death is, for people of this age, sex and smoking history.

    A population base rate from a US survey, validated out of era (see
    experiments/navigator_evidence.py question 6). It does not use the person's labs or
    symptoms and is not their risk. It exists so that an alert can be read against what
    is ordinary for their age, since age explains most of the difference between people.
    Returns None when age or sex is missing or outside the range it was fitted on.
    """
    global _CTX
    if age is None or sex not in ("male", "female") or not os.path.exists(CONTEXT_PATH):
        return None
    if _CTX is None:
        with open(CONTEXT_PATH, encoding="utf-8") as f:
            _CTX = json.load(f)
    lo, hi = _CTX["age_range"]
    if not (lo <= age <= hi):
        return None
    c = _CTX["coefficients"]
    smoked = _CTX["mean_ever_smoked"] if ever_smoked is None else float(bool(ever_smoked))
    z = (c["intercept"] + c["age_minus_60"] * (age - 60) + c["age_minus_60_sq_over_20"] * (age - 60) ** 2 / 20
         + c["male"] * (sex == "male") + c["ever_smoked"] * smoked)
    per_1000 = 1000 / (1 + math.exp(-z))
    shown = round(per_1000) if per_1000 >= 10 else round(per_1000, 1)
    who = "adults" if ever_smoked is None else ("adults who have smoked" if ever_smoked else "adults who have never smoked")
    return {
        "per_1000": shown,
        "text": (f"For context: in a US health survey, about {shown:g} in every 1,000 {sex} {who} aged "
                 f"around {int(round(age))} died of cancer within five years. This is a group average for your age "
                 "and not your own risk. Age explains most of the difference between people."),
        "smoking_assumed_average": ever_smoked is None,
        "basis": "NHANES 1999-2014, 24,180 adults 40 and over, deaths from the National Death Index. Checked on a later "
                 "survey period (calibration slope %.2f). Does not use your labs or symptoms." % _CTX["validation_out_of_era"]["calibration_slope"],
    }


def vocabulary() -> dict:
    """What the form can ask for, so the interface never drifts from the rules."""
    kb = load()
    # A symptom needs a frequency question only if some rule asks how often it happens.
    # Derived from the rules so the form cannot drift from them.
    freq_keys = {k for r in kb["rules"] for path in r["paths"] for c in path
                 if "freq_min" in c for k in c["freq_min"]["of"]}
    groups: Dict[str, list] = {}
    for s in kb["symptoms"]:
        groups.setdefault(s["group"], []).append(
            {"key": s["key"], "label": s["label"], "hint": s.get("hint", ""),
             "asks_frequency": s["key"] in freq_keys})
    return {
        "symptom_groups": [{"group": g, "symptoms": v} for g, v in groups.items()],
        "findings": kb["findings"],
        "melanoma_features": kb["melanoma_checklist"]["features"],
        "labs": [
            {"key": "hemoglobin", "label": "Haemoglobin", "unit": "g/dL"},
            {"key": "mcv", "label": "MCV (red cell size)", "unit": "fL"},
            {"key": "ferritin", "label": "Ferritin (iron stores)", "unit": "ng/mL"},
            {"key": "platelets", "label": "Platelets", "unit": "K/uL"},
            {"key": "wbc", "label": "White blood cells", "unit": "K/uL"},
            {"key": "ca125", "label": "CA-125", "unit": "U/mL"},
        ],
        "meta": {k: kb["meta"][k] for k in ("guideline_version", "review_status", "known_limitations",
                                            "source_document")},
    }


# ------------------------------------------------------------------ labs
def derive_lab_flags(labs: Dict[str, Optional[float]], sex: Optional[str]) -> Dict[str, Any]:
    """The named findings the rules ask about, from numeric results.

    Each is True, False or None. None means "cannot tell from what was given".
    Notes record every assumption so it can be shown to the user.
    """
    t = load()["lab_thresholds"]
    notes: List[str] = []
    flags: Dict[str, Optional[bool]] = {}
    hb, mcv, fer = labs.get("hemoglobin"), labs.get("mcv"), labs.get("ferritin")

    if hb is None or sex not in ("male", "female"):
        flags["anaemia"] = None
    else:
        flags["anaemia"] = hb < t["anaemia_hemoglobin_g_dl"][sex]

    if fer is not None:
        flags["iron_deficiency"] = fer < t["iron_deficiency_ferritin_ng_ml"]
    elif mcv is not None and mcv < t["microcytic_mcv_fl"]:
        flags["iron_deficiency"] = True
        notes.append("Iron deficiency is assumed from small red cells (MCV under "
                     f"{t['microcytic_mcv_fl']:.0f}) because no ferritin was given. A ferritin result "
                     "would confirm it.")
    else:
        flags["iron_deficiency"] = None

    if flags["anaemia"] is False:
        flags["iron_deficiency_anaemia"] = False
    elif flags["anaemia"] is True and flags["iron_deficiency"] is not None:
        flags["iron_deficiency_anaemia"] = flags["iron_deficiency"]
    else:
        flags["iron_deficiency_anaemia"] = None

    p, w, ca = labs.get("platelets"), labs.get("wbc"), labs.get("ca125")
    flags["thrombocytosis"] = None if p is None else p > t["thrombocytosis_platelets_k_ul"]
    flags["raised_wbc"] = None if w is None else w > t["raised_white_count_k_ul"]
    flags["ca125_high"] = None if ca is None else ca >= t["ca125_u_ml"]
    return {"flags": flags, "notes": notes}


LAB_FLAG_NEEDS = {
    "anaemia": "your haemoglobin (and your sex)",
    "iron_deficiency": "your ferritin (or your MCV)",
    "iron_deficiency_anaemia": "your haemoglobin and ferritin",
    "thrombocytosis": "your platelet count",
    "raised_wbc": "your white blood cell count",
    "ca125_high": "your CA-125 result",
}
LAB_FLAG_TEXT = {
    "anaemia": "low haemoglobin (anaemia)",
    "iron_deficiency": "low iron stores",
    "iron_deficiency_anaemia": "iron-deficiency anaemia",
    "thrombocytosis": "a high platelet count",
    "raised_wbc": "a raised white blood cell count",
    "ca125_high": "CA-125 of 35 or above",
}


# ------------------------------------------------------------ conditions
class Ctx:
    def __init__(self, age, sex, ever_smoked, asbestos, symptoms, findings, flags):
        self.age, self.sex = age, sex
        self.ever_smoked, self.asbestos = ever_smoked, asbestos
        self.symptoms = symptoms          # key -> {"times_per_month": float|None}
        self.findings = findings          # set of keys
        self.flags = flags


def eval_cond(c: dict, x: Ctx) -> Optional[bool]:
    """True, False, or None when the answer depends on something not provided."""
    if "age_min" in c:
        return None if x.age is None else x.age >= c["age_min"]
    if "age_lt" in c:
        return None if x.age is None else x.age < c["age_lt"]
    if "sex" in c:
        return None if x.sex is None else x.sex == c["sex"]
    if "any_symptom" in c:
        return len(x.symptoms) > 0
    if "sym" in c:
        return c["sym"] in x.symptoms
    if "not_sym" in c:
        return c["not_sym"] not in x.symptoms
    if "sym_min" in c:
        n = sum(1 for s in c["sym_min"]["of"] if s in x.symptoms)
        return n >= c["sym_min"]["n"]
    if "freq_min" in c:
        present = [s for s in c["freq_min"]["of"] if s in x.symptoms]
        if not present:
            return False
        freqs = [x.symptoms[s].get("times_per_month") for s in present]
        if any(f is not None and f >= c["freq_min"]["per_month"] for f in freqs):
            return True
        return None if any(f is None for f in freqs) else False
    if "ever_smoked" in c:
        return None if x.ever_smoked is None else x.ever_smoked == c["ever_smoked"]
    if "asbestos" in c:
        return None if x.asbestos is None else x.asbestos == c["asbestos"]
    if "lab" in c:
        return x.flags.get(c["lab"])
    if "finding" in c:
        return c["finding"] in x.findings
    raise ValueError(f"unknown condition: {c}")


def eval_path(path: List[dict], x: Ctx):
    states = [eval_cond(c, x) for c in path]
    if any(s is False for s in states):
        return False, []
    unknown = [c for c, s in zip(path, states) if s is None]
    return (None if unknown else True), unknown


def describe(c: dict) -> str:
    kb = load()
    lab = kb["_sym_labels"]
    if "age_min" in c:
        return f"aged {c['age_min']} or over"
    if "age_lt" in c:
        return f"under {c['age_lt']}"
    if "sex" in c:
        return "female" if c["sex"] == "female" else "male"
    if "any_symptom" in c:
        return "at least one symptom with no known cause"
    if "sym" in c:
        return lab[c["sym"]].lower()
    if "not_sym" in c:
        return "no " + lab[c["not_sym"]].lower()
    if "sym_min" in c:
        names = "; ".join(lab[s].lower() for s in c["sym_min"]["of"])
        n = c["sym_min"]["n"]
        return f"{'any one' if n == 1 else str(n) + ' or more'} of: {names}"
    if "freq_min" in c:
        names = "; ".join(lab[s].lower() for s in c["freq_min"]["of"])
        return f"any of these on {c['freq_min']['per_month']} or more days a month: {names}"
    if "ever_smoked" in c:
        return "have ever smoked" if c["ever_smoked"] else "have never smoked"
    if "asbestos" in c:
        return "exposed to asbestos"
    if "lab" in c:
        return LAB_FLAG_TEXT.get(c["lab"], c["lab"])
    if "finding" in c:
        return kb["_finding_labels"][c["finding"]].lower()
    return str(c)


def need_text(c: dict, x: Optional[Ctx] = None) -> str:
    if "age_min" in c or "age_lt" in c:
        return "your age"
    if "sex" in c:
        return "your sex"
    if "ever_smoked" in c:
        return "whether you have ever smoked"
    if "asbestos" in c:
        return "whether you have been exposed to asbestos"
    if "lab" in c:
        return LAB_FLAG_NEEDS.get(c["lab"], c["lab"])
    if "freq_min" in c:
        # Name only what the person actually ticked, not every symptom the rule could use.
        asked = [k for k in c["freq_min"]["of"]
                 if x is None or (k in x.symptoms and x.symptoms[k].get("times_per_month") is None)]
        names = " or ".join(load()["_sym_labels"][k].lower() for k in asked)
        return f"how many days a month you have: {names}"
    return describe(c)


def tier_of(rule: dict) -> str:
    tf, st = rule["timeframe"], rule["strength"]
    if st == "recommended" and tf in ("immediate", "48h", "2w"):
        return "talk_soon"
    if tf == "routine":
        return "mention"
    return "worth_raising"


# ---------------------------------------------------------------- inputs
def _clean_symptoms(items: Optional[list], known: dict, ignored: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for it in items or []:
        key = it.get("key") if isinstance(it, dict) else it
        if key not in known:
            ignored.append(str(key))
            continue
        if isinstance(it, dict) and it.get("explained"):
            continue                       # a known cause: the guideline says "unexplained"
        tpm = it.get("times_per_month") if isinstance(it, dict) else None
        if tpm is not None:
            tpm = float(tpm)
            if tpm < 0 or tpm > 31:
                raise ValueError("times_per_month must be between 0 and 31")
        out[key] = {"times_per_month": tpm}
    return out


def _bool_or_none(v):
    return None if v is None else bool(v)


def _num(labs: dict, key: str, lo: float, hi: float) -> Optional[float]:
    v = labs.get(key)
    if v is None or v == "":
        return None
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"{key} of {v} is outside the range a living person can have")
    return v


# -------------------------------------------------------------- evaluate
def evaluate(request: dict) -> dict:
    """Apply the rule file to one person. Raises ValueError on impossible input."""
    kb = load()
    age = request.get("age")
    if age is not None:
        age = float(age)
        if not (0 <= age <= 120):
            raise ValueError("age must be between 0 and 120")
    sex = request.get("sex")
    if sex not in (None, "male", "female"):
        sex = None

    ignored: List[str] = []
    symptoms = _clean_symptoms(request.get("symptoms"), kb["_sym_labels"], ignored)
    findings = {f for f in (request.get("findings") or []) if f in kb["_finding_labels"]}
    labs_in = request.get("labs") or {}
    labs = {
        "hemoglobin": _num(labs_in, "hemoglobin", 3, 25),
        "mcv": _num(labs_in, "mcv", 40, 140),
        "ferritin": _num(labs_in, "ferritin", 0, 5000),
        "platelets": _num(labs_in, "platelets", 5, 2000),
        "wbc": _num(labs_in, "wbc", 0.1, 500),
        "ca125": _num(labs_in, "ca125", 0, 100000),
    }
    derived = derive_lab_flags(labs, sex)
    x = Ctx(age, sex, _bool_or_none(request.get("ever_smoked")),
            _bool_or_none(request.get("asbestos")), symptoms, findings, derived["flags"])
    labs_provided = any(v is not None for v in labs.values())

    matches, could_apply = [], []
    for rule in kb["rules"]:
        hit, hit_path, pending = None, None, []
        for path in rule["paths"]:
            state, unknown = eval_path(path, x)
            if state is True:
                hit, hit_path = True, path
                break
            if state is None:
                pending.append((path, unknown))
        if hit:
            matches.append({
                "id": rule["id"], "site": rule["site"], "tier": tier_of(rule),
                "timeframe": rule["timeframe"], "strength": rule["strength"],
                "action": rule["action"], "source": rule["source"],
                "because": [describe(c) for c in hit_path],
            })
        elif pending:
            # Only worth asking about when something the person told us anchors the rule,
            # so a healthy adult is not nagged for every lab the guideline mentions.
            for path, unknown in pending:
                anchored = any(("sym" in c or "sym_min" in c or "freq_min" in c or "any_symptom" in c)
                               for c in path)
                lab_only_noise = (not anchored) and (not labs_provided)
                if anchored or (not lab_only_noise):
                    could_apply.append({
                        "id": rule["id"], "site": rule["site"], "tier": tier_of(rule),
                        "needs": sorted({need_text(c, x) for c in unknown}),
                        "if_so": rule["action"]})
                    break

    # Melanoma: a checklist the person scores, shown only if they described a lesion.
    mel = kb["melanoma_checklist"]
    lesion = request.get("skin_lesion")
    mel_score = None
    if lesion:
        mel_score = sum(f["points"] for f in mel["features"] if lesion.get(f["key"]))
        if mel_score >= mel["threshold"]:
            matches.append({
                "id": mel["id"], "site": mel["site"], "tier": tier_of(mel),
                "timeframe": mel["timeframe"], "strength": mel["strength"],
                "action": mel["action"], "source": mel["source"],
                "because": [f"a self-scored checklist total of {mel_score} (3 or more is the threshold)"]})

    matches.sort(key=lambda m: (TIER_ORDER[m["tier"]], TIMEFRAME_ORDER[m["timeframe"]], m["site"]))
    # For one site, a lower-tier match adds nothing once a higher-tier one is shown (an urgent
    # bowel referral already covers the lower-tier stool test).
    best = {}
    for m in matches:
        best.setdefault(m["site"], TIER_ORDER[m["tier"]])
    matches = [m for m in matches if TIER_ORDER[m["tier"]] == best[m["site"]]]
    # One could-apply entry per site, and none for a site already matched at the same tier.
    seen = {(m["site"], m["tier"]) for m in matches}
    uniq, taken = [], set()
    for c in could_apply:
        k = (c["site"], tuple(c["needs"]))
        if (c["site"], c["tier"]) in seen or k in taken:
            continue
        taken.add(k)
        uniq.append(c)
    could_apply = sorted(uniq, key=lambda c: (TIER_ORDER[c["tier"]], c["site"]))

    if matches:
        state = matches[0]["tier"]
    elif could_apply:
        state = "need_info"
    else:
        state = "nothing_meets"

    lab_notes = list(derived["notes"])
    if matches and not symptoms and labs_provided:
        lab_notes.append(
            "This came from lab results alone, with no symptoms entered. Common causes are not cancer, such as low iron "
            "from diet, kidney disease or heart disease. In a US survey, these lab findings did not help tell who would "
            "later die of cancer once age and sex were known. Take it to a doctor as a question, not as a warning.")

    if (sex == "female" and age is not None and age < 50 and derived["flags"].get("iron_deficiency_anaemia")
            and any(m["site"] == "bowel cancer" for m in matches)):
        lab_notes.append(
            "Low iron is very common in women who still have periods. The guideline still says to look for other causes, so "
            "mention it to a doctor, who can weigh up your periods and your diet.")

    guideline_notes = []
    if any(m["site"] == "bowel cancer" for m in matches):
        guideline_notes.append(
            "These bowel rules are from the 2015 edition of the guideline. NICE changed its bowel cancer advice in 2023 to use a "
            "stool test called FIT as an early step. Your doctor will follow the current version.")

    headline = {
        "talk_soon": "This combination meets a guideline threshold for a prompt check. Most people who meet it do not have cancer, but it is worth seeing a doctor soon.",
        "worth_raising": "Nothing here is urgent by the guideline, but some of it is worth raising with a doctor.",
        "mention": "Nothing here is urgent. It is worth mentioning at a routine visit.",
        "need_info": "No threshold is met on what you entered, but a few more facts could change that.",
        "nothing_meets": "No guideline threshold is met by what you entered. That is not the same as nothing being wrong.",
    }[state]

    return {
        "state": state,
        "headline": headline,
        "matches": matches,
        "could_apply_if": could_apply,
        "melanoma_score": mel_score,
        "lab_notes": lab_notes,
        "guideline_notes": guideline_notes,
        "lab_findings": {k: v for k, v in derived["flags"].items() if v is not None},
        "ignored_symptoms": ignored,
        "context": context_for(age, sex, x.ever_smoked),
        "safety": SAFETY,
        "disclaimer": DISCLAIMER,
        "rules": {"guideline_version": kb["meta"]["guideline_version"],
                  "review_status": kb["meta"]["review_status"],
                  "n_rules": len(kb["rules"]) + 1},
    }


# ---------------------------------------------------------------- audit
SUPPORTED = {"age_min", "age_lt", "sex", "any_symptom", "sym", "not_sym", "sym_min", "freq_min",
             "ever_smoked", "asbestos", "lab", "finding"}


def validate_knowledge_base() -> List[str]:
    """Structural problems in the rule file. Empty means it is internally consistent.

    A rule that refers to a symptom the form cannot ask about can never fire, which
    is a silent failure; this exists to make it a loud one.
    """
    kb = load()
    problems: List[str] = []
    syms, finds = set(kb["_sym_labels"]), set(kb["_finding_labels"])
    ids = set()
    flags = set(LAB_FLAG_TEXT)
    for r in kb["rules"]:
        rid = r.get("id", "?")
        if rid in ids:
            problems.append(f"{rid}: duplicate id")
        ids.add(rid)
        for field in ("site", "timeframe", "strength", "paths", "action", "source"):
            if not r.get(field):
                problems.append(f"{rid}: missing {field}")
        if r.get("timeframe") not in VALID_TIMEFRAMES:
            problems.append(f"{rid}: bad timeframe {r.get('timeframe')}")
        if r.get("strength") not in VALID_STRENGTHS:
            problems.append(f"{rid}: bad strength {r.get('strength')}")
        for path in r.get("paths", []):
            for c in path:
                keys = set(c)
                if not keys & SUPPORTED:
                    problems.append(f"{rid}: unsupported condition {c}")
                if "sym" in c and c["sym"] not in syms:
                    problems.append(f"{rid}: unknown symptom {c['sym']}")
                if "not_sym" in c and c["not_sym"] not in syms:
                    problems.append(f"{rid}: unknown symptom {c['not_sym']}")
                for k in ("sym_min", "freq_min"):
                    if k in c:
                        for s in c[k]["of"]:
                            if s not in syms:
                                problems.append(f"{rid}: unknown symptom {s}")
                if "lab" in c and c["lab"] not in flags:
                    problems.append(f"{rid}: unknown lab flag {c['lab']}")
                if "finding" in c and c["finding"] not in finds:
                    problems.append(f"{rid}: unknown finding {c['finding']}")
    return problems
