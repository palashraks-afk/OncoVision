"""
How readable is the text the navigator shows a person?

The original aim was results "readable to someone without a medical background". That
should be a number and not a feeling. This scores every sentence the tool can show with
the Flesch-Kincaid grade level, which estimates the US school grade needed to read it.

Health-literacy guidance commonly aims for about grade 6 to 8 for material meant for the
general public, and the average US adult reads at roughly grade 8. The target here is
set before looking: every patient-facing action line at or below grade 10, and the
average at or below grade 8. Medical words that cannot be avoided (endoscopy, ultrasound)
push a score up, so the longest offenders are listed rather than hidden.

What this does not measure: comprehension. A short sentence can still be misunderstood.
Reading level is a floor for the tool, and a real test needs real readers.

Run:  python experiments/navigator_readability.py
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import navigator as nav  # noqa: E402

OUT = "experiments/navigator_readability_result.json"
TARGET_EACH, TARGET_MEAN = 10.0, 8.0


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def fk_grade(text: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if re.search(r"[A-Za-z]", s)]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not sentences or not words:
        return 0.0
    syl = sum(syllables(w) for w in words)
    return 0.39 * len(words) / len(sentences) + 11.8 * syl / len(words) - 15.59


def main():
    kb = nav.load()
    items = []
    for r in kb["rules"]:
        items.append(("action", r["id"], r["action"]))
    items.append(("action", kb["melanoma_checklist"]["id"], kb["melanoma_checklist"]["action"]))
    for s in kb["symptoms"]:
        items.append(("symptom", s["key"], s["label"] + ". " + s["hint"] if s["hint"] else s["label"] + "."))
    for i, s in enumerate(nav.SAFETY):
        items.append(("safety", f"safety{i}", s))
    items.append(("disclaimer", "disclaimer", nav.DISCLAIMER))
    for k, v in {
        "talk_soon": "This combination meets a guideline threshold for a prompt check. Most people who meet it do not have cancer, but it is worth seeing a doctor soon.",
        "nothing_meets": "No guideline threshold is met by what you entered. That is not the same as nothing being wrong.",
    }.items():
        items.append(("headline", k, v))

    scored = [(kind, name, text, fk_grade(text)) for kind, name, text in items]
    by_kind = {}
    for kind, _, _, g in scored:
        by_kind.setdefault(kind, []).append(g)

    result = {"target_each": TARGET_EACH, "target_mean": TARGET_MEAN, "kinds": {}}
    print(f"{'kind':<12}{'n':>4}{'mean grade':>12}{'worst':>8}{'over '+str(int(TARGET_EACH)):>9}")
    for kind, gs in by_kind.items():
        over = sum(1 for g in gs if g > TARGET_EACH)
        result["kinds"][kind] = {"n": len(gs), "mean": round(sum(gs) / len(gs), 1),
                                 "worst": round(max(gs), 1), "over_target": over}
        print(f"{kind:<12}{len(gs):>4}{sum(gs) / len(gs):>12.1f}{max(gs):>8.1f}{over:>9}")

    actions = [g for kind, _, _, g in scored if kind in ("action", "headline")]
    mean_actions = sum(actions) / len(actions)
    over_actions = [(n, round(g, 1)) for kind, n, _, g in scored if kind == "action" and g > TARGET_EACH]
    result["patient_facing_action_mean"] = round(mean_actions, 1)
    result["actions_over_target"] = over_actions
    result["meets_target"] = bool(mean_actions <= TARGET_MEAN and not over_actions)
    print(f"\nPatient-facing action and headline lines: mean grade {mean_actions:.1f} "
          f"(target {TARGET_MEAN:.0f}), {len(over_actions)} over {TARGET_EACH:.0f}")
    # One-word symptom labels ("Indigestion.") score absurdly high because the formula
    # divides by a one-word sentence, so they are left out of the list of hardest text.
    worst = sorted(((g, kind, n, t) for kind, n, t, g in scored if kind != "symptom"), reverse=True)[:6]
    print("\nHardest to read:")
    for g, kind, n, t in worst:
        print(f"  grade {g:>4.1f}  {kind}/{n}: {t[:110]}{'...' if len(t) > 110 else ''}")
    print(f"\n  VERDICT: " + ("meets the target." if result["meets_target"] else "does not meet the target."))
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
