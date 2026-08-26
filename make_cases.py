"""
Regenerate frontend/src/app/cases.ts, the pool the Generate case button draws from.

Every case is a real record from the training data, expressed in the application's
input schema, with panels the record does not cover left at a normal adult reading.
Nothing is invented.

Selection rules:
  - positives are sampled evenly across each model's probability range rather than
    taking the most confident record, so generated cases land anywhere from the
    fifties to the high nineties instead of always reading 99
  - records the model scores at 100 are excluded, to keep the ceiling at 99
  - negatives are included so some generated cases come back clean
  - every case is scored through the same logic the API uses and is only kept if
    the panel that comes out on top matches the label shown to the user

Run:  python make_cases.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

import train_models as tm

MODEL_DIR = "models"
OUT = os.path.join("frontend", "src", "app", "cases.ts")

# A healthy adult reading for every field, used wherever a record has no value.
NORMAL = {
    "age": 55, "bmi": 24.1, "wbc": 6.4, "rbc": 4.7, "hemoglobin": 14, "platelets": 250,
    "glucose": 88, "calcium": 9.4, "bun": 14, "creatinine": 0.9, "protein_total": 7,
    "albumin": 4.4, "ast": 22, "alt": 20, "bilirubin": 0.6, "alkaline_phosphatase": 78,
    "alpha_fetoprotein_level": 3.1, "psa": 0.9, "plasma_ca19_9": 12,
    "radius_mean": 11.4, "texture_mean": 17.2, "perimeter_mean": 72.6, "area_mean": 400,
    "gender": 0, "smoking": 0, "alcohol_intake": 0.5, "physical_activity": 6,
    "hepatitis_b": 0, "hepatitis_c": 0, "diabetes": 0,
    # Red cell and platelet indices, GGT, and the pelvic-mass tumour markers.
    "hematocrit": 42, "mcv": 89, "mch": 30, "rdw": 13.1, "mpv": 9.8,
    "neutrophil_pct": 58, "ggt": 22, "ca125": 12, "he4": 45, "cea": 1.5,
    # Reproductive and sexual history, at an unremarkable reading.
    "menopause": 0, "sexual_partners": 2, "first_intercourse_age": 19,
    "pregnancies": 1, "smokes": 0, "smoking_years": 0, "smoking_packyears": 0,
    "hormonal_contraceptives": 0, "hormonal_contraceptives_years": 0,
    "iud": 0, "iud_years": 0, "stds": 0, "stds_number": 0, "stds_hpv": 0,
    "stds_diagnoses": 0,
    # Tobacco exposure, inflammation, prostate work-up.
    "cotinine": 0.05, "crp": 1.2, "smoking_packyears": 0,
    "prostate_volume": 28, "psa_density": 0.1, "pi_rads": 2,
}

LABELS = {
    "general": "General", "breast": "Breast", "liver": "Liver",
    "pancreatic": "Pancreatic", "prostate": "Prostate",
    "ovarian": "Ovarian", "cervical": "Cervical", "colorectal": "Bowel",
    "lung": "Lung",
}

SOURCES = {
    "general": "NHANES 2005-2018, US adults",
    "breast": "Wisconsin Diagnostic Breast Cancer",
    "liver": "NHANES 2005-2018, US adults",
    "pancreatic": "Pancreatic biomarker cohort",
    "prostate": "Transperineal biopsy cohort, 212 men",
    "ovarian": "Soochow ovarian mass cohort",
    "cervical": "Caracas colposcopy referral cohort",
    "colorectal": "NHANES 2005-2014, US adults",
    "lung": "NHANES 1999-2018, adults with tobacco exposure",
}

models = {
    f.replace("model_", "").replace(".joblib", ""): joblib.load(os.path.join(MODEL_DIR, f))
    for f in sorted(os.listdir(MODEL_DIR)) if f.endswith(".joblib")
}


def probabilities(cases: list[dict]) -> dict[str, np.ndarray]:
    """Score a batch of full app cases through every model, as the API does."""
    out = {}
    frame = pd.DataFrame(cases)
    for name, b in models.items():
        feats, med = b["feature_names"], b.get("feature_medians", {})
        X = pd.DataFrame({f: frame[f] if f in frame else med.get(f, 0.0) for f in feats})[feats]
        out[name] = b["model"].predict_proba(X)[:, 1] * 100
    return out


def top_panel(per_model: dict[str, float]) -> tuple[str, int]:
    """Reproduce the API's ranking, including the benign complement."""
    best = max(per_model, key=lambda k: per_model[k])
    best_risk = int(round(per_model[best]))
    benign = 100 - best_risk
    if benign > best_risk:
        return "No cancer detected", benign
    return LABELS[best], best_risk


def phrase(v: dict) -> str:
    return "man" if v.get("gender") == 1 else "woman"


def note_for(domain: str, v: dict) -> str:
    age = int(v["age"])
    who = phrase(v)
    if domain == "breast":
        return (f"{age} year old woman. Biopsy imaging shows a nuclear radius of "
                f"{v['radius_mean']:g} and an area of {v['area_mean']:g}. Blood work is normal.")
    if domain == "liver":
        risks = []
        if v.get("hepatitis_b") == 1: risks.append("hepatitis B")
        if v.get("hepatitis_c") == 1: risks.append("hepatitis C")
        if v.get("diabetes") == 1: risks.append("diabetes")
        tail = ", ".join(risks) if risks else "no viral hepatitis or cirrhosis on record"
        return f"{age} year old {who} with {tail}. AFP at {v['alpha_fetoprotein_level']:g} ng/mL."
    if domain == "pancreatic":
        return (f"{age} year old {who}. CA 19-9 at {v['plasma_ca19_9']:g} U/mL, "
                f"bilirubin {v['bilirubin']:g}, glucose {v['glucose']:g}, "
                f"creatinine {v['creatinine']:g}.")
    if domain == "prostate":
        return (f"{age} year old man referred for prostate biopsy. PSA {v['psa']:g} ng/mL, "
                f"prostate volume {v['prostate_volume']:g} mL, PSA density "
                f"{v['psa_density']:g}, PI-RADS {int(v['pi_rads'])} on MRI.")
    if domain == "lung":
        smoke = {0: "never smoked", 1: "former smoker", 2: "current smoker"}[int(v.get("smoking", 0))]
        return (f"{age} year old {who}, {smoke}, {v['smoking_packyears']:g} pack-years. "
                f"Serum cotinine {v['cotinine']:g} ng/mL, CRP {v['crp']:g} mg/L, "
                f"WBC {v['wbc']:g}, haemoglobin {v['hemoglobin']:g}.")
    if domain == "colorectal":
        return (f"{age} year old {who}. Blood count and chemistry only: haemoglobin "
                f"{v['hemoglobin']:g}, platelets {v['platelets']:g}, WBC {v['wbc']:g}, "
                f"albumin {v['albumin']:g}. No bowel symptoms recorded.")
    if domain == "ovarian":
        when = "post-menopausal" if v.get("menopause") == 1 else "pre-menopausal"
        return (f"{age} year old {when} woman with an ovarian mass already found on "
                f"imaging. CA 125 at {v['ca125']:g} U/mL, HE4 at {v['he4']:g} pmol/L, "
                f"CEA {v['cea']:g}, platelets {v['platelets']:g}, albumin {v['albumin']:g}.")
    if domain == "cervical":
        bits = []
        if v.get("stds_hpv") == 1: bits.append("HPV on record")
        if v.get("smokes") == 1: bits.append(f"smokes, {v['smoking_packyears']:g} pack-years")
        if float(v.get("hormonal_contraceptives_years", 0)) > 0:
            bits.append(f"{v['hormonal_contraceptives_years']:g} years on hormonal contraceptives")
        tail = ", ".join(bits) if bits else "no HPV recorded and no tobacco use"
        return (f"{age} year old woman being assessed for colposcopy. "
                f"{int(v['sexual_partners'])} lifetime partners, first intercourse at "
                f"{int(v['first_intercourse_age'])}, {int(v['pregnancies'])} pregnancies. {tail}.")
    smoke = {0: "never smoked", 1: "former smoker", 2: "current smoker"}[int(v.get("smoking", 0))]
    drink = float(v.get("alcohol_intake", 0))
    return (f"{age} year old {who}, {smoke}. BMI {v['bmi']:g}, "
            f"{v['physical_activity']:g} hours of exercise a week, "
            f"alcohol {drink:g} of 5.")


def build(domain: str, config: dict):
    X, y, _ = tm.prepare(config)

    cases = [{**NORMAL, **{k: float(v) for k, v in row.items()}} for _, row in X.iterrows()]

    # Sex-specific panels do not carry gender as a model feature, because within
    # those cohorts it is constant and therefore useless to the model. That left
    # generated prostate cases inheriting the default of female, so a case
    # narrated as "62 year old man" arrived at the API without a sex and picked
    # up an ovarian score. Set it explicitly here so the sample cases exercise
    # the same anatomy gate a real user would.
    forced_sex = {"prostate": 1.0, "ovarian": 0.0, "cervical": 0.0}.get(domain)
    if forced_sex is not None:
        for c in cases:
            c["gender"] = forced_sex
    per_model = probabilities(cases)
    own = per_model[domain]

    kept = []

    # Positives: spread across the 50 to 99 band, one per target level.
    pos_idx = [i for i in range(len(y)) if y.iloc[i] == 1 and 50 <= own[i] < 99.5]
    for target in [52, 65, 78, 86, 92, 96, 99]:
        if not pos_idx:
            break
        i = min(pos_idx, key=lambda j: abs(own[j] - target))
        pos_idx.remove(i)
        kept.append((i, True))

    # Negatives: records the model is confident are clean.
    neg_idx = sorted([i for i in range(len(y)) if y.iloc[i] == 0], key=lambda j: own[j])[:40]
    for i in neg_idx[:: max(1, len(neg_idx) // 3)][:3]:
        kept.append((i, False))

    out = []
    for i, positive in kept:
        values = cases[i]
        label, risk = top_panel({m: per_model[m][i] for m in per_model})
        expected = LABELS[domain] if positive else "No cancer detected"

        # Only keep a case whose banner promise matches what the models return.
        if label != expected:
            print(f"    dropped {domain} row {i}: promised {expected}, models return {label} at {risk}%")
            continue

        delta = {k: round(float(v), 4) for k, v in values.items()
                 if abs(float(v) - float(NORMAL[k])) > 1e-9}
        out.append({
            "domain": LABELS[domain],
            "expect": expected,
            "positive": positive,
            "source": f"{SOURCES[domain]}, {'positive' if positive else 'negative'} record",
            "note": note_for(domain, values),
            "delta": delta,
            "_risk": risk,
        })
    return out


def num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(round(float(v), 4))


def main():
    pool = []
    for config in tm.DATASETS:
        domain = config["name"]
        if domain in tm.WITHDRAWN or domain not in models:
            print(f"  {domain}: skipped, panel is not shipped")
            continue
        print(f"  {domain}")
        pool.extend(build(domain, config))

    for n, c in enumerate(pool):
        c["id"] = f"{c['domain'].lower()}{'' if c['positive'] else '-neg'}-{n}"

    lines = [
        "// Sample cases for the Generate case button.",
        "//",
        "// Every entry is a real record from the training data, expressed in the",
        "// application's input schema, with panels the record does not cover left at a",
        "// normal adult reading. Nothing here is invented.",
        "//",
        "// Positives are sampled evenly across each model's probability range rather than",
        "// taking the most confident record, so a generated case lands anywhere from the",
        "// fifties to the high nineties instead of always reading 99. Negative records are",
        "// included too, so some generated cases come back clean. Every case was scored",
        "// before being written here, and only kept if the panel that comes out on top",
        "// matches the expectation shown to the user.",
        "//",
        "// Generated by make_cases.py. Regenerate if the models are retrained.",
        "",
        "export type CaseValues = Record<string, number>;",
        "",
        "export const NORMAL: CaseValues = {",
    ]
    for k, v in NORMAL.items():
        lines.append(f"  {k}: {num(v)},")
    lines += [
        "};",
        "",
        "export type DemoCase = {",
        "  id: string;",
        "  domain: string;",
        "  expect: string;",
        "  positive: boolean;",
        "  source: string;",
        "  note: string;",
        "  delta: CaseValues;",
        "};",
        "",
        "export const CASE_POOL: DemoCase[] = [",
    ]

    for c in pool:
        delta = ", ".join(f"{k}: {num(v)}" for k, v in c["delta"].items())
        lines += [
            "  {",
            f'    id: "{c["id"]}",',
            f'    domain: "{c["domain"]}",',
            f'    expect: "{c["expect"]}",',
            f'    positive: {"true" if c["positive"] else "false"},',
            f'    source: "{c["source"]}",',
            f'    note: "{c["note"]}",',
            f"    delta: {{ {delta} }},",
            "  },",
        ]

    # Open on a clear breast case, but one nearer the middle of the range than
    # the ceiling, so the first thing anyone sees is not a pinned 99.
    opening = min(
        (c for c in pool if c["domain"] == "Breast" and c["positive"]),
        key=lambda c: abs(c["_risk"] - 93),
    )

    lines += [
        "];",
        "",
        "export const caseValues = (c: DemoCase): CaseValues => ({ ...NORMAL, ...c.delta });",
        "",
        "/** Draw a random case, never the same one twice in a row. */",
        "export function randomCase(previousId?: string): DemoCase {",
        "  const options = CASE_POOL.filter(c => c.id !== previousId);",
        "  return options[Math.floor(Math.random() * options.length)];",
        "}",
        "",
        "/**",
        " * The case the page opens on. Fixed rather than random so the first render",
        " * matches between server and client. Every press of Generate case after that",
        " * draws at random.",
        " */",
        f'export const OPENING_CASE = CASE_POOL.find(c => c.id === "{opening["id"]}")!;',
        "",
    ]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    risks = sorted(c["_risk"] for c in pool if c["positive"])
    print(f"\nwrote {OUT}")
    print(f"{len(pool)} cases, {sum(c['positive'] for c in pool)} positive, "
          f"{sum(not c['positive'] for c in pool)} negative")
    print(f"positive score range {risks[0]} to {risks[-1]}, median {risks[len(risks)//2]}")
    print(f"opening case {opening['id']} at {opening['_risk']}%")


if __name__ == "__main__":
    main()
