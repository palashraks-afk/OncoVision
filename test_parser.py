"""
Field-level accuracy test for the PDF parser.

Why this matters more than any model metric: every panel sits downstream of this
parser. If it misreads albumin, the liver model is scored on a number the patient
does not have, and no amount of calibration upstream fixes that. The parser has
never been measured, and it needs no dataset, no cohort and no ethics approval to
measure, which made it the largest cheap gap in the project.

The test builds lab report PDFs in five deliberately different layouts, because
the failure mode that matters is layout sensitivity rather than average accuracy.
Ground truth is known exactly, since the PDFs are generated from it.

  1. two-column table with a reference range column, the common commercial layout
  2. narrow single column, no rules, values on the line below the analyte
  3. wide table where the reference range sits immediately left of the result,
     which is the layout most likely to make a regex grab the wrong number
  4. abbreviation-only report using assay codes rather than analyte names
  5. a report carrying flag letters and out-of-range markers in the value column

Reported per field and per layout, along with the specific failures, because an
aggregate number would hide exactly the thing this is looking for.

Run:  python test_parser.py          (needs the API running on :8000)
      python test_parser.py --build  (only write the PDFs)
"""

import io
import json
import os
import sys
import urllib.request

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

OUT_DIR = os.path.join("tests", "lab_reports")
API = os.getenv("ONCOVISION_API", "http://127.0.0.1:8000")

# The truth. Every PDF is rendered from exactly these values, so any difference
# the parser reports is a parser error and not a transcription error.
TRUTH = {
    "age": 58,
    "wbc": 7.2,
    "rbc": 4.55,
    "hemoglobin": 13.6,
    "platelets": 233,
    "glucose": 104,
    "calcium": 9.7,
    "bun": 17,
    "creatinine": 1.05,
    "protein_total": 7.1,
    "albumin": 4.2,
    "ast": 31,
    "alt": 27,
    "bilirubin": 0.8,
    "alkaline_phosphatase": 92,
    "psa": 2.4,
    "alpha_fetoprotein_level": 5.1,
    "plasma_ca19_9": 22.0,
}

# How each analyte is printed, per layout style: full name, abbreviation, units
# and a plausible reference range string.
ANALYTES = [
    # key,                      full name,                 abbrev,  units,   ref range
    ("wbc",                     "White Blood Cell Count",  "WBC",   "K/uL",  "4.0 - 11.0"),
    ("rbc",                     "Red Blood Cell Count",    "RBC",   "M/uL",  "4.20 - 5.80"),
    ("hemoglobin",              "Hemoglobin",              "HGB",   "g/dL",  "13.0 - 17.0"),
    ("platelets",               "Platelet Count",          "PLT",   "K/uL",  "150 - 400"),
    ("glucose",                 "Glucose, Fasting",        "GLU",   "mg/dL", "70 - 99"),
    ("calcium",                 "Calcium, Total",          "CA",    "mg/dL", "8.6 - 10.3"),
    ("bun",                     "Urea Nitrogen",           "BUN",   "mg/dL", "7 - 20"),
    ("creatinine",              "Creatinine",              "CREAT", "mg/dL", "0.70 - 1.30"),
    ("protein_total",           "Total Protein",           "TP",    "g/dL",  "6.0 - 8.3"),
    ("albumin",                 "Albumin",                 "ALB",   "g/dL",  "3.5 - 5.0"),
    ("ast",                     "AST (SGOT)",              "AST",   "U/L",   "10 - 40"),
    ("alt",                     "ALT (SGPT)",              "ALT",   "U/L",   "9 - 46"),
    ("bilirubin",               "Bilirubin, Total",        "TBIL",  "mg/dL", "0.2 - 1.2"),
    ("alkaline_phosphatase",    "Alkaline Phosphatase",    "ALP",   "U/L",   "44 - 121"),
    ("psa",                     "Prostate Specific Antigen", "PSA", "ng/mL", "0.0 - 4.0"),
    ("alpha_fetoprotein_level", "Alpha-Fetoprotein",       "AFP",   "ng/mL", "0.0 - 10.0"),
    ("plasma_ca19_9",           "CA 19-9",                 "CA19-9", "U/mL", "0 - 37"),
]


def fmt(key):
    v = TRUTH[key]
    return str(int(v)) if float(v).is_integer() else str(v)


def header(c, title, subtitle):
    c.setFont("Helvetica-Bold", 13)
    c.drawString(0.75 * inch, 10.4 * inch, title)
    c.setFont("Helvetica", 8)
    c.drawString(0.75 * inch, 10.2 * inch, subtitle)
    c.drawString(0.75 * inch, 10.05 * inch, f"PATIENT: DOE, JANE    AGE: {TRUTH['age']}    SEX: F")
    c.setStrokeColor(colors.black)
    c.line(0.75 * inch, 9.95 * inch, 7.75 * inch, 9.95 * inch)


def layout_1(path):
    """Two-column table, analyte / result / units / reference range."""
    c = canvas.Canvas(path, pagesize=LETTER)
    header(c, "COMPREHENSIVE METABOLIC PANEL + CBC", "Accession 4471182  Collected 03/14  Reported 03/15")
    y = 9.7 * inch
    c.setFont("Helvetica-Bold", 8)
    for x, lab in [(0.75, "TEST"), (3.6, "RESULT"), (4.6, "UNITS"), (5.7, "REFERENCE RANGE")]:
        c.drawString(x * inch, y, lab)
    y -= 0.18 * inch
    c.setFont("Helvetica", 9)
    for key, name, _ab, units, ref in ANALYTES:
        c.drawString(0.75 * inch, y, name)
        c.drawString(3.6 * inch, y, fmt(key))
        c.drawString(4.6 * inch, y, units)
        c.drawString(5.7 * inch, y, ref)
        y -= 0.26 * inch
    c.save()


def layout_2(path):
    """Narrow single column, value on the line beneath the analyte name."""
    c = canvas.Canvas(path, pagesize=LETTER)
    header(c, "Laboratory Report", "Specimen: Serum / Whole Blood")
    y = 9.6 * inch
    for key, name, _ab, units, ref in ANALYTES:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0.9 * inch, y, name.upper())
        y -= 0.16 * inch
        c.setFont("Helvetica", 9)
        c.drawString(1.1 * inch, y, f"{fmt(key)} {units}")
        y -= 0.14 * inch
        c.setFont("Helvetica-Oblique", 7)
        c.drawString(1.1 * inch, y, f"normal {ref} {units}")
        y -= 0.24 * inch
        if y < 0.9 * inch:
            c.showPage()
            y = 10.2 * inch
    c.save()


def layout_3(path):
    """
    Reference range printed immediately to the LEFT of the result.
    This is the adversarial case: a regex scanning forward from the analyte name
    reaches the range before it reaches the patient's value.
    """
    c = canvas.Canvas(path, pagesize=LETTER)
    header(c, "CHEMISTRY / HEMATOLOGY", "Ref ranges shown before result column")
    y = 9.7 * inch
    c.setFont("Helvetica-Bold", 8)
    for x, lab in [(0.75, "ANALYTE"), (3.2, "EXPECTED"), (5.2, "PATIENT"), (6.4, "UNITS")]:
        c.drawString(x * inch, y, lab)
    y -= 0.18 * inch
    c.setFont("Helvetica", 9)
    for key, name, _ab, units, ref in ANALYTES:
        c.drawString(0.75 * inch, y, name)
        c.drawString(3.2 * inch, y, ref)
        c.drawString(5.2 * inch, y, fmt(key))
        c.drawString(6.4 * inch, y, units)
        y -= 0.26 * inch
    c.save()


def layout_4(path):
    """Abbreviation codes only, no full analyte names."""
    c = canvas.Canvas(path, pagesize=LETTER)
    header(c, "LAB RESULTS - CODED", "Analyte codes per local formulary")
    y = 9.7 * inch
    c.setFont("Helvetica", 9)
    for key, _name, ab, units, ref in ANALYTES:
        c.drawString(0.9 * inch, y, f"{ab:<8} {fmt(key):>8} {units:<7} ({ref})")
        y -= 0.26 * inch
    c.save()


def layout_5(path):
    """Flag letters and markers sitting in the value column."""
    c = canvas.Canvas(path, pagesize=LETTER)
    header(c, "PATHOLOGY REPORT", "H = above range, L = below range")
    y = 9.7 * inch
    c.setFont("Helvetica", 9)
    for key, name, _ab, units, ref in ANALYTES:
        lo, hi = [p.strip() for p in ref.split("-")]
        try:
            flag = "H" if float(TRUTH[key]) > float(hi) else ("L" if float(TRUTH[key]) < float(lo) else "")
        except ValueError:
            flag = ""
        marker = f"  [{flag}]" if flag else ""
        c.drawString(0.9 * inch, y, f"{name}: {fmt(key)} {units}{marker}   ref {ref}")
        y -= 0.26 * inch
    c.save()


LAYOUTS = [
    ("layout_1_two_column_table.pdf", layout_1, "Two-column table with reference range column"),
    ("layout_2_stacked_single_column.pdf", layout_2, "Narrow single column, value beneath the name"),
    ("layout_3_range_before_result.pdf", layout_3, "Reference range printed before the result"),
    ("layout_4_abbreviation_codes.pdf", layout_4, "Abbreviation codes, no full names"),
    ("layout_5_flag_markers.pdf", layout_5, "Flag letters in the value column"),
]


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for fname, fn, _desc in LAYOUTS:
        p = os.path.join(OUT_DIR, fname)
        fn(p)
        paths.append(p)
        print(f"  wrote {p}")
    return paths


def post_pdf(path):
    """Multipart upload to /parse-pdf without adding a dependency."""
    boundary = "----OncovisionParserTest"
    with open(path, "rb") as f:
        content = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(path)}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{API}/parse-pdf",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.load(urllib.request.urlopen(req, timeout=180))


def main():
    print("Building lab report PDFs")
    paths = build()
    if "--build" in sys.argv:
        return

    print(f"\nParsing against {API}/parse-pdf")
    results = {}
    for (fname, _fn, desc), path in zip(LAYOUTS, paths):
        r = post_pdf(path)
        got = r.get("data", {}) if r.get("status") == "success" else {}

        correct, wrong, missed = [], [], []
        for key, truth in TRUTH.items():
            if key not in got:
                missed.append(key)
            elif abs(float(got[key]) - float(truth)) < 1e-6:
                correct.append(key)
            else:
                wrong.append((key, truth, got[key]))

        extra = [k for k in got if k not in TRUTH]
        results[fname] = {
            "layout": desc,
            "status": r.get("status"),
            "fields_expected": len(TRUTH),
            "correct": len(correct),
            "wrong": [{"field": k, "truth": t, "parsed": g} for k, t, g in wrong],
            "missed": missed,
            "unexpected": extra,
            "accuracy": round(len(correct) / len(TRUTH), 3),
        }

        print(f"\n  {desc}")
        print(f"    correct {len(correct)}/{len(TRUTH)}   wrong {len(wrong)}   missed {len(missed)}")
        for k, t, g in wrong:
            print(f"      WRONG   {k}: report says {t}, parser read {g}")
        if missed:
            print(f"      MISSED  {', '.join(missed)}")
        if extra:
            print(f"      EXTRA   {', '.join(extra)}")

    # Per field across layouts, which is where a systematic weakness shows up.
    per_field = {}
    for key in TRUTH:
        ok = sum(1 for r in results.values()
                 if key not in r["missed"] and key not in [w["field"] for w in r["wrong"]])
        per_field[key] = f"{ok}/{len(results)}"

    with open("parser_accuracy.json", "w") as f:
        json.dump({"truth": TRUTH, "per_layout": results, "per_field": per_field}, f, indent=2)

    total = sum(r["correct"] for r in results.values())
    possible = len(TRUTH) * len(results)
    print("\n" + "=" * 72)
    print(f"OVERALL FIELD-LEVEL ACCURACY  {total}/{possible} = {total/possible:.1%}")
    print("=" * 72)
    print("\nPer field across all five layouts:")
    for key, score in sorted(per_field.items(), key=lambda kv: kv[1]):
        print(f"  {key:<26}{score}")
    print("\nwrote parser_accuracy.json")


if __name__ == "__main__":
    main()
