"""
For a human second analyst: re-runs the red-flag analyses and compares every headline number with the committed value.
It prints MATCH or DIFFERS for each. Bootstrap intervals are not compared (they move slightly); point estimates are.

Run from the repository root:  python tools/check_headline_numbers.py
Needs the data files committed in data/ and data/audit/ (no downloads). Takes a few minutes.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
FILES = ["red_flag_labs_result.json", "red_flag_angles_result.json", "red_flag_more_result.json"]
SCRIPTS = ["red_flag_labs.py", "red_flag_angles.py", "red_flag_more.py"]
CHECKS = [
    ("Share of US adults 50+ with guideline IDA", FILES[0], ["A_societal_2017_2020", "ida_prevalence", "est"]),
    ("People with it, millions", FILES[0], ["A_societal_2017_2020", "ida_prevalence", "millions"]),
    ("Not being treated (survey question)", FILES[0], ["A_societal_2017_2020", "ida_untreated_share", "est"]),
    ("Not treated, 2001-2002 replication", FILES[0], ["R1_replication_2001_2002", "ida_untreated_share", "est"]),
    ("5-year cancer-death odds ratio", FILES[0], ["B_scientific_5y", "pooled", "or_cancer_ida"]),
    ("5-year other-cause-death odds ratio", FILES[0], ["B_scientific_5y", "pooled", "or_other_ida"]),
    ("No iron at all, even a multivitamin", FILES[1], ["A1_untreated", "plus_any_iron_supplement", "est"]),
    ("On a blood thinner or acid reducer or NSAID (with the flag)", FILES[1], ["A4_medicines", "any_of_these", "ida", "est"]),
    ("Share of cases the small-red-cell shortcut finds (2017-2020)", FILES[2], ["T1", "2017-2020", "microcytic_anaemia", "sensitivity"]),
    ("Ferritin tests per case if added for any anaemia", FILES[2], ["T1", "2017-2020", "any_anaemia", "ferritin_tests_per_case"]),
]


def read(name, path):
    v = json.load(open(os.path.join(EXP, name)))
    for k in path:
        v = v[k]
    return v


def main():
    before = {c[0]: read(c[1], c[2]) for c in CHECKS}
    originals = {f: open(os.path.join(EXP, f), encoding="utf-8").read() for f in FILES}
    try:
        for s in SCRIPTS:
            print(f"running {s} ...", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", s], cwd=EXP, check=True, stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        bad = 0
        for label, name, path in CHECKS:
            now = read(name, path)
            same = abs(now - before[label]) < 0.005
            bad += not same
            print(f"{'MATCH  ' if same else 'DIFFERS'}  {label}: committed {before[label]:.4g}, re-run {now:.4g}")
        print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} match.")
    finally:
        for f, text in originals.items():
            open(os.path.join(EXP, f), "w", encoding="utf-8").write(text)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
