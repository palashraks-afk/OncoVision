"""
Structural checks that catch the specific mistakes this project has actually made.

Not a linter. Every check here exists because the corresponding bug shipped at
some point and was found by hand:

  duplicate keys      a second `prostate` entry in COHORT_DESIGN silently
                      overwrote the first, so the live app showed stale text
                      that no longer matched the panel. Python does not warn.
  schema drift        a model may only use features the application can
                      actually collect. When a panel gains a feature and the
                      form does not, the panel scores on an imputed median for
                      every real patient and looks fine in CV.
  bounds gap          a numeric field the API accepts but does not
                      range-check will happily take a typo of 5000.
  withdrawn leakage   a panel withdrawn for instability must be gone from the
                      form and the API too, not just from training.

Run:  python tools/audit.py
Exits non-zero if anything is wrong, so it can gate a commit.
"""

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

problems = []
notes = []


def fail(msg):
    problems.append(msg)


# --------------------------------------------------------------- duplicate keys
def duplicate_keys(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = {}
        for k in node.keys:
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                continue
            if k.value in seen:
                fail(f"{os.path.relpath(path, ROOT)}:{k.lineno} duplicate key "
                     f"{k.value!r} (first at line {seen[k.value]}) — the later "
                     f"one silently wins")
            seen[k.value] = k.lineno


for rel in ("train_models.py", "backend/api.py", "evaluate.py"):
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        duplicate_keys(p)

# --------------------------------------------------------------- load the config
import train_models as tm  # noqa: E402

# --------------------------------------------------------------- schema drift
form_src = open(os.path.join(ROOT, "frontend/src/app/fields.ts"), encoding="utf-8").read()
form_keys = set(re.findall(r'\bkey:\s*"([A-Za-z0-9_]+)"', form_src))
if not form_keys:
    fail("fields.ts: could not read any field keys — the regex no longer matches")

api_src = open(os.path.join(ROOT, "backend/api.py"), encoding="utf-8").read()

DEMOGRAPHIC = {"age", "gender"}

for cfg in tm.DATASETS:
    name = cfg["name"]
    if name in tm.WITHDRAWN:
        continue
    for feat in cfg["features"]:
        if feat in DEMOGRAPHIC:
            continue
        if feat not in form_keys:
            fail(f"panel {name!r} trains on {feat!r}, which the form never asks "
                 f"for — every real patient gets the imputed median")

# Fields the form asks for that no live panel reads are dead questions.
used = {f for cfg in tm.DATASETS if cfg["name"] not in tm.WITHDRAWN
        for f in cfg["features"]}
dead = sorted(form_keys - used - DEMOGRAPHIC)
if dead:
    notes.append(f"{len(dead)} form fields no live panel reads: {', '.join(dead)}")

# --------------------------------------------------------------- bounds coverage
bounds_block = re.search(r"FIELD_BOUNDS\s*[:=][^{]*\{(.*?)\n\}", api_src, re.S)
if not bounds_block:
    bounds_block = re.search(r"BOUNDS\s*[:=][^{]*\{(.*?)\n\}", api_src, re.S)
if bounds_block:
    bounded = set(re.findall(r'"([A-Za-z0-9_]+)"\s*:', bounds_block.group(1)))
    numeric_unbounded = sorted(
        f for f in used
        if f not in bounded and f not in DEMOGRAPHIC
        and f not in {"diabetes", "hepatitis_b", "hepatitis_c"}
    )
    if numeric_unbounded:
        notes.append(f"{len(numeric_unbounded)} model features have no range check: "
                     f"{', '.join(numeric_unbounded[:12])}"
                     f"{' ...' if len(numeric_unbounded) > 12 else ''}")
else:
    notes.append("could not locate the bounds table in api.py to check it")

# --------------------------------------------------------------- withdrawn leakage
# A withdrawn panel must not still be claimed as a reader of any input. This is
# how the cervical panel's sexual-history questions stayed wired into the API
# long after the panel itself was withdrawn: nothing scored them, but the
# service still accepted them and still named cervical as their consumer.
for w in tm.WITHDRAWN:
    for m in re.finditer(rf'\[([^\]]*)\]', api_src):
        if f'"{w}"' in m.group(1):
            line = api_src[:m.start()].count("\n") + 1
            fail(f"backend/api.py:{line} names withdrawn panel {w!r} as a reader "
                 f"of an input. Nothing scores it, so the field is accepted and "
                 f"discarded — remove the field rather than the mention.")

# --------------------------------------------------------------- inputs nothing reads
# The service should not accept what it cannot use. Beyond dead weight, this is
# a privacy point: an API that accepts a sexual history no model reads is
# collecting sensitive data for nothing.
req = re.search(r"class\s+PatientData\(BaseModel\):(.*?)(?=\nclass |\n\n\n)", api_src, re.S)
if req:
    accepted = set(re.findall(r"^\s{4}([a-z_0-9]+)\s*:\s*Optional", req.group(1), re.M))
    unread = sorted(accepted - used - DEMOGRAPHIC - form_keys)
    if unread:
        fail(f"backend/api.py accepts {len(unread)} field(s) that no live panel "
             f"reads and no form sends: {', '.join(unread)}")

# --------------------------------------------------------------- sample-case coverage
# make_cases.py needs a healthy reference reading for every feature a live panel
# scores, and it raises KeyError on the first one it lacks. That is how the
# breast expansion broke the sample-case pool: the panel grew from four features
# to thirty and the reference table still had four.
try:
    import make_cases  # noqa: E402

    no_reference = sorted({
        f for cfg in tm.DATASETS if cfg["name"] not in tm.WITHDRAWN
        for f in cfg["features"] if f not in make_cases.NORMAL
    })
    if no_reference:
        fail(f"make_cases.py has no healthy reference reading for "
             f"{len(no_reference)} scored feature(s): {', '.join(no_reference[:10])}"
             f"{' ...' if len(no_reference) > 10 else ''} — the sample case pool "
             f"will fail to build")
except Exception as exc:  # pragma: no cover
    notes.append(f"could not import make_cases to check its reference table: {exc}")

# --------------------------------------------------------------- panel metadata
for cfg in tm.DATASETS:
    name = cfg["name"]
    if name in tm.WITHDRAWN:
        continue
    if name not in tm.PANEL_KIND:
        fail(f"panel {name!r} has no entry in PANEL_KIND, so the app cannot say "
             f"whether it is a screening test")

# --------------------------------------------------------------- report
print("=" * 74)
if problems:
    print(f"{len(problems)} PROBLEM(S)\n")
    for p in problems:
        print(f"  ! {p}")
else:
    print("no structural problems")
if notes:
    print(f"\n{len(notes)} note(s) — worth a look, not necessarily wrong\n")
    for n in notes:
        print(f"  - {n}")
print("=" * 74)
sys.exit(1 if problems else 0)
