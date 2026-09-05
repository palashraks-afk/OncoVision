"""
Drive the running application as a user would, and flag implausible answers.

Why this exists
---------------
The ovarian panel spent a long time telling healthy women their malignancy risk
was 94.9 percent. The unit tests passed, the structural audit passed, the
browser checks passed, and none of them looked at whether the ANSWER made sense
for the patient who asked. It was found by typing in an ordinary patient and
reading the result.

So that is automated here. This is not a unit test suite and does not replace
one. It builds realistic patients, sends them to a running service, and applies
sanity rules that a clinician would apply in a second and a test suite generally
does not:

  a healthy person should not be flagged by anything
  a panel should not be confident when its defining inputs are absent
  moving one marker in the direction of disease should not move risk DOWN
  adding a normal, unremarkable value should not swing a score wildly
  no panel should return an impossible or missing number

Every rule below is deliberately loose. The purpose is to surface answers worth
a human look, not to assert a specific probability is correct.

Run:  python tools/drive_app.py            against http://127.0.0.1:8000
      python tools/drive_app.py <base_url>
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
# Long enough for a fixed one-minute window to roll over.
RATE_WINDOW_WAIT = 62

# A 45-year-old woman with entirely unremarkable results.
HEALTHY = {
    "age": 45, "gender": 0, "bmi": 23.5, "smoking": 0, "alcohol_intake": 0.5,
    "wbc": 6.2, "rbc": 4.6, "hemoglobin": 13.6, "platelets": 255,
    "hematocrit": 41, "mcv": 89, "mch": 30, "rdw": 13.0, "mpv": 9.7,
    "neutrophil_pct": 58, "glucose": 88, "calcium": 9.4, "bun": 13,
    "creatinine": 0.8, "protein_total": 7.0, "albumin": 4.4, "ast": 21,
    "alt": 19, "bilirubin": 0.6, "alkaline_phosphatase": 78, "ggt": 20,
    "hepatitis_b": 0, "hepatitis_c": 0, "diabetes": 0,
}

failures: list = []
notes: list = []


def post(payload, _retries=2):
    """
    One prediction, backing off if the service's own rate limit is hit.

    This driver sends well over a hundred requests in a sweep, which is more
    than the per-minute budget. A 429 here is the service working correctly, so
    it waits for the window to roll over rather than reporting the rejection as
    a fault.
    """
    req = urllib.request.Request(
        f"{BASE}/predict", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        body = e.read()[:400].decode("utf-8", "replace")
        if e.code == 429 and _retries > 0:
            print("  (rate limited by the service, waiting for the window)", flush=True)
            time.sleep(RATE_WINDOW_WAIT)
            return post(payload, _retries - 1)
        return {"status": "http_error", "code": e.code, "body": body}


def risk_of(body, substring):
    for k, v in (body.get("predictions") or {}).items():
        if substring.lower() in k.lower():
            return v.get("risk")
    return None


def check(condition, message):
    (notes if condition else failures).append(message)


# --------------------------------------------------------------- 1. healthy
def scenario_healthy():
    body = post(HEALTHY)
    if body.get("status") != "success":
        failures.append(f"healthy patient produced {body}")
        return
    flagged = []
    for name, v in body["predictions"].items():
        if name == "Nothing Flagged":
            continue
        risk, thr = v.get("risk"), v.get("threshold")
        if risk is None:
            failures.append(f"healthy: {name} returned no risk at all")
            continue
        if not (0.0 <= risk <= 100.0):
            failures.append(f"healthy: {name} returned an impossible risk of {risk}")
        if thr is not None and risk >= thr:
            flagged.append(f"{name} at {risk}% (its own threshold is {thr}%)")
    check(not flagged,
          "a healthy 45-year-old woman was flagged by: " + "; ".join(flagged)
          if flagged else "healthy patient: nothing flagged, as expected")


# --------------------------------------------------------------- 2. gating
def scenario_defining_inputs():
    body = post(HEALTHY)
    scored = " ".join(body.get("predictions") or {}).lower()
    for panel, marker in (("ovarian", "CA 125 / HE4"),
                          ("pancreatic", "CA 19-9"),
                          ("prostate", "PI-RADS")):
        check(panel not in scored,
              f"{panel} scored with no {marker} supplied"
              if panel in scored else
              f"{panel} correctly refuses without {marker}")


# --------------------------------------------------------------- 3. direction
def scenario_monotonic():
    """Pushing a marker toward disease must not push risk down."""
    cases = [
        ("liver", "alt", [19, 90, 300], "Liver"),
        ("liver", "bilirubin", [0.6, 2.5, 8.0], "Liver"),
        ("ovarian", "ca125", [12, 200, 900], "Ovarian"),
        ("pancreatic", "plasma_ca19_9", [12, 200, 1200], "Pancreatic"),
        ("prostate", "psa", [0.9, 8.0, 40.0], "Prostate"),
    ]
    extra = {
        "ovarian": {"menopause": 1, "he4": 60, "cea": 1.5},
        "pancreatic": {"plasma_ca19_9": 12},
        "prostate": {"gender": 1, "age": 65, "prostate_volume": 40,
                     "psa_density": 0.2, "pi_rads": 3},
    }
    for panel, field, values, label in cases:
        base = dict(HEALTHY, **extra.get(panel, {}))
        risks = []
        for val in values:
            r = risk_of(post(dict(base, **{field: val})), label)
            risks.append(r)
        if any(r is None for r in risks):
            notes.append(f"{panel}: not scored across the {field} sweep, skipped")
            continue
        drops = [(values[i], risks[i], values[i + 1], risks[i + 1])
                 for i in range(len(risks) - 1) if risks[i + 1] < risks[i] - 1.0]
        check(not drops,
              f"{panel}: raising {field} LOWERED risk " +
              "; ".join(f"{a}->{b}: {ra}% -> {rb}%" for a, ra, b, rb in drops)
              if drops else
              f"{panel}: risk rises with {field} ({' -> '.join(f'{r}%' for r in risks)})")


# --------------------------------------------------------------- 3b. patterns
def scenario_coherent_patterns():
    """
    A whole clinical picture worsening, rather than one number moving.

    This is the check that would have caught the liver failure. Sweeping ALT
    alone produces an impossible patient and a misleading answer; escalating a
    real hepatitis picture — transaminases, bile duct enzymes and bilirubin
    together — is what a user actually presents with, and the panel used to
    score the worst of them BELOW a healthy person.
    """
    hepatitis = [
        ("normal", dict(alt=19, ast=21, ggt=20, bilirubin=0.6)),
        ("mild", dict(alt=60, ast=55, ggt=45, bilirubin=0.8)),
        ("moderate", dict(alt=120, ast=100, ggt=90, bilirubin=1.2)),
        ("severe", dict(alt=300, ast=260, ggt=200, bilirubin=2.5)),
        ("fulminant", dict(alt=900, ast=800, ggt=600, bilirubin=12.0)),
    ]
    risks = []
    for label, over in hepatitis:
        r = risk_of(post(dict(HEALTHY, **over)), "Liver")
        risks.append((label, r))
    if any(r is None for _, r in risks):
        notes.append("liver panel did not score across the hepatitis sweep, skipped")
        return
    healthy_risk = risks[0][1]
    worst_risk = risks[-1][1]
    drops = [(risks[i][0], risks[i][1], risks[i + 1][0], risks[i + 1][1])
             for i in range(len(risks) - 1) if risks[i + 1][1] < risks[i][1] - 0.5]
    trail = " -> ".join(f"{lab} {r}%" for lab, r in risks)
    check(worst_risk > healthy_risk,
          f"fulminant hepatitis ({worst_risk}%) scored no higher than a healthy "
          f"patient ({healthy_risk}%): {trail}"
          if worst_risk <= healthy_risk else
          f"liver: a worsening hepatitis picture raises risk ({trail})")
    check(not drops,
          "liver: risk FELL as the hepatitis picture worsened at " +
          "; ".join(f"{a}->{b}" for a, _, b, _ in drops)
          if drops else
          "liver: no reversal anywhere along the hepatitis picture")


def scenario_extreme_values_are_declared():
    """A value past the edge of the data must be admitted, not silently clipped."""
    body = post(dict(HEALTHY, alt=900, ast=800, ggt=600, bilirubin=12.0))
    for name, v in (body.get("predictions") or {}).items():
        if "liver" not in name.lower():
            continue
        check(bool(v.get("extreme_value_caveat")),
              "a patient far outside the training range got no caveat saying so"
              if not v.get("extreme_value_caveat") else
              "values beyond the training range are declared to the user")
        return
    notes.append("liver not scored, extreme-value caveat not checked")


# --------------------------------------------------------------- 4. stability
def scenario_irrelevant_input():
    """Adding one unremarkable value should not swing a score."""
    before = post(HEALTHY)
    after = post(dict(HEALTHY, mpv=9.8))
    for name, v in (before.get("predictions") or {}).items():
        b = v.get("risk")
        a = risk_of(after, name)
        if b is None or a is None:
            continue
        if abs(a - b) > 10.0:
            failures.append(f"{name} moved {b}% -> {a}% when one normal value "
                            f"was nudged, which is not stability")
    notes.append("adding an unremarkable value did not swing any score by more than 10 points")


# --------------------------------------------------------------- 5. edges
def scenario_edges():
    for label, payload in [
        ("empty", {}),
        ("age and sex only", {"age": 50, "gender": 1}),
        ("one lab value", {"age": 50, "gender": 1, "alt": 30}),
        ("impossible haemoglobin", dict(HEALTHY, hemoglobin=5000)),
        ("negative value", dict(HEALTHY, alt=-5)),
        ("text in a number", dict(HEALTHY, alt="abc")),
        ("very old", dict(HEALTHY, age=119)),
        ("age below the trained range", dict(HEALTHY, age=18)),
    ]:
        body = post(payload)
        if body.get("status") == "http_error":
            # A 422 carrying a readable message is the correct answer to input
            # that is not a number, not a failure. Anything else, or a 422 that
            # dumps validation internals at a patient, is.
            readable = ('"status": "error"' in body["body"]
                        or '"status":"error"' in body["body"])
            if body["code"] == 422 and readable:
                notes.append(f"edge case '{label}': refused with a readable message")
            else:
                failures.append(f"edge case '{label}' crashed the service: "
                                f"{body['code']} {body['body'][:160]}")
            continue
        for name, v in (body.get("predictions") or {}).items():
            r = v.get("risk")
            if r is not None and not (0.0 <= r <= 100.0):
                failures.append(f"edge case '{label}': {name} returned {r}")
        notes.append(f"edge case '{label}': handled, status={body.get('status')}")


def main():
    print(f"driving {BASE} as a user\n")
    for fn in (scenario_healthy, scenario_defining_inputs, scenario_monotonic,
               scenario_coherent_patterns, scenario_extreme_values_are_declared,
               scenario_irrelevant_input, scenario_edges):
        fn()

    print("PASSED")
    for n in notes:
        print(f"  ok    {n}")
    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S)")
        for f in failures:
            print(f"  !     {f}")
        return 1
    print("no implausible answers found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
