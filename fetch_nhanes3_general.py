"""
NHANES III (1988-1994) as an external test cohort for the general panel.

Why this one is left
--------------------
The general panel is the last shipped rule-out cut with no test outside the
survey it was fitted on. The bowel cut was applied unchanged to NHANES III and
kept its promise inside tolerance. The liver and lung panels ship no cut at all
-- the cost model removed liver's, and lung's was dropped because catching
every case there means excluding almost nobody. That leaves general.

It matters more than the others, not less. The general panel excludes 22% of
everyone who runs it, on five questions and no blood test at all, and a cut
that over-promises here reassures more people than any other panel on the site.

What is harmonised, and what is not
-----------------------------------
    age, gender   direct
    bmi           BMPBMI from the examination file, decimal point written
    smoking       HAR1 and HAR3, the same never/former/current coding the
                  continuous survey builds from SMQ020 and SMQ040
    alcohol       NOT harmonised, and this is the honest limitation of this
                  cohort

NHANES III did not ask the question the panel trains on. The continuous survey
asks ALQ130, average drinks on a drinking day; NHANES III recorded beer, wine
and liquor as times per month in a food frequency questionnaire. Those are
different quantities and mapping one onto the other would be inventing a
conversion and then validating against it.

So alcohol is filled with the training median, which is exactly what the
service does for a patient who leaves the question blank. That makes this a
test of the panel as used by someone who answered four questions of five,
rather than a test of the whole panel, and it is written up that way. It also
means the result is a floor: the full panel cannot do worse than this.

Target
------
Any cancer first diagnosed within RECENT_YEARS of the exam, matching
fetch_nhanes_screening.py. Anyone diagnosed longer ago is EXCLUDED rather than
counted as a negative, because a cured survivor with ordinary bloodwork is not
a negative example of "has cancer now" -- they are a different question.

Run:  python fetch_nhanes3_general.py
"""

import io
import ssl
import urllib.request

import pandas as pd

LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"
EXAM = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat"
OUT = "data/nhanes3_general.csv"

RECENT_YEARS = 4          # matches fetch_nhanes_screening.py

# One-based inclusive positions, as written in the SAS input statements
# published beside each file.
ADULT_SPEC = {
    "SEQN": (1, 5),
    "HSSEX": (15, 15),
    # 18-19 in adult.dat, NOT the 16-17 that lab.dat uses for the same name.
    # The two files have different layouts and copying the lab position across
    # parsed age as a single digit, which the age filter then silently removed.
    "HSAGEIR": (18, 19),
    "smoked_100": (2281, 2281),    # HAR1,  1 yes 2 no
    "smokes_now": (2285, 2285),    # HAR3,  1 yes 2 no
    "skin_cancer": (1478, 1478),   # HAC1N
    "other_cancer": (1479, 1479),  # HAC1O
    "age_at_dx": (1524, 1526),     # HAC3OR, 004-089 valid, 090 is 90+
}

EXAM_SPEC = {
    "SEQN": (1, 5),
    "bmi_raw": (1524, 1527),       # BMPBMI, Z6.1 -- decimal point is written
}

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=1800, context=_ctx).read()


def read_fixed(raw: bytes, spec: dict) -> pd.DataFrame:
    return pd.read_fwf(io.BytesIO(raw),
                       colspecs=[(a - 1, b) for a, b in spec.values()],
                       names=list(spec))


def clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s >= 88888).mask(s.isin([8888, 9999, 888, 999, 88, 99]))


def main():
    print("NHANES III (1988-1994), external test cohort for the general panel\n")

    print("  downloading the household adult file (65 MB) ...", flush=True)
    adult = read_fixed(get(LAB), ADULT_SPEC)
    print(f"    {len(adult):,} records", flush=True)

    print("  downloading the examination file ...", flush=True)
    exam = read_fixed(get(EXAM), EXAM_SPEC)
    print(f"    {len(exam):,} records", flush=True)

    df = adult.merge(exam, on="SEQN", how="left")
    age = pd.to_numeric(df["HSAGEIR"], errors="coerce")
    df = df[age >= 20].copy()

    # BMPBMI is declared Z6.1 and that format WRITES the decimal point rather
    # than implying it: bytes 1524-1527 hold the four characters "25.5". The
    # first version of this file divided by ten, on the usual NHANES III
    # convention, and produced a cohort with a median BMI of 2.6.
    bmi = clean(df["bmi_raw"])
    # A misread decimal would move every BMI by a factor of ten and leave every
    # downstream number looking perfectly plausible, so it is checked against
    # the range a population of adults actually occupies rather than trusted.
    median_bmi = float(bmi.median())
    if not 18.0 < median_bmi < 40.0:
        raise ValueError(
            f"BMI median parsed as {median_bmi:.1f}, which is not a plausible "
            f"adult population value; the implied decimal in BMPBMI is wrong")
    print(f"    BMI parses to a median of {median_bmi:.1f}")

    # Never 0, former 1, current 2 -- the coding the continuous cohort uses.
    smoked_100 = pd.to_numeric(df["smoked_100"], errors="coerce")
    smokes_now = pd.to_numeric(df["smokes_now"], errors="coerce")
    smoking = pd.Series(float("nan"), index=df.index)
    smoking[smoked_100 == 2] = 0
    smoking[(smoked_100 == 1) & (smokes_now == 2)] = 1
    smoking[(smoked_100 == 1) & (smokes_now == 1)] = 2

    ever_any = (pd.to_numeric(df["other_cancer"], errors="coerce") == 1) | \
               (pd.to_numeric(df["skin_cancer"], errors="coerce") == 1)
    age_dx = clean(df["age_at_dx"]).mask(lambda s: s > 90)
    years_since = pd.to_numeric(df["HSAGEIR"], errors="coerce") - age_dx

    recent = ever_any & years_since.notna() & years_since.between(0, RECENT_YEARS)
    never = ~ever_any
    keep = recent | never

    out = pd.DataFrame({
        "age": pd.to_numeric(df["HSAGEIR"], errors="coerce"),
        # NHANES III codes 1 male, 2 female; this application uses 0 female.
        "gender": (pd.to_numeric(df["HSSEX"], errors="coerce") == 1).astype(int),
        "bmi": bmi,
        "smoking": smoking,
        "years_since_diagnosis": years_since,
        "recent_cancer": recent.astype(int),
    })[keep.values]
    out = out.dropna(subset=["age", "gender", "bmi", "smoking", "recent_cancer"])

    n, pos = len(out), int(out["recent_cancer"].sum())
    out.to_csv(OUT, index=False)
    print(f"\n  n={n:,}  cancers within {RECENT_YEARS} years={pos}  "
          f"({pos / n:.2%})  ->  {OUT}")
    print(f"  survivors diagnosed longer ago than {RECENT_YEARS} years are "
          f"excluded, not counted as negatives")
    print("  alcohol_intake is absent by design; see the module docstring")


if __name__ == "__main__":
    main()
