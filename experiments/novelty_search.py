"""
Recorded novelty search for the cheap-markers paper (addendum N in docs/CHEAP_MARKERS_PREREG.md).

Fixed Europe PMC query for blood-count and inflammation markers in relation to cancer, all
records 2005-2026, then cue rules over title and abstract to find the papers closest to ours:
several markers, a comparison of cancer with other causes of death (negative control,
non-cancer death, competing risks, decedent-only), AND validation in separate data.
Every record that passes all three cues is written to data/audit/novelty_search.csv to be read.

Run:  python experiments/novelty_search.py
"""

import os
import re
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MARKERS = ('("neutrophil-to-lymphocyte" OR "neutrophil to lymphocyte" OR NLR OR "systemic immune-inflammation" OR "systemic immune inflammation" OR "platelet-to-lymphocyte" OR '
           '"monocyte-to-lymphocyte" OR SIRI OR NPAR OR "neutrophil percentage-to-albumin" OR "lung cancer inflammation index" OR "red cell distribution width" OR RDW OR '
           '"prognostic nutritional index" OR "C-reactive protein")')
QUERY = (f"(TITLE_ABS:{MARKERS}) AND (TITLE_ABS:(cancer OR neoplasm OR carcinoma OR tumour OR tumor)) AND "
         "(TITLE_ABS:(mortality OR death OR incidence OR \"risk prediction\" OR \"cancer risk\")) AND "
         "(TITLE_ABS:(\"general population\" OR population-based OR NHANES OR \"UK Biobank\" OR prospective OR cohort)) AND PUB_YEAR:[2005 TO 2026]")

MARK_CUES = {
    "NLR": r"neutrophil[- ]to[- ]lymphocyte|\bNLR\b", "SII": r"immune[- ]inflammation|\bSII\b", "PLR": r"platelet[- ]to[- ]lymphocyte|\bPLR\b",
    "MLR": r"monocyte[- ]to[- ]lymphocyte|\bMLR\b", "SIRI": r"\bSIRI\b", "NPAR": r"\bNPAR\b|neutrophil percentage", "ALI": r"lung cancer inflammation index|\bALI\b",
    "PNI": r"prognostic nutritional|\bPNI\b", "RDW": r"red cell distribution|\bRDW\b", "CRP": r"C-reactive protein|\bCRP\b", "albumin": r"albumin"}
NEG = r"negative[- ]control|non-?cancer (death|mortality)|other[- ]cause|competing|cause-specific|specificity|cancer-specific and|cardiovascular and cancer"
DEC = r"among (those|people|participants|individuals|patients) who died|decedent|death-only|case-only"
VAL = r"external validation|validation cohort|independent cohort|held-out|hold-out|temporal validation|test set|validation set|separate cohort|replicat"


def main():
    recs, cursor = [], "*"
    while True:
        r = requests.get(BASE, params={"query": QUERY, "format": "json", "pageSize": 1000, "resultType": "core", "cursorMark": cursor}, timeout=180).json()
        recs += r["resultList"]["result"]
        print(f"  fetched {len(recs)} of {r['hitCount']}", flush=True)
        nxt = r.get("nextCursorMark")
        if not nxt or nxt == cursor or len(recs) >= min(r["hitCount"], 8000):
            break
        cursor = nxt
        time.sleep(0.5)
    total = r["hitCount"]
    rows = []
    for x in recs:
        text = f"{x.get('title', '')} {re.sub('<[^>]+>', ' ', x.get('abstractText') or '')}"
        marks = [k for k, p in MARK_CUES.items() if re.search(p, text, re.I)]
        neg, dec, val = bool(re.search(NEG, text, re.I)), bool(re.search(DEC, text, re.I)), bool(re.search(VAL, text, re.I))
        rows.append({"pmid": x.get("pmid") or x.get("id"), "year": x.get("pubYear"), "title": (x.get("title") or "")[:180], "n_markers": len(marks), "markers": ";".join(marks),
                     "neg_or_cause_cue": neg, "decedent_cue": dec, "validation_cue": val})
    df = pd.DataFrame(rows)
    cand = df[(df["n_markers"] >= 3) & (df["neg_or_cause_cue"] | df["decedent_cue"]) & df["validation_cue"]]
    close = df[(df["n_markers"] >= 2) & (df["neg_or_cause_cue"] | df["decedent_cue"])]
    out = os.path.join(ROOT, "data", "audit", "novelty_search.csv")
    cand.sort_values("year", ascending=False).to_csv(out, index=False)
    close.to_csv(os.path.join(ROOT, "data", "audit", "novelty_search_looser.csv"), index=False)
    print(f"\nrecords matching the query: {total:,}; screened by cue: {len(df):,}")
    print(f"  at least 3 markers + cause-comparison cue + validation cue: {len(cand)}  -> {out}")
    print(f"  looser (2+ markers + cause-comparison cue, no validation requirement): {len(close)}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    print(cand[["pmid", "year", "n_markers", "title"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
