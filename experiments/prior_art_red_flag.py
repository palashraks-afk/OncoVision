"""
Prior-art check for the red-flag-labs findings (26 November 2026). Searches Europe PMC (which includes PubMed, PubMed
Central and preprint servers such as medRxiv) with many phrasings of the two core claims, lists every hit, and keeps
those whose title or abstract touches the specific claims for reading. Writes data/audit/prior_art_red_flag.csv.

Claims checked:
  C1  a national (US) estimate of how many older adults have guideline-defined iron-deficiency anaemia that is untreated,
      including among people in regular contact with care, and its persistence over time;
  C2  the cancer-death signal of iron-deficiency anaemia measured in a population tested regardless of symptoms, with a
      specificity check against other-cause death.

Run:  python experiments/prior_art_red_flag.py
"""
import csv
import os
import re
import time

import requests

B = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def ta(x):
    return f"(TITLE:({x}) OR ABSTRACT:({x}))"


IDA = '"iron deficiency anemia" OR "iron deficiency anaemia" OR "iron-deficiency anemia" OR "iron-deficiency anaemia" OR "iron deficiency"'
Q = {
    # C1
    "C1a IDA + untreated/undertreated + national/NHANES": ta(IDA) + " AND " + ta('untreated OR "not treated" OR undertreat* OR undertreated OR "treatment gap" OR "unaddressed"') + " AND " + ta('NHANES OR "nationally representative" OR national OR "United States"'),
    "C1b IDA + unevaluated/uninvestigated + adults": ta(IDA) + " AND " + ta('unevaluated OR uninvestigated OR "not evaluated" OR "not investigated" OR "lack of evaluation" OR "inadequate evaluation" OR "diagnostic evaluation"') + " AND " + ta("adults OR older OR men OR postmenopausal"),
    "C1c IDA + guideline adherence + endoscopy + population": ta(IDA) + " AND " + ta('adherence OR "guideline-concordant" OR concordance OR underuse OR "appropriate evaluation"') + " AND " + ta("endoscop* OR colonoscop* OR \"gastrointestinal evaluation\""),
    "C1d anaemia awareness/treatment in NHANES": ta("anemia OR anaemia") + " AND " + ta('NHANES OR "National Health and Nutrition"') + " AND " + ta("awareness OR aware OR treated OR treatment OR untreated OR undiagnosed"),
    "C1e iron deficiency prevalence US adults 2017-2020": ta(IDA) + " AND " + ta("prevalence") + " AND " + ta('NHANES OR "nationally representative"') + " AND " + ta("adults"),
    "C1f usual source of care + anaemia or iron": ta('"usual source of care" OR "regular source of care" OR "health care visits" OR "healthcare utilization"') + " AND " + ta("anemia OR anaemia OR \"iron deficiency\" OR ferritin"),
    # C2
    "C2a IDA + cancer mortality + population/cohort": ta(IDA) + " AND " + ta('"cancer mortality" OR "cancer death" OR "cancer-specific" OR "colorectal cancer" OR "gastrointestinal cancer"') + " AND " + ta('cohort OR "population-based" OR NHANES OR "general population"'),
    "C2b ferritin + haemoglobin + mortality cause-specific": ta("ferritin") + " AND " + ta("hemoglobin OR haemoglobin OR anemia OR anaemia") + " AND " + ta('"cause-specific" OR "cancer mortality" OR "competing risk*"'),
    "C2c anaemia + indication bias / unselected population + cancer": ta("anemia OR anaemia OR \"iron deficiency\"") + " AND " + ta('"unselected" OR "indication bias" OR "confounding by indication" OR "tested regardless" OR "asymptomatic population" OR "general population"') + " AND " + ta("cancer"),
    "C2d NHANES III ferritin + mortality": ta('"NHANES III" OR "Third National Health"') + " AND " + ta("ferritin OR \"iron status\" OR \"iron deficiency\"") + " AND " + ta("mortality OR death"),
}
KEEP = re.compile(r"untreat|undertreat|unevaluat|uninvestigat|not (been )?(evaluated|investigated|treated)|guideline|adherence|NHANES|national|usual source|ferritin.*(cancer|mortalit)|iron deficiency.*(cancer|mortalit|endoscop|colonoscop)|anaemi?a.*(cancer|mortalit)|anemi?a.*(cancer|mortalit)", re.I)


def main():
    rows = []
    for name, q in Q.items():
        for attempt in range(4):
            r = requests.get(B, params={"query": q, "format": "json", "pageSize": 500, "resultType": "core"}, timeout=180)
            if r.ok and r.text.strip().startswith("{"):
                break
            time.sleep(5)
        d = r.json()
        hits = d.get("resultList", {}).get("result", [])
        kept = 0
        for h in hits:
            text = (h.get("title") or "") + " " + re.sub(r"<[^>]+>", "", h.get("abstractText") or "")
            rel = bool(KEEP.search(h.get("title") or "")) or bool(re.search(r"untreat|unevaluat|uninvestigat|usual source|NHANES", text, re.I))
            kept += rel
            rows.append([name, h.get("pmid") or h.get("id"), h.get("source"), h.get("pubYear"), h.get("title"), int(rel)])
        print(f"## {name}: {d.get('hitCount')} hits, {len(hits)} read, {kept} flagged for reading", flush=True)
        time.sleep(1)
    os.makedirs("data/audit", exist_ok=True)
    with open("data/audit/prior_art_red_flag.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "id", "source", "year", "title", "flagged"])
        w.writerows(rows)
    print(len(rows), "records written")


if __name__ == "__main__":
    main()
