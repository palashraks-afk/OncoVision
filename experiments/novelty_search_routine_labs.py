"""
Recorded literature search (Europe PMC, titles and abstracts only) for the routine-lab red-flag angle:
abnormal results on ordinary lab reports that guidelines say should prompt a cancer work-up, how common they are,
whether people know, whether they are followed up, and whether they predict cancer.
Writes data/audit/novelty_routine_labs.csv.

Run:  python experiments/novelty_search_routine_labs.py
"""
import csv
import os
import time

import requests

B = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def ta(x):
    return f"(TITLE:({x}) OR ABSTRACT:({x}))"


IDA = '"iron deficiency anemia" OR "iron deficiency anaemia" OR "iron-deficiency anemia" OR "iron-deficiency anaemia"'
FLAG = 'anemia OR anaemia OR thrombocytosis OR hypercalcemia OR hypercalcaemia OR "alkaline phosphatase" OR "raised platelet*" OR "low haemoglobin" OR "low hemoglobin"'
Q = {
    "Q1 IDA follow-up / GI investigation rates":
        ta(IDA) + " AND " + ta('follow-up OR investigat* OR endoscop* OR colonoscop* OR "diagnostic delay" OR missed') + " AND " + ta("cancer OR malignan*"),
    "Q2 pre-diagnostic routine blood abnormalities":
        ta('pre-diagnostic OR prediagnostic OR "before diagnosis" OR "prior to diagnosis" OR "diagnostic window"') + " AND " + ta(FLAG + ' OR "blood test*" OR "blood count"') + " AND " + ta("cancer"),
    "Q3 population prevalence of unrecognised red-flag labs":
        ta('NHANES OR "nationally representative" OR "population-based"') + " AND " + ta(FLAG) + " AND " + ta('undiagnosed OR unrecognized OR unrecognised OR unaware OR "red flag" OR actionable OR untreated'),
    "Q4 abnormal test result follow-up failure and cancer":
        ta('"missed opportunit*" OR "test result follow-up" OR "abnormal test result*" OR "lack of follow-up" OR "failure to follow"') + " AND " + ta("cancer") + " AND " + ta('anemia OR anaemia OR laboratory OR "blood test*" OR platelet* OR "iron deficiency"'),
    "Q5 NHANES anaemia / iron and cancer mortality":
        ta('NHANES OR "National Health and Nutrition"') + " AND " + ta('anemia OR anaemia OR "iron deficiency" OR hemoglobin OR ferritin OR "transferrin saturation" OR "serum iron"') + " AND " + ta('"cancer mortality" OR "cancer death" OR "cancer-specific mortality"'),
    "Q6 red-flag lab cancer yield in unselected population":
        ta(FLAG) + " AND " + ta('"positive predictive value" OR "cancer risk" OR "risk of cancer"') + " AND " + ta('"primary care" OR "general population" OR cohort OR "population-based"'),
    "Q7 national count of people needing cancer work-up from routine labs":
        ta('million OR "national estimate*" OR prevalence') + " AND " + ta(IDA) + " AND " + ta('colonoscopy OR "gastrointestinal evaluation" OR endoscopy OR "GI evaluation"'),
}


def main():
    rows = []
    for name, q in Q.items():
        for attempt in range(3):
            r = requests.get(B, params={"query": q, "format": "json", "pageSize": 200, "resultType": "lite"}, timeout=120)
            if r.ok and r.text.strip().startswith("{"):
                break
            time.sleep(3)
        d = r.json()
        hits = d.get("resultList", {}).get("result", [])
        print(f"## {name}: {d.get('hitCount')} hits (first {len(hits)} kept)")
        for h in hits[:12]:
            print("   ", h.get("pubYear"), h.get("pmid") or h.get("id"), (h.get("title") or "")[:140])
        rows += [[name, h.get("pmid") or h.get("id"), h.get("pubYear"), h.get("title")] for h in hits]
        time.sleep(1)
    os.makedirs("data/audit", exist_ok=True)
    with open("data/audit/novelty_routine_labs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "id", "year", "title"])
        w.writerows(rows)
    print(len(rows), "rows written")


if __name__ == "__main__":
    main()
