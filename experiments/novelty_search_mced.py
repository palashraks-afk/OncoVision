"""
Recorded novelty search for the MCED-triage angle (Europe PMC REST API). Writes data/audit/novelty_mced.csv.
Run:  python experiments/novelty_search_mced.py
"""
import csv
import os

import requests

Q = {
    "Q1 MCED + risk-based eligibility": '("multi-cancer early detection" OR "multicancer early detection" OR MCED OR Galleri) AND ("risk-based" OR "risk stratif*" OR "high-risk" OR "risk-adapted" OR enrich*)',
    "Q2 MCED + questionnaire / risk factors / triage": '("multi-cancer early detection" OR "multicancer early detection" OR MCED OR Galleri) AND (questionnaire OR "risk factor*" OR triage OR "pre-screen*" OR "two-step")',
    "Q3 MCED + positive predictive value + population risk": '("multi-cancer early detection" OR "multicancer early detection" OR MCED OR Galleri) AND ("positive predictive value" OR PPV) AND (prevalence OR "population risk" OR incidence)',
    "Q4 MCED + survey / national cohort / mortality linkage": '("multi-cancer early detection" OR "multicancer early detection" OR MCED OR Galleri) AND (NHIS OR NHANES OR "National Health Interview Survey" OR "linked mortality" OR "national cohort")',
    "Q5 MCED + Medicare eligibility / allocation / budget": '("multi-cancer early detection" OR "multicancer early detection" OR MCED OR Galleri) AND (Medicare OR allocation OR budget OR "fixed number")',
}
rows = []
for name, q in Q.items():
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": q, "format": "json", "pageSize": 200, "resultType": "lite"}, timeout=60).json()
    hits = r.get("resultList", {}).get("result", [])
    print(f"{name}: {r.get('hitCount')} hits")
    for h in hits:
        rows.append([name, h.get("pmid") or h.get("id"), h.get("pubYear"), h.get("title")])
os.makedirs("data/audit", exist_ok=True)
with open("data/audit/novelty_mced.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query", "id", "year", "title"])
    w.writerows(rows)
print(len(rows), "rows written")
