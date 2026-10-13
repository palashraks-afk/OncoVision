"""
Recorded novelty searches (Europe PMC) for the free-data multi-cancer risk screen: questions + routine checkup
+ phone/watch-type activity data, predicting death from any cancer. Three independent rounds with different
wording, so the novelty claim does not rest on one phrasing. Writes data/audit/novelty_freedata.csv.
Run:  python experiments/novelty_search_freedata.py
"""
import csv
import os

import requests

ACC = '(accelerometer OR accelerometry OR wearable OR "step count" OR "daily steps" OR actigraphy OR "physical activity monitor" OR smartwatch OR smartphone)'
CA = '("cancer mortality" OR "cancer death" OR "cancer-specific mortality" OR "death from cancer" OR "multi-cancer" OR "pan-cancer" OR "any cancer")'
PRED = '(prediction OR predictive OR "risk score" OR "risk model" OR discrimination OR "C-statistic" OR AUC OR "risk stratification")'
ROUNDS = {
    "R1 wearable + cancer death + prediction": f"{ACC} AND {CA} AND {PRED}",
    "R1 NHANES accelerometer + cancer mortality": '(NHANES OR "National Health and Nutrition") AND (accelerometer OR accelerometry) AND ("cancer mortality" OR "cancer death")',
    "R2 digital biomarker + cancer screening triage": '("digital biomarker*" OR "consumer wearable*" OR "passively collected" OR "mobile health data") AND cancer AND (screening OR "early detection" OR triage)',
    "R2 routine data layers + incremental value + cancer death": '("incremental value" OR "added value" OR "net reclassification" OR "data ladder" OR "layered") AND ("routine" OR "free" OR "low-cost" OR "readily available") AND ("cancer mortality" OR "cancer death")',
    "R2 unintentional weight loss + population + cancer risk prediction": '("weight loss" OR "weight change") AND ("cancer mortality" OR "cancer incidence" OR "cancer death") AND (NHANES OR "population-based" OR cohort) AND (prediction OR "risk model" OR "risk score")',
    "R3 physical activity pattern + cancer mortality + machine learning": '("activity pattern*" OR "rest-activity" OR "circadian" OR "activity fragmentation" OR "sedentary bout*") AND ("cancer mortality" OR "cancer death")',
    "R3 MCED + wearable or routine data": '("multi-cancer early detection" OR MCED OR "liquid biopsy") AND (wearable OR accelerometer OR "routine blood" OR "electronic health record" OR questionnaire) AND (triage OR "pre-screen*" OR enrich* OR "risk stratification")',
}
rows = []
for name, q in ROUNDS.items():
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": q, "format": "json", "pageSize": 300, "resultType": "lite"}, timeout=90).json()
    hits = r.get("resultList", {}).get("result", [])
    print(f"{name}: {r.get('hitCount')} hits")
    rows += [[name, h.get("pmid") or h.get("id"), h.get("pubYear"), h.get("title")] for h in hits]
os.makedirs("data/audit", exist_ok=True)
with open("data/audit/novelty_freedata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query", "id", "year", "title"])
    w.writerows(rows)
print(len(rows), "rows")
