"""
Recorded novelty search for the cancer-risk-age paper (docs/CANCER_AGE_PREREG.md).

Fixed Europe PMC queries, all years, for work that expresses cancer (or all-cause / smoking-related)
risk as an equivalent age, or that uses free information to set when screening should start. Every
result is written to data/audit/novelty_cancer_age.csv with the query that found it, and the closest
are read. The count per query is printed so the search is reproducible.

Run:  python experiments/novelty_search_age.py
"""

import os
import re

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CANCER = "(cancer OR neoplasm OR carcinoma OR malignan*)"
QUERIES = {
    "Q1 risk advancement period + cancer": f'TITLE_ABS:"risk advancement period" AND TITLE_ABS:{CANCER}',
    "Q2 equivalent / risk age + cancer mortality": f'(TITLE_ABS:"risk age" OR TITLE_ABS:"risk-equivalent age" OR TITLE_ABS:"equivalent age" OR TITLE_ABS:"age equivalent" OR TITLE_ABS:"cancer age" OR TITLE_ABS:"lung age") AND TITLE_ABS:{CANCER} AND TITLE_ABS:(mortality OR death OR "10-year")',
    "Q3 all-cause or cancer risk expressed as years of age": f'TITLE_ABS:("years of age" OR "equivalent to" OR "age-equivalent" OR "risk-advancement" OR "age advancement") AND TITLE_ABS:(smoking OR obesity OR BMI) AND TITLE_ABS:{CANCER} AND TITLE_ABS:(mortality OR death)',
    "Q4 questionnaire-only cancer mortality risk score": f'TITLE_ABS:(questionnaire OR "self-reported" OR "non-laboratory" OR "simple") AND TITLE_ABS:{CANCER} AND TITLE_ABS:("risk score" OR "risk prediction" OR "prediction model") AND TITLE_ABS:(mortality OR death) AND TITLE_ABS:("general population" OR "National Health Interview Survey" OR NHIS OR NHANES OR cohort)',
    "Q5 risk-based age to start cancer screening, all cancers or free info": f'TITLE_ABS:("age to start screening" OR "starting age" OR "initial screening age" OR "risk-adapted" OR "risk-based screening") AND TITLE_ABS:{CANCER} AND TITLE_ABS:(smoking OR "risk factors" OR "risk score" OR questionnaire)',
    "Q6 NHIS linked mortality + cancer mortality + risk": f'TITLE_ABS:("National Health Interview Survey" OR NHIS) AND TITLE_ABS:("linked mortality" OR "National Death Index") AND TITLE_ABS:{CANCER} AND TITLE_ABS:(smoking OR "risk")',
}


def main():
    rows = []
    for name, q in QUERIES.items():
        r = requests.get(BASE, params={"query": q, "format": "json", "pageSize": 200, "resultType": "lite"}, timeout=120).json()
        n = r["hitCount"]
        print(f"{name}: {n} records")
        for x in r["resultList"]["result"]:
            rows.append({"query": name, "pmid": x.get("pmid") or x.get("id"), "year": x.get("pubYear"), "title": (x.get("title") or "")[:200]})
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "data", "audit", "novelty_cancer_age.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    for name in QUERIES:
        sub = df[df["query"] == name].head(12)
        print(f"\n== {name}")
        for _, x in sub.iterrows():
            print("  ", x["year"], re.sub(r"[^\x00-\x7f]", "?", x["title"])[:135])


if __name__ == "__main__":
    main()
