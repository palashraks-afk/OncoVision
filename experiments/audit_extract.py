"""
Step 1 of the NHANES machine-learning cancer audit (docs/AUDIT_PREREG.md).

Runs the fixed Europe PMC search, downloads open-access full text, and applies the cue
rules written in the pre-registration. The output is a first-pass coding that is then
adjudicated by reading the matched sentences (experiments/audit_adjudicate.py holds the
adjudicated codes). Nothing here decides inclusion.

Run:  python experiments/audit_extract.py
Writes data/audit/records.csv, data/audit/auto_codes.csv, data/audit/snippets.json
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "audit")
os.makedirs(OUT, exist_ok=True)
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
QUERY = ('(ABSTRACT:"NHANES" OR ABSTRACT:"National Health and Nutrition Examination Survey") AND '
         '(ABSTRACT:"machine learning" OR ABSTRACT:"deep learning" OR ABSTRACT:"random forest" OR ABSTRACT:"XGBoost" OR ABSTRACT:"gradient boosting" OR ABSTRACT:"neural network" OR ABSTRACT:"support vector") AND '
         '(ABSTRACT:cancer OR ABSTRACT:carcinoma OR ABSTRACT:neoplasm OR ABSTRACT:malignan* OR ABSTRACT:tumor OR ABSTRACT:tumour) AND '
         '(PUB_YEAR:[2015 TO 2026]) AND SRC:MED')

CUES = {
    "cross_sectional": r"cross-sectional|history of cancer|self-reported cancer|cancer status|ever (been )?told|MCQ220|cancer prevalence",
    "mortality": r"mortality|death|survival|National Death Index|\bNDI\b|follow-up",
    "random_split": r"randomly (split|divided|partitioned|assigned|separated)|train(ing)?[- /]*(and )?(test|testing|validation)[- ]*(set|split|cohort|data)|\b70[:/ -]+30\b|\b80[:/ -]+20\b|\b75[:/ -]+25\b|k-fold|\d+-fold|cross-validation",
    "external_validation": r"external validation|independent (validation|cohort|dataset|test)|validation cohort|temporal validation|held-out (cycle|year|survey)|(different|later|subsequent|separate) (NHANES )?(cycle|survey cycle|wave)s?|UK Biobank|\bSEER\b|\bPLCO\b|\bCHARLS\b|\bNHIS\b|hospital (cohort|dataset|data)",
    "baseline_agesex": r"age[- ]only|(age|age and sex|age, sex)[^.]{0,40}\b(alone|only)\b|demographic[s]?[- ]only|baseline (model|logistic)|null model|reference model|conventional (model|risk)",
    "logistic_comparator": r"logistic regression",
    "survey_weights": r"sampling weight|survey weight|WTMEC|WTINT|svydesign|complex survey|survey-weighted|weighted (analysis|logistic|model)",
    "calibration": r"calibration|Brier|Hosmer",
    "resampling": r"SMOTE|oversampl|undersampl|class imbalance|ADASYN|resampl",
    "missing_method": r"multiple imputation|\bMICE\b|k-?nearest|KNN imputation|missForest|median imputation|mean imputation|imputed|imputation",
    "missing_indicator": r"missing[- ](value )?indicator|missingness indicator|indicator (for|of) missing",
}
AUC_RE = re.compile(r"(?:AUC|AUROC|C-statistic|c-index|area under the (?:receiver operating characteristic|ROC)[^0-9]{0,40})[^0-9]{0,25}(0\.\d{2,3}|1\.00)", re.I)


def search():
    recs, cursor = [], "*"
    while True:
        r = requests.get(f"{BASE}/search", params={"query": QUERY, "format": "json", "pageSize": 100, "resultType": "core", "cursorMark": cursor}, timeout=90).json()
        recs += r["resultList"]["result"]
        nxt = r.get("nextCursorMark")
        if not nxt or nxt == cursor or len(recs) >= r["hitCount"]:
            break
        cursor = nxt
    return recs


def fulltext(pmcid):
    r = requests.get(f"{BASE}/{pmcid}/fullTextXML", timeout=90)
    if r.status_code != 200:
        return None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None
    parts = []
    for sec in root.iter():
        if sec.tag in ("p", "title", "label", "caption"):
            txt = "".join(sec.itertext()).strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.strip()) > 20]


def main():
    recs = search()
    print(f"{len(recs)} records from the fixed search")
    rows, codes, snippets = [], [], {}
    for i, r in enumerate(recs):
        pmid, pmcid = r.get("pmid") or r.get("id"), r.get("pmcid")
        oa = r.get("isOpenAccess") == "Y"
        abstract = re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")
        title = r.get("title") or ""
        text = None
        if oa and pmcid:
            text = fulltext(pmcid)
            time.sleep(0.3)
        rows.append({"pmid": pmid, "pmcid": pmcid, "year": r.get("pubYear"), "journal": r.get("journalTitle"), "title": title,
                     "open_access_full_text": bool(text), "abstract": abstract})
        body = text if text else abstract
        sents = sentences(body)
        c = {"pmid": pmid, "pmcid": pmcid, "full_text": bool(text)}
        snip = {}
        for k, pat in CUES.items():
            hits = [s for s in sents if re.search(pat, s, re.I)]
            c[k] = len(hits)
            snip[k] = [" ".join(h.split()[:40]) for h in hits[:3]]
        aucs = [float(x) for x in AUC_RE.findall(abstract + " " + (title or ""))]
        c["auc_abstract_max"] = max(aucs) if aucs else None
        aucs_all = [float(x) for x in AUC_RE.findall(body)]
        c["auc_any_max"] = max(aucs_all) if aucs_all else None
        codes.append(c)
        snippets[str(pmid)] = {"title": title, "snips": snip}
        print(f"  {i + 1:>3}/{len(recs)} {str(r.get('pubYear'))} {'FT ' if text else 'abs'} {title[:80].encode('ascii', 'replace').decode()}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "records.csv"), index=False)
    pd.DataFrame(codes).to_csv(os.path.join(OUT, "auto_codes.csv"), index=False)
    json.dump(snippets, open(os.path.join(OUT, "snippets.json"), "w"), indent=1)
    print(f"\nfull text for {sum(1 for r in rows if r['open_access_full_text'])} of {len(rows)}")


if __name__ == "__main__":
    main()
