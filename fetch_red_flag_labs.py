"""
Data for the red-flag-labs study (docs/RED_FLAG_LABS_PREREG.md).

  1. NHANES 2017-March 2020 pre-pandemic (current national picture): demographics and design, blood count, ferritin,
     anaemia treatment, cancer history, insurance, usual care, health-care visits, menstruation.
  2. NHANES 2001-2002 with ferritin and deaths through 2019: already in data/nhanes_navigator.csv.gz (taken from there).
  3. NHANES III (1988-1994) with ferritin and deaths through 2019: ferritin and MCV read from the NHANES III lab file
     (positions from CDC's own lab.sas: FRP 1463-1466, MVPSI 1340-1344), joined to data/nhanes3_markers.csv.gz.

Downloads about 45 MB in total. Writes data/red_flag_current.csv.gz and data/red_flag_followup.csv.gz.

Run:  python fetch_red_flag_labs.py
"""

import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
P = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{f}.xpt"
NH3_LAB = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=1800, context=_ctx).read()


def xpt(name, cols):
    df = pd.read_sas(io.BytesIO(get(P.format(f=name))), format="xport")
    keep = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    print(f"  {name}: {len(df):,} rows; missing columns {missing}", flush=True)
    return df[["SEQN"] + [c for c in keep if c != "SEQN"]]


def current():
    print("NHANES 2017-March 2020 pre-pandemic", flush=True)
    d = xpt("P_DEMO", ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "DMDEDUC2", "WTMECPRP", "SDMVSTRA", "SDMVPSU"])
    for name, cols in (("P_CBC", ["LBXHGB", "LBXMCVSI", "LBXPLTSI"]), ("P_FERTIN", ["LBXFER"]), ("P_MCQ", ["MCQ053", "MCQ220"]),
                       ("P_HIQ", ["HIQ011"]), ("P_HUQ", ["HUQ030", "HUQ051"]), ("P_RHQ", ["RHQ031"]), ("P_SMQ", ["SMQ020", "SMQ040"])):
        d = d.merge(xpt(name, cols), on="SEQN", how="left")
    d.to_csv(os.path.join("data", "red_flag_current.csv.gz"), index=False, compression="gzip")
    print(f"  wrote data/red_flag_current.csv.gz ({len(d):,} rows)", flush=True)


def followup():
    print("NHANES 2001-2002 (from nhanes_navigator) and NHANES III", flush=True)
    nav = pd.read_csv(os.path.join("data", "nhanes_navigator.csv.gz"))
    b = nav[nav["cycle"] == "2001-2002"].copy()
    b = pd.DataFrame({"dataset": "NHANES 2001-2002", "SEQN": b["SEQN"], "age": b["age"], "male": (b["sex"] == "male").astype(float),
                      "smoked": b["ever_smoked"], "hemoglobin": b["hemoglobin"], "mcv": b["mcv"], "platelets": b["platelets"], "ferritin": b["ferritin"],
                      "prior_cancer": 0, "died": b["died"], "ucod": b["ucod"], "months": b["months"], "months_source": "exam"})
    raw = get(NH3_LAB)
    lab = pd.read_fwf(io.BytesIO(raw), colspecs=[(0, 5), (1462, 1466), (1339, 1344)], names=["SEQN", "FRP", "MVPSI"])
    for c in ("FRP", "MVPSI"):
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["FRP"] = lab["FRP"].mask(lab["FRP"] >= 8888)
    lab["MVPSI"] = lab["MVPSI"].mask(lab["MVPSI"] >= 888)
    print(f"  NHANES III lab: {len(lab):,} rows; ferritin present {lab['FRP'].notna().mean():.0%}; ferritin median {lab['FRP'].median():.0f}; MCV median {lab['MVPSI'].median():.1f}", flush=True)
    m3 = pd.read_csv(os.path.join("data", "nhanes3_markers.csv.gz")).merge(lab, on="SEQN", how="left")
    # 416 NHANES III decedents have no months-from-exam; fall back to months from interview (the exam usually followed within
    # weeks), flagged in months_source. Found by the independent re-check (docs/independent_check/red_flag/).
    mort = pd.read_fwf(io.BytesIO(get("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_III_MORT_2019_PUBLIC.dat")),
                       colspecs=[(0, 6), (42, 45), (45, 48)], names=["SEQN", "PERMTH_INT", "PERMTH_EXM"])
    for c in ("PERMTH_INT", "PERMTH_EXM"):
        mort[c] = pd.to_numeric(mort[c], errors="coerce")
    m3 = m3.merge(mort, on="SEQN", how="left")
    m3["months_num"] = pd.to_numeric(m3["months"], errors="coerce")
    m3["months_source"] = np.where(m3["months_num"].notna(), "exam", np.where(m3["PERMTH_INT"].notna(), "interview", "none"))
    m3["months"] = m3["months_num"].fillna(m3["PERMTH_INT"])
    c = pd.DataFrame({"dataset": "NHANES III 1988-1994", "SEQN": m3["SEQN"], "age": m3["age"], "male": m3["male"], "smoked": m3["smoked"],
                      "hemoglobin": m3["hemoglobin"], "mcv": m3["MVPSI"], "platelets": m3["platelets"], "ferritin": m3["FRP"],
                      "prior_cancer": m3["prior_cancer"], "died": m3["died"], "ucod": m3["ucod"], "months": m3["months"], "months_source": m3["months_source"]})
    out = pd.concat([b, c], ignore_index=True)
    out.to_csv(os.path.join("data", "red_flag_followup.csv.gz"), index=False, compression="gzip")
    print(out.groupby("dataset")[["age", "hemoglobin", "mcv", "ferritin", "platelets", "died", "months"]].median().round(1).to_string())
    print(out.groupby("dataset").agg(n=("SEQN", "size"), ferritin=("ferritin", lambda s: int(s.notna().sum()))).to_string())


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    current()
    followup()
