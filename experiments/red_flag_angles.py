"""
Angles A1, A3 and A4 for the red-flag-labs study (Addenda 6 and 7 of docs/RED_FLAG_LABS_PREREG.md, committed first).
NHANES 2017-March 2020, adults 50+, survey-weighted with design-based intervals. A2 was dropped (see Addendum 7).

Run:  python experiments/red_flag_angles.py
"""

import json
import os
import re

import numpy as np
import pandas as pd

from red_flag_labs import flags, ratio_est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "data", "audit")
OUT = os.path.join(ROOT, "experiments", "red_flag_angles_result.json")
IRON_RX = r"FERROUS|FERRIC|IRON|FERUMOXYTOL"      # whole words: a bare "IRON" also matches busPIRONE and spIRONolactone
DRUGS = {
    "blood_thinner": "WARFARIN|APIXABAN|RIVAROXABAN|DABIGATRAN|EDOXABAN|ENOXAPARIN|HEPARIN|CLOPIDOGREL|PRASUGREL|TICAGRELOR|DIPYRIDAMOLE|ASPIRIN",
    "nsaid": "IBUPROFEN|NAPROXEN|MELOXICAM|DICLOFENAC|CELECOXIB|INDOMETHACIN|NABUMETONE|ETODOLAC|KETOROLAC|PIROXICAM",
    "acid_reducer_ppi": "OMEPRAZOLE|PANTOPRAZOLE|ESOMEPRAZOLE|LANSOPRAZOLE|RABEPRAZOLE|DEXLANSOPRAZOLE",
}


def rx_flags():
    rx = pd.read_csv(os.path.join(AUDIT, "P_RXQ_RX.csv.gz"))
    name = rx["RXDDRUG"].astype(str).str.replace(r"^b['\"]|['\"]$", "", regex=True).str.upper()
    out = pd.DataFrame({"SEQN": rx["SEQN"], "iron_rx": name.str.contains(IRON_RX, regex=True)})
    for k, pat in DRUGS.items():
        out[k] = name.str.contains(pat, regex=True)
    print("distinct iron prescription names:", sorted(set(name[out["iron_rx"]]))[:20])
    return out.groupby("SEQN").max().reset_index()


def est(y, x, d):
    r, ci = ratio_est(np.asarray(y, float), np.asarray(x, float), d["WTMECPRP"].to_numpy(), d["SDMVSTRA"].to_numpy(), d["SDMVPSU"].to_numpy())
    return {"est": r, "ci": ci}


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "red_flag_current.csv.gz"))
    d = d[(d["RIDAGEYR"] >= 50) & d["LBXHGB"].notna() & d["LBXFER"].notna() & (d["WTMECPRP"] > 0)].copy()
    d["male"] = (d["RIAGENDR"] == 1).astype(int)
    d = d[~((d["male"] == 0) & (d["RHQ031"] == 1))]
    anaemia, ida, _ = flags(d["LBXHGB"].to_numpy(), d["LBXFER"].to_numpy(), d["male"].to_numpy())
    d["ida"], d["anaemia"] = ida, anaemia
    tot = pd.read_csv(os.path.join(AUDIT, "P_DSQTOT.csv.gz"))[["SEQN", "DSQTIRON"]]
    hiq = pd.read_csv(os.path.join(AUDIT, "P_HIQ.csv.gz"))[["SEQN", "HIQ032A", "HIQ032B", "HIQ032D"]]
    d = d.merge(rx_flags(), on="SEQN", how="left").merge(tot, on="SEQN", how="left").merge(hiq, on="SEQN", how="left")
    for c in ["iron_rx"] + list(DRUGS):
        d[c] = d[c].fillna(False).astype(bool)
    mg = d["DSQTIRON"].fillna(0).to_numpy()
    asked = d["MCQ053"].eq(1).to_numpy()
    known = d["MCQ053"].isin([1, 2]).to_numpy()
    base = d["ida"].to_numpy() & known
    res = {"n_50plus": int(len(d)), "ida_known_n": int(base.sum())}

    # A1: untreated under progressively broader definitions of "treated"
    defs = {"survey_question_only": asked,
            "plus_iron_prescription": asked | d["iron_rx"].to_numpy(),
            "plus_treatment_dose_supplement": asked | d["iron_rx"].to_numpy() | (mg >= 45),
            "plus_any_iron_supplement": asked | d["iron_rx"].to_numpy() | (mg > 0)}
    res["A1_untreated"] = {}
    for k, treated in defs.items():
        e = est(base & ~treated, base, d)
        e["untreated_n"] = int((base & ~treated).sum())
        res["A1_untreated"][k] = e
    broad = defs["plus_any_iron_supplement"]
    pop = est(base, known, d)["est"] * float(d["WTMECPRP"].sum())
    res["A1_untreated_millions_broadest"] = round(res["A1_untreated"]["plus_any_iron_supplement"]["est"] * pop / 1e6, 2)
    res["bar_A1_met"] = bool(res["A1_untreated"]["plus_any_iron_supplement"]["ci"][0] > 0.5)

    # A3: untreated (broadest definition) by insurance, not exclusive
    groups = {"private": d["HIQ032A"].eq(1), "medicare": d["HIQ032B"].eq(2), "medicaid": d["HIQ032D"].eq(4), "uninsured": d["HIQ011"].eq(2)}
    res["A3_by_payer"] = {}
    unt = base & ~broad
    unt_total = unt.sum()
    for k, g in groups.items():
        g = g.to_numpy()
        res["A3_by_payer"][k] = {"share_of_untreated_covered": est(unt & g, unt, d), "untreated_n": int((unt & g).sum()),
                                 "untreated_share_within_group": est(base & g & ~broad, base & g, d) if (base & g).sum() >= 15 else None,
                                 "ida_n_in_group": int((base & g).sum())}
    res["A3_untreated_total_n"] = int(unt_total)

    # A4: medicines that can cause iron loss or reduce iron uptake, IDA against no anaemia
    res["A4_medicines"] = {}
    none = ~d["anaemia"].to_numpy()
    anyc = (d["blood_thinner"] | d["nsaid"] | d["acid_reducer_ppi"]).to_numpy()
    for k, v in [(k, d[k].to_numpy()) for k in DRUGS] + [("any_of_these", anyc)]:
        res["A4_medicines"][k] = {"ida": est(d["ida"].to_numpy() & v, d["ida"].to_numpy(), d), "no_anaemia": est(none & v, none, d)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
