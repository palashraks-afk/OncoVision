"""
NHIS Sample Adult files 1997-2009 with EVERY numeric questionnaire variable kept (the earlier cohort builder kept
about 20), linked to the National Death Index through 2019, for the discovery screen in
docs/DISCOVERY_SCREEN_PREREG.md.

Per year this writes data/audit/nhis_full/{year}.csv.gz: pid-free, with the adult variables, the person-file
variables used for race and ethnicity, and the mortality fields. Layouts come from NCHS's own SAS programs.
Character variables are skipped. Reuses the helpers in fetch_nhis_cancer_risk.py.

Run:  python fetch_nhis_full.py            (about an hour; resumes where it stopped)
"""

import io
import os
import zipfile

import pandas as pd

from fetch_nhis_cancer_risk import CODE, DATA, LMF, PERSON_VARS, get, key, parse_layout

OUT = os.path.join("data", "audit", "nhis_full")
YEARS = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009]


def parse_layout_types(txt):
    """Like parse_layout but also reports which variables are character ($)."""
    import re
    best, chars = {}, set()
    for m0 in re.finditer(r"\bINPUT\b", txt):
        j = txt.find(";", m0.end())
        out, ch = {}, set()
        for m in re.finditer(r"([A-Z][A-Z0-9_]*)\s+(\$\s*)?(\d+)\s*-\s*(\d+)(?:\s+\.(\d))?", txt[m0.end():j]):
            out[m.group(1)] = (int(m.group(3)), int(m.group(4)), int(m.group(5) or 0))
            if m.group(2):
                ch.add(m.group(1))
        if len(out) > len(best):
            best, chars = out, ch
    return best, chars


def read_all(raw, layout, chars, keep_extra=()):
    use = [v for v in layout if v not in chars or v in keep_extra]
    cols = [(layout[v][0] - 1, layout[v][1]) for v in use]
    df = pd.read_fwf(io.BytesIO(raw), colspecs=cols, names=use, dtype=str)
    for v in use:
        if v in keep_extra:
            continue
        df[v] = pd.to_numeric(df[v], errors="coerce")
        if layout[v][2]:
            df[v] = df[v] / (10 ** layout[v][2])
    return df


def one_year(y):
    lay_a, ch_a = parse_layout_types(get(CODE.format(y=y, f="samadult")).decode("latin-1"))
    lay_p = parse_layout(get(CODE.format(y=y, f="personsx")).decode("latin-1"))
    if y == 2003:
        raw_a = get("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2003/samadult/SAMADULT.DAT")
    else:
        raw_a = zipfile.ZipFile(io.BytesIO(get(DATA.format(y=y, f="samadult.zip")))).read(zipfile.ZipFile(io.BytesIO(get(DATA.format(y=y, f="samadult.zip")))).namelist()[0])
    zp = zipfile.ZipFile(io.BytesIO(get(DATA.format(y=y, f="personsx.zip"))))
    raw_p = zp.read(zp.namelist()[0])
    keycols = ("HHX", "FMX", "FPX", "PX")
    a = read_all(raw_a, lay_a, ch_a, keep_extra=keycols)
    use_p = [v for v in PERSON_VARS if v in lay_p]
    p = pd.read_fwf(io.BytesIO(raw_p), colspecs=[(lay_p[v][0] - 1, lay_p[v][1]) for v in use_p], names=use_p, dtype=str)
    a["k"], p["k"] = key(a), key(p)
    p = p.drop(columns=[c for c in ("SRVY_YR", "HHX", "FMX", "FPX", "PX") if c in p.columns]).drop_duplicates("k")
    for c in p.columns:
        if c != "k":
            p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.rename(columns={c: "P_" + c for c in p.columns if c != "k"})
    df = a.merge(p, on="k", how="left")
    df["pid"] = df["SRVY_YR"].astype(int).astype(str) + df["k"]
    mort = pd.read_fwf(io.BytesIO(get(LMF.format(y=y))), colspecs=[(0, 14), (14, 15), (15, 16), (16, 19), (21, 22), (22, 26)], names=["pid", "elig", "died", "ucod", "dodq", "dody"], dtype=str)
    df = df.merge(mort, on="pid", how="left").drop(columns=[c for c in ("HHX", "FMX", "FPX", "PX") if c in df.columns])
    df["year"] = y
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    for y in YEARS:
        f = os.path.join(OUT, f"{y}.csv.gz")
        if os.path.exists(f):
            print(y, "already done", flush=True)
            continue
        df = one_year(y)
        df.to_csv(f, index=False, compression="gzip")
        print(f"{y}: {len(df):,} adults, {df.shape[1]} columns, linked {df['died'].notna().mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
