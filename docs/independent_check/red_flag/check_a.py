"""Independent re-implementation of Part A (societal, NHANES 2017-Mar 2020)."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("data/red_flag_current.csv.gz")
for c in d.columns:
    if d[c].dtype.kind == "f":
        d.loc[d[c].abs() < 1e-30, c] = 0.0  # SAS tiny-zero artefact (5.4e-79)

def design_mean(y, dom, w, st, psu):
    """Weighted domain mean, Taylor linearisation over full design (domain estimation)."""
    y = np.where(dom, y, 0.0); wd = np.where(dom, w, 0.0)
    W = wd.sum(); r = (wd * y).sum() / W
    z = wd * (y - r) / W
    df = pd.DataFrame({"z": z, "h": st, "p": psu}).groupby(["h", "p"]).z.sum().reset_index()
    v = 0.0
    for h, g in df.groupby("h"):
        n = len(g)
        if n > 1: v += n / (n - 1) * ((g.z - g.z.mean()) ** 2).sum()
    dfree = df.shape[0] - df.h.nunique()
    se = np.sqrt(v)
    return r, se, dfree, W

def rep(name, y, dom):
    r, se, dfree, W = design_mean(y, dom, d.WTMECPRP.values, d.SDMVSTRA.values, d.SDMVPSU.values)
    t = stats.t.ppf(.975, dfree)
    print(f"{name}: {100*r:.2f}% z-CI {100*(r-1.96*se):.2f}-{100*(r+1.96*se):.2f} t({dfree})-CI {100*(r-t*se):.2f}-{100*(r+t*se):.2f} n={int(dom.sum())} Wpop={W/1e6:.1f}M")
    return r

male = d.RIAGENDR == 1
age50 = d.RIDAGEYR >= 50
menst = (d.RIAGENDR == 2) & (d.RHQ031 == 1)
haslab = d.LBXHGB.notna() & d.LBXFER.notna()
pop = age50 & ~menst & haslab
print("weights: all 50+ MEC", round(d.loc[age50 & (d.WTMECPRP > 0), "WTMECPRP"].sum() / 1e6, 1),
      "| 50+ not menstruating", round(d.loc[age50 & ~menst, "WTMECPRP"].sum() / 1e6, 1),
      "| analytic (labs)", round(d.loc[pop, "WTMECPRP"].sum() / 1e6, 1))
print("women 50+ RHQ031 dist", d.loc[age50 & ~male, "RHQ031"].value_counts(dropna=False).to_dict())
print("50+ with Hb but no ferritin", int((age50 & d.LBXHGB.notna() & d.LBXFER.isna()).sum()))
anaem = np.where(male, d.LBXHGB < 13, d.LBXHGB < 12)
ida = pop & anaem & (d.LBXFER < 45)
nia = pop & anaem & (d.LBXFER >= 45)
p = rep("IDA prevalence", ida.astype(float).values, pop.values)
Wall = d.loc[age50 & ~menst & (d.WTMECPRP > 0), "WTMECPRP"].sum()
print(f"  count via analytic W: {p*d.loc[pop,'WTMECPRP'].sum()/1e6:.2f}M; via all-50+ MEC W: {p*Wall/1e6:.2f}M")
rep("non-iron anaemia", nia.astype(float).values, pop.values)
rep("IDA men", ida.astype(float).values, (pop & male).values)
rep("IDA women", ida.astype(float).values, (pop & ~male).values)
rep("IDA no prior cancer", ida.astype(float).values, (pop & (d.MCQ220 != 1)).values)
print("MCQ053 among IDA", d.loc[ida, "MCQ053"].value_counts(dropna=False).to_dict())
v53 = ida & d.MCQ053.isin([1, 2])
rep("IDA untreated (MCQ053==2 | 1,2)", (d.MCQ053 == 2).astype(float).values, v53.values)
un = ida & (d.MCQ053 == 2)
print("untreated n", int(un.sum()), "HUQ030", d.loc[un, "HUQ030"].value_counts().to_dict(), "HUQ051", d.loc[un, "HUQ051"].value_counts().sort_index().to_dict(), "HIQ011", d.loc[un, "HIQ011"].value_counts().to_dict())
rep("  usual place (1,3 | 1,2,3)", d.HUQ030.isin([1, 3]).astype(float).values, (un & d.HUQ030.isin([1, 2, 3])).values)
rep("  2+ visits (>=2 | 0-8)", (d.HUQ051 >= 2).astype(float).values, (un & d.HUQ051.between(0, 8)).values)
rep("  1+ visits (>=1 | 0-8)", (d.HUQ051 >= 1).astype(float).values, (un & d.HUQ051.between(0, 8)).values)
rep("  4+ visits (>=3 | 0-8)", (d.HUQ051 >= 3).astype(float).values, (un & d.HUQ051.between(0, 8)).values)
rep("  insured (1 | 1,2)", (d.HIQ011 == 1).astype(float).values, (un & d.HIQ011.isin([1, 2])).values)
rep("  usual place among ALL 50+ (benchmark)", d.HUQ030.isin([1, 3]).astype(float).values, (pop & d.HUQ030.isin([1, 2, 3])).values)
# Treated-anaemia share among all-anaemia, and ferritin<45 among treated: does treatment hide IDA?
rep("MCQ053==1 among all 50+ pop", (d.MCQ053 == 1).astype(float).values, (pop & d.MCQ053.isin([1, 2])).values)
tr = pop & (d.MCQ053 == 1)
print("treated 50+ n", int(tr.sum()), "of whom anaemic", int((tr & anaem).sum()), "IDA", int((tr & ida).sum()))
rep("untreated, ferritin<30 IDA", (d.MCQ053 == 2).astype(float).values, (pop & anaem & (d.LBXFER < 30) & d.MCQ053.isin([1, 2])).values)
