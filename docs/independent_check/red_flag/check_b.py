"""Independent re-implementation of Part B (follow-up, pooled NHANES III + 2001-02). 300 bootstrap resamples."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

b = pd.read_csv("data/red_flag_followup.csv.gz", low_memory=False)
b["months"] = pd.to_numeric(b.months, errors="coerce")
b = b[(b.age >= 50) & (b.age <= 84) & (b.prior_cancer == 0)]
b = b.dropna(subset=["hemoglobin", "ferritin", "smoked", "died"])
print("died w/o months:", int(((b.died == 1) & b.months.isna()).sum()), "alive w/o months:", int(((b.died == 0) & b.months.isna()).sum()))
b = b[b.months.notna()]
H = 60
b = b[~((b.died == 0) & (b.months < H))]
anaem = np.where(b.male == 1, b.hemoglobin < 13, b.hemoglobin < 12)
b["ida"] = (anaem & (b.ferritin < 45)).astype(int)
b["nia"] = (anaem & (b.ferritin >= 45)).astype(int)
b["ca"] = ((b.died == 1) & (b.ucod == 2) & (b.months <= H)).astype(int)
b["oth"] = ((b.died == 1) & (b.ucod != 2) & (b.months <= H)).astype(int)
b["d01"] = (b.dataset == "NHANES 2001-2002").astype(int)
print("n", len(b), "ida", b.ida.sum(), "nia", b.nia.sum(), "ca", b.ca.sum(), "ca_ida", b.loc[b.ida == 1, "ca"].sum(), "oth_ida", b.loc[b.ida == 1, "oth"].sum())

def rcs(x, knots):
    k = knots; out = [x]
    def p(u): return np.maximum(u, 0) ** 3
    for j in range(len(k) - 2):
        out.append((p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2]) + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2)
    return np.column_stack(out)
KN = np.percentile(b.age, [5, 35, 65, 95])

def fit(df, y):
    X = np.column_stack([df.ida, df.nia, rcs(df.age.values, KN), df.male, df.smoked, df.d01])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1); sd = df[["ida", "nia"]].std().values
    m = LogisticRegression(C=1e8, max_iter=2000).fit(X, df[y])
    return m.coef_[0][:2] / sd  # log-ORs for ida, nia

def stats_(df):
    ci, cn = fit(df, "ca"); oi, on = fit(df, "oth")
    return np.array([ci, cn, oi, ci - oi, ci - cn])

est = stats_(b)
rng = np.random.default_rng(1); bs = []
for _ in range(300):
    # resample within dataset
    idx = np.concatenate([rng.choice(g, len(g)) for g in b.groupby("dataset").indices.values()])
    bs.append(stats_(b.iloc[idx]))
bs = np.array(bs)
names = ["OR cancer IDA", "OR cancer NIA", "OR other IDA", "ratio ORs cancer/other IDA", "ratio ORs IDA/NIA cancer"]
for i, n in enumerate(names):
    lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
    print(f"{n}: {np.exp(est[i]):.2f} ({np.exp(lo):.2f}-{np.exp(hi):.2f})")
# sensitivity: NHANES III only, dropping first 2 years (reverse causation / prevalent cancer)
for lab, sub in [("NHANES III only", b[b.d01 == 0]), ("pooled, men only", b[b.male == 1])]:
    sub = sub.copy(); sub["d01"] = sub.d01 if sub.d01.nunique() > 1 else 0
    e = stats_(sub); print(lab, "OR ca IDA %.2f, OR oth IDA %.2f, ratio %.2f, ca_ida n=%d" % (np.exp(e[0]), np.exp(e[2]), np.exp(e[3]), sub.loc[sub.ida == 1, "ca"].sum()))
ida = b[b.ida == 1]
print("IDA cancer deaths by year of death:", np.histogram(ida.loc[ida.ca == 1, "months"], [0, 12, 24, 36, 48, 60])[0])
print("IDA men/women:", ida.male.value_counts().to_dict(), "cancer deaths men/women:", ida.groupby("male").ca.sum().to_dict())
