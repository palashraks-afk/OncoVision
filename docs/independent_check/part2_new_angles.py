"""
Part 2: two additional analyses on the repo's NHIS file (no new data, no accounts).
 A. Time trend / model decay: fit on the oldest years, test on later blocks.
 B. Where does the gain come from? Smoking-stratified decomposition (a PROXY for lung vs non-lung;
    the linked mortality file carries only the 113-group leading-cause code, ucod==2 = all malignant neoplasms).
Reuses the helper functions I wrote in part1_reproduce.py (not the repo's modelling code).
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part1_reproduce import (load, Model, auc, cal_slope, coverage_curve, logit_fit, logit_pred, rcs, ROOT, HERE)
from scipy.stats import poisson

rng = np.random.default_rng(4242)
d = load()
d["block"] = pd.cut(d.year, [1996, 1999, 2001, 2003, 2006, 2009], labels=["1997-99", "2000-01", "2002-03", "2004-06", "2007-09"])
d["block"] = np.select([d.year <= 1999, d.year <= 2001, d.year <= 2003, d.year <= 2006], ["1997-99", "2000-01", "2002-03", "2005-06", ], "2007-09")
R = {}

# ----------------------------------------------------------------- A. decay
tr = d[d.year <= 1999].reset_index(drop=True)
M0, M2 = Model(tr, 0), Model(tr, 2)
rows = []
for b in ["1997-99", "2000-01", "2002-03", "2005-06", "2007-09"]:
    t = d[d.block == b].reset_index(drop=True); y = t.y.to_numpy()
    p0, p2 = M0.predict(t), M2.predict(t)
    ex = p2.sum(); lo, hi = poisson.ppf([0.025, 0.975], y.sum()) / ex
    inv = t[t.age.between(40, 74)].reset_index(drop=True); yi = inv.y.to_numpy(); k = [int(0.2 * len(inv))]
    cov_a = coverage_curve(yi, inv.age.to_numpy(float), k)[0]; cov_r = coverage_curve(yi, M2.predict(inv), k)[0]
    rows.append(dict(block=b, n=len(t), events=int(y.sum()), OE=float(y.sum() / ex), OE_ci=[float(lo), float(hi)],
                     slope=cal_slope(y, p2)[0], auc_F0=auc(y, p0), auc_F2=auc(y, p2), gain=auc(y, p2) - auc(y, p0),
                     cov20_age=float(cov_a), cov20_F2=float(cov_r), cov_diff=float(cov_r - cov_a),
                     current_smoker_share=float((t.smoke_status == 2).mean()), never_share=float((t.smoke_status == 0).mean()),
                     mean_age=float(t.age.mean())))
R["A_decay_fit_1997_99"] = rows
# age-sex standardised observed ten-year cancer-death rate by block (direct standardisation to the pooled 40-74 age/sex mix)
s = d[d.age.between(40, 74)].copy(); s["ab"] = (s.age // 5) * 5
std = s.groupby(["ab", "male"]).size(); std = std / std.sum()
tab = s.groupby(["block", "ab", "male"]).y.mean().unstack(0)
R["A_standardised_10y_cancer_death_per_1000"] = {b: float((tab[b] * std).sum() * 1000) for b in tab.columns}
# smoking-specific trend: observed rate in current smokers 50-69, never smokers 50-69
s2 = d[d.age.between(50, 69)]
R["A_rate_per_1000_age50_69"] = {str(int(k)): {b: float(v) for b, v in s2[s2.smoke_status == k].groupby("block").y.mean().mul(1000).items()} for k in (0, 1, 2)}
# calendar-year drift in log odds after adjusting for F2 covariates (single pooled fit, bootstrap people)
def drift(dd):
    X = np.column_stack([rcs(dd.age.to_numpy(float), np.quantile(dd.age, [0.05, .275, .5, .725, .95])), dd.male,
                         (dd.smoke_status == 1), (dd.smoke_status == 2), (dd.smoke_status == 2) * dd.cpd / 20, (dd.smoke_status == 1) * dd.quit / 10,
                         (dd.bmi - 27) / 5, ((dd.bmi - 27) / 5) ** 2, (dd.year - 2003) / 10]).astype(float)
    return logit_fit(X, dd.y.to_numpy(float))[-1]
dd = d[d.age.between(35, 84)]
bs = [drift(dd.iloc[rng.integers(0, len(dd), len(dd))]) for _ in range(40)]
R["A_calendar_drift_logodds_per_decade"] = dict(est=float(drift(dd)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_boot=40)

# ----------------------------------------------------------------- B. decomposition (main split 1997-2002 / 2005-2009)
tr2, te = d[d.year <= 2002].reset_index(drop=True), d[d.year >= 2005].reset_index(drop=True)
F0, F2 = Model(tr2, 0), Model(tr2, 2)
y = te.y.to_numpy(); p0, p2 = F0.predict(te), F2.predict(te)
B = {}
lab = {0: "never", 1: "former", 2: "current"}
B["share_of_events_by_smoking"] = {lab[k]: float(y[te.smoke_status == k].sum() / y.sum()) for k in (0, 1, 2)}
B["share_of_people_by_smoking"] = {lab[k]: float((te.smoke_status == k).mean()) for k in (0, 1, 2)}
B["within_stratum_auc_F0_vs_F2"] = {}
for k in (0, 1, 2):
    m = (te.smoke_status == k).to_numpy()
    B["within_stratum_auc_F0_vs_F2"][lab[k]] = dict(n=int(m.sum()), events=int(y[m].sum()), auc_F0=auc(y[m], p0[m]), auc_F2=auc(y[m], p2[m]))
# never smokers: does anything beyond age/sex help? refit on never-smokers only
tn, en = tr2[tr2.smoke_status == 0].reset_index(drop=True), te[te.smoke_status == 0].reset_index(drop=True)
N0, N2 = Model(tn, 0), Model(tn, 2)
# (Model with model>=1 uses smoking columns which are constant zero for never smokers: harmless but singular only through ridge)
B["never_smokers_only_refit"] = dict(n_test=len(en), events=int(en.y.sum()), auc_age_sex=auc(en.y.to_numpy(), N0.predict(en)), auc_age_sex_bmi=auc(en.y.to_numpy(), N2.predict(en)))
# coverage in 40-74: where do the extra events captured by F2 (vs oldest 20%) sit?
m = te.age.between(40, 74).to_numpy(); ti = te[m].reset_index(drop=True); yi = ti.y.to_numpy(); n = len(ti); k = int(0.2 * n)
pi2 = p2[m]
sel_risk = np.zeros(n, bool); sel_risk[np.argsort(-pi2, kind="stable")[:k]] = True
# age-only selection, expected under random tie-break: weight per person
a = ti.age.to_numpy(float); order = np.argsort(-a, kind="stable"); thr = a[order[k - 1]]
wage = np.where(a > thr, 1.0, 0.0); tie = a == thr; wage[tie] = (k - (a > thr).sum()) / tie.sum()
B["top20_extra_events_by_smoking"] = {}
tot_extra = (yi * (sel_risk.astype(float) - wage)).sum()
for kk in (0, 1, 2):
    mm = (ti.smoke_status == kk).to_numpy()
    B["top20_extra_events_by_smoking"][lab[kk]] = dict(extra_events=float((yi * (sel_risk.astype(float) - wage))[mm].sum()),
                                                     invited_risk=int(sel_risk[mm].sum()), invited_age=float(wage[mm].sum()))
B["top20_extra_events_total"] = float(tot_extra)
B["top20_share_invited_risk_by_smoking"] = {lab[kk]: float(sel_risk[(ti.smoke_status == kk).to_numpy()].sum() / k) for kk in (0, 1, 2)}
# parsimonious alternatives: how much of F2 is recoverable from fewer questions?
def lite(tr_, te_, smoke_cols):
    ks = {}; bs_ = {}; out = np.zeros(len(te_))
    for sx in (0.0, 1.0):
        s_ = tr_[tr_.male == sx]; kn = np.quantile(s_.age, [0.05, .275, .5, .725, .95])
        f = lambda q: np.column_stack([rcs(q.age.to_numpy(float), kn)] + smoke_cols(q))
        b = logit_fit(f(s_), s_.y.to_numpy(float)); mm = (te_.male == sx).to_numpy(); out[mm] = logit_pred(b, f(te_[mm]))
    return out
ever = lambda q: [np.column_stack([(q.smoke_status > 0).astype(float)])]
cur = lambda q: [np.column_stack([(q.smoke_status == 1).astype(float), (q.smoke_status == 2).astype(float)])]
cur_cpd = lambda q: [np.column_stack([(q.smoke_status == 1).astype(float), (q.smoke_status == 2).astype(float), (q.smoke_status == 2) * q.cpd / 20])]
alt = {}
for nm, f in (("age+sex+ever_smoked", ever), ("age+sex+never/former/current", cur), ("age+sex+never/former/current+cigs/day", cur_cpd)):
    pp = lite(tr2, te, f); alt[nm] = dict(auc=auc(y, pp), cov20=float(coverage_curve(yi, pp[m], [k])[0]))
alt["F2 (full, my spec)"] = dict(auc=auc(y, p2), cov20=float(coverage_curve(yi, pi2, [k])[0]))
alt["age+sex (F0)"] = dict(auc=auc(y, p0), cov20=float(coverage_curve(yi, p0[m], [k])[0]))
B["parsimony_ladder"] = alt
R["B_decomposition"] = B
json.dump(R, open(os.path.join(HERE, "part2_result.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
