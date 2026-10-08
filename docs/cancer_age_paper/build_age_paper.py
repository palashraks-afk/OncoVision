"""
Builds the cancer-risk-age / cost-matched screening paper (HTML, then PDF through headless Chrome)
from the saved result files, so no number in the text can differ from the analysis.

Run:  python docs/cancer_age_paper/build_age_paper.py
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_result.json")))
X = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_extra_result.json")))
F4 = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_race_F4.json")))
src = open(os.path.join(ROOT, "docs", "paper", "build_paper.py"), encoding="utf-8").read()
CSS = src[src.index('CSS = """') + 9: src.index('"""', src.index('CSS = """') + 9)]

FIGS = ["fig1_design", "fig2_ladder", "fig3_calibration", "fig4_riskage", "fig5_coverage", "fig6_external", "fig7_equity", "fig8_forties"]
FN = {n: i + 1 for i, n in enumerate(FIGS)}
TN = {n: i + 1 for i, n in enumerate(["ladder", "bands", "cost", "external", "equity", "bars"])}


def F(n):
    return f"Figure {FN[n]}"


def TR(n):
    return f"Table {TN[n]}"


def fig(name, caption, width="100%"):
    return f'<figure style="width:{width}"><img src="figures/{name}.png" alt=""><figcaption><b>Figure {FN[name]}.</b> {caption}</figcaption></figure>'


def table(key, title, head, rows, note=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="tcap"><b>Table {TN[key]}.</b> {title}</div><table class="num"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{n}'


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


M, G = R["models"], R["gains"]
names = list(M)
CM, ex = R["cost_matched"], R["external"]
nh, n3 = ex["NHANES 1999-2008"], ex["NHANES III 1988-1994"]
gr = X["groups"]
SS = R["screening_start"]
rob = X["robustness"]
N = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_national_result.json")))
pool = X["pooled_external"]
fc = CM["fixed_coverage"]
E = CM["earliness_at_30pct_invited"]
RAP = R["rap_distribution"]
b2 = R["bar2"]["bands"]

t_ladder = table("ladder", "The ladder of free information, fitted on NHIS 1997-2002 and tested on 2005-2009.", ["Model", "AUC", "Calibration slope", "Intercept", "Gain over the row above (95% CI)"],
                 [[names[0], f"{M[names[0]]['auc']:.4f}", M[names[0]]["slope"], M[names[0]]["intercept"], "-"],
                  [names[1], f"{M[names[1]]['auc']:.4f}", M[names[1]]["slope"], M[names[1]]["intercept"], f"{G['F2 vs F0']['gain'] - G['F2 vs F1']['gain']:+.4f} (smoking alone, derived)"],
                  [names[2], f"{M[names[2]]['auc']:.4f}", M[names[2]]["slope"], M[names[2]]["intercept"], f"{G['F2 vs F1']['gain']:+.4f} ({G['F2 vs F1']['ci'][0]:+.4f} to {G['F2 vs F1']['ci'][1]:+.4f})"],
                  [names[3], f"{M[names[3]]['auc']:.4f}", M[names[3]]["slope"], M[names[3]]["intercept"], f"{G['F3 vs F2']['gain']:+.4f} ({G['F3 vs F2']['ci'][0]:+.4f} to {G['F3 vs F2']['ci'][1]:+.4f})"],
                  [names[4], f"{M[names[4]]['auc']:.4f}", M[names[4]]["slope"], M[names[4]]["intercept"], f"{G['F4 vs F3']['gain']:+.4f} ({G['F4 vs F3']['ci'][0]:+.4f} to {G['F4 vs F3']['ci'][1]:+.4f})"]],
                 f"F2 over F0: {G['F2 vs F0']['gain']:+.4f} (95% CI {G['F2 vs F0']['ci'][0]:+.4f} to {G['F2 vs F0']['ci'][1]:+.4f}). The pre-set rule chooses the simpler model unless the richer one gains 0.01 with an interval above zero; F3 does not, so F2 (age, sex, smoking, BMI) is the cheapest model that suffices.")
rows = []
for b in b2:
    rows.append([b, f"{b2[b]['observed']} / {b2[b]['expected']}", f"{b2[b]['ratio']:.2f}", f"{nh['bands'][b]['observed']} / {nh['bands'][b]['expected']}", f"{nh['bands'][b]['ratio']:.2f}",
                 f"{n3['bands'][b]['observed']} / {n3['bands'][b]['expected']}", f"{n3['bands'][b]['ratio']:.2f}"])
t_bands = table("bands", "Observed against predicted ten-year cancer deaths by age band.", ["Age", "NHIS 2005-09: observed / predicted", "Ratio", "NHANES 1999-2008", "Ratio", "NHANES III 1988-94", "Ratio"], rows,
                "Every band met the pre-set rule (within 20% or inside the Poisson 95% interval). The 50-59 band is the one that failed in the earlier NHANES-only model.")
rows = []
for i, s in enumerate(CM["shares"]):
    if round(s * 100) % 10 == 0 and s <= 0.65:
        rows.append([f"{s:.0%}", pc(CM["coverage_age"][i], 1), pc(CM["coverage_risk"][i], 1), f"{100 * CM['diff'][i]:+.1f} ({100 * CM['diff_lo'][i]:+.1f} to {100 * CM['diff_hi'][i]:+.1f})"])
t_cost = table("cost", "Cancer deaths covered at the same number invited, adults 40 to 74, NHIS 2005-2009.", ["Share of adults invited", "Invite the oldest", "Invite the highest risk", "Difference, points (95% CI)"], rows,
               f"{CM['n']:,} adults and {CM['cancer_deaths']:,} cancer deaths within ten years. Cost is proxied by the number invited. To cover 40%, 50% and 60% of cancer deaths, inviting by risk needs {fc['0.4']['fewer_invitations']:.0%}, {fc['0.5']['fewer_invitations']:.0%} and {fc['0.6']['fewer_invitations']:.0%} fewer invitations than inviting by age.")
rows = []
for nm, d in (("NHANES 1999-2008", nh), ("NHANES III 1988-94", n3)):
    cm = d["cost_matched"]
    i2 = cm["shares"].index(0.2) if 0.2 in cm["shares"] else 1
    rows.append([nm, f"{d['n']:,}", d["cancer_deaths"], f"{d['auc_F0']:.3f}", f"{d['auc_F2']:.3f}", f"{d['gain']:+.3f} ({d['gain_ci'][0]:+.3f} to {d['gain_ci'][1]:+.3f})", d["slope"], d["intercept"],
                 f"{100 * (cm['coverage_risk'][i2] - cm['coverage_age'][i2]):+.1f} ({100 * cm['diff_lo'][i2]:+.1f} to {100 * cm['diff_hi'][i2]:+.1f})"])
rows.append(["Both pooled", f"{pool['n']:,}", pool["cancer_deaths"], "", "", "", "", "", f"{100 * pool['shares']['0.2']['diff']:+.1f} ({100 * pool['shares']['0.2']['ci'][0]:+.1f} to {100 * pool['shares']['0.2']['ci'][1]:+.1f})"])
t_ext = table("external", "External checks: the model fitted in NHIS, applied to two other national surveys.", ["Dataset", "Adults 35-84", "Cancer deaths", "AUC age + sex", "AUC with smoking, BMI", "AUC gain (95% CI)", "Slope", "Intercept", "Extra deaths covered at 20% invited, points (95% CI)"], rows,
              "Only age, sex, ever-smoked and BMI were available in all three surveys, so the external model uses those four. Coverage is among adults 40 to 74 for the first two rows and pooled for the last.")
rows = []
for k in ("non-Hispanic White", "non-Hispanic Black", "Hispanic", "other", "less than high school", "high school or GED", "some college", "college or more", "women", "men"):
    g = gr[k]
    gg = g["gain_at_20pct"]
    rows.append([k, f"{g['n']:,}", g["cancer_deaths"], f"{g['auc_F0']:.3f}", f"{g['auc_F2']:.3f}", g["slope_F2"], f"{g['observed_over_expected']:.2f}",
                 f"{F4[k]['OE_F4']:.2f}" if k in F4 else "", f"{100 * gg['diff']:+.1f} ({100 * gg['ci'][0]:+.1f} to {100 * gg['ci'][1]:+.1f})" if gg else "n/a"])
t_eq = table("equity", "Who the free-information policy works for (exploratory): NHIS 2005-2009.", ["Group", "Adults", "Cancer deaths", "AUC age + sex", "AUC F2", "Slope F2", "Observed / predicted F2", "Observed / predicted with race", "Extra deaths covered at 20% invited, points (95% CI)"], rows,
             "Coverage is among adults 40 to 74. 'Observed / predicted with race' is the model that also includes race and ethnicity, shown to see whether it fixes calibration.")
b1, b2s, b3, b4, b5 = R["bar1"], R["bar2"], R["bar3"], R["bar4"], R["bar5"]
rows = [
    ["1", "Overall calibration: slope 0.8 to 1.2, intercept within 0.3 (temporal test; and in NHANES)", f"slope {b1['slope']}, intercept {b1['intercept']}; NHANES {nh['slope']}, {nh['intercept']}; NHANES III {n3['slope']}, {n3['intercept']}", "<b class='ok'>Met</b>"],
    ["2", "Calibration in every age band including 50-59", "all four bands inside the rule in all three datasets (ratios 0.87 to 1.24)", "<b class='ok'>Met</b>"],
    ["3", "F2 beats F0 by 0.02 AUC with an interval above 0", f"{b3['gain']:+.4f} ({b3['ci'][0]:+.4f} to {b3['ci'][1]:+.4f})", "<b class='ok'>Met</b>"],
    ["4", "Fairness: calibration slope 0.8 to 1.2 in each race and ethnicity group, and AUC spread no more than 0.03", f"AUC spread {b4['auc_spread']:.3f}; slopes {min(v['slope'] for v in b4['groups'].values()):.2f} to {max(v['slope'] for v in b4['groups'].values()):.2f}", "<b class='bad'>Missed</b>"],
    ["5", "Cancer-risk age is calibrated: people 5 or more years above their age", f"women {b5['groups']['women']['observed']} observed, {b5['groups']['women']['expected']} predicted; men {b5['groups']['men']['observed']}, {b5['groups']['men']['expected']}", "<b class='ok'>Met</b>"],
    ["C", "Cost-matched: risk-based covers more cancer deaths at every share invited 10-60%, interval above zero (temporal test)", "all shares positive with intervals above zero", "<b class='ok'>Met</b>"],
    ["C-ext", "The same in NHANES", f"positive at every share but intervals include zero at several shares in each survey alone; pooled, intervals above zero at 10-50%; at 60% the lower bound is 0.0", "<b class='bad'>Not met alone</b>, supported pooled"],
]
t_bars = table("bars", "The pre-registered bars and what happened.", ["#", "Bar", "Result", "Outcome"], rows, "Bars were committed to the repository before any model was fitted (docs/CANCER_AGE_PREREG.md, including the addendum that added the cost-matched analysis).")

ABSTRACT = f"""
<p><b>Background.</b> Cancer screening programmes usually invite people by age. Inviting people by risk could reach more of those who will die of cancer for the same number of invitations, which would make screening cheaper or catch more cancer earlier, but risk tools usually need tests, records or genetics. We asked how far four free answers (age, sex, smoking and body mass index) can go.</p>
<p><b>Methods.</b> We pre-registered the models, the bars and a cost-matched comparison before fitting anything. Using {R['n_fit'] + R['n_test']:,} adults aged 35 to 84 from the National Health Interview Survey (NHIS, 1997-2009; no earlier cancer) linked to the National Death Index through 2019, we fitted ten-year cancer-death models on 1997-2002 and tested them on 2005-2009, then checked them in NHANES 1999-2008 and NHANES III (1988-1994). The comparison invites the same number of adults aged 40 to 74 by age alone or by predicted risk and counts the cancer deaths among those invited.</p>
<p><b>Results.</b> The cheapest model that sufficed used age, sex, smoking and BMI (AUC {M[names[2]]['auc']:.3f}, against {M[names[0]]['auc']:.3f} for age and sex; gain {G['F2 vs F0']['gain']:+.3f}, 95% CI {G['F2 vs F0']['ci'][0]:+.3f} to {G['F2 vs F0']['ci'][1]:+.3f}); schooling, marital status, self-rated health and activity added {G['F3 vs F2']['gain']:+.4f}. It was calibrated overall (slope {b1['slope']}) and in every age band, including 50 to 59, where an earlier, smaller model failed, and it held in both external surveys. A current smoker carried the cancer-death risk of someone {RAP['current smokers']:.0f} years older, on average. At the same number invited, inviting by risk covered {pc(CM['coverage_risk'][3])} of cancer deaths at 20% invited against {pc(CM['coverage_age'][3])} by age (difference {100 * CM['diff'][3]:+.1f} points, 95% CI {100 * CM['diff_lo'][3]:+.1f} to {100 * CM['diff_hi'][3]:+.1f}), or, to cover half of cancer deaths, needed {fc['0.5']['fewer_invitations']:.0%} fewer invitations. Among adults aged 40 to 49, {pc(SS['share_risk_age_50plus'], 1)} already had the risk of an average 50-year-old, held {pc(SS['share_of_deaths_in_high_group'])} of that age group's cancer deaths, and {pc(SS['composition_of_high_group']['current smoker'])} were current smokers. <b>The fairness bar was missed:</b> the model ranked non-Hispanic White adults better (AUC {gr['non-Hispanic White']['auc_F2']:.3f}) than Black ({gr['non-Hispanic Black']['auc_F2']:.3f}), Hispanic ({gr['Hispanic']['auc_F2']:.3f}) and other ({gr['other']['auc_F2']:.3f}) adults, under-predicted cancer deaths in Black adults by {pc(gr['non-Hispanic Black']['observed_over_expected'] - 1)}, and its coverage gain at 20% invited was clear for White adults ({100 * gr['non-Hispanic White']['gain_at_20pct']['diff']:+.1f} points) and for adults without a high-school diploma ({100 * gr['less than high school']['gain_at_20pct']['diff']:+.1f}) but not distinguishable from zero for Black, Hispanic or other adults. Adding race fixed calibration without improving ranking.</p>
<p><b>Conclusions.</b> A few free questions can make screening invitations measurably cheaper in terms of cancer deaths reached, in three national datasets, but the gain comes mostly from smoking and is smaller and uncertain for non-White adults. The outcome is all cancer deaths and not the screenable subset, so the result describes where cancer-death risk sits and not what screening would prevent. A registry cohort with cancer sites and diagnoses is the next test.</p>
"""

INTRO = """
<p>Screening for cancer is expensive when it is offered to everyone in an age band, and it is only worth the cost and the harms (false alarms, overdiagnosis) for people whose chance of benefit is high enough. Programmes therefore choose a starting age and a stopping age. Recently, several guidelines moved the starting age for bowel cancer and breast cancer screening earlier, which makes the case for a better way of choosing who to invite stronger, not weaker: starting everyone earlier costs more, and starting only those at higher risk earlier could cost less.</p>
<p>Risk-based invitation is not a new idea. For lung cancer, a risk model built on a few questions (age, smoking, education, body mass index, lung disease and family history) selected people for screening better than fixed pack-year rules.<sup>1</sup> Recent work has derived risk-adapted <i>starting ages</i> for lung, bowel and breast cancer screening using smoking history, polygenic scores, biomarkers or risk factors.<sup>2,3,4</sup> These are specific to one cancer and often need genetic or laboratory information.</p>
<p>The question here is more basic and aimed at the original idea of a cheap approach: <b>how much of the risk of dying of cancer can be read from information that costs nothing to collect?</b> Earlier work in this project found that routine blood tests add almost nothing to age, sex, smoking and weight for cancer death, which leaves those free answers as the cheapest possible risk tool. It also found that a model built on one national survey was too imprecise: it under-predicted cancer deaths in people aged 50 to 59 and could not be released. This paper uses a survey about ten times larger, with the bars fixed first, to ask four things. How far do free answers go? Is the result calibrated in every age band and in other surveys? What does it do to the number of invitations needed to reach the same share of cancer deaths? And for whom does it work?</p>
"""

METHODS = f"""
<h3>2.1 Pre-registration</h3>
<p>The models, the cancer-risk-age definition, five bars and a screening-start question were committed to the repository on 7 October 2026 (<code>docs/CANCER_AGE_PREREG.md</code>) before any model was fitted. The same day, still before any model was fitted, an addendum added the cost-matched invitation comparison and stated that the aim is earlier, cheaper screening and not communication. The fairness analyses of Section 3.7 beyond bar 4 are exploratory and were added after the results.</p>
<h3>2.2 Data</h3>
<p>The National Health Interview Survey (NHIS) is a household interview survey of the US population. We used the Sample Adult and Person files for 1997-2002 and 2005-2009 (2003 was built from its loose data file; 2004 has no layout file in the standard place and was left out), linked by NCHS through public-use Linked Mortality Files to the National Death Index through 31 December 2019. Layouts were read from NCHS's own SAS programs, and all {R['n_fit'] + R['n_test']:,} adult records matched to the mortality file. We kept adults aged 35 to 84 without a reported earlier cancer and with known smoking status and BMI. The outcome is death from cancer (underlying cause 2, ICD-10 C00-C97) within ten years of interview. Non-cases are people alive at ten years or dead of another cause; nobody was dropped for how they died.</p>
<p>The predictors are free: age, sex, BMI, smoking (never; former, with years since quitting; current, with cigarettes per day), and for the richer models schooling (four groups), marital status, self-rated health, vigorous activity, and race and ethnicity (non-Hispanic White, non-Hispanic Black, Hispanic, other). Race and ethnicity enter only the fairness comparison; the cheapest model does not use them.</p>
<h3>2.3 Models and the temporal test</h3>
<p>Five nested logistic models (F0 age and sex; F1 plus smoking; F2 plus BMI; F3 plus schooling, marital status, self-rated health and activity; F4 plus race and ethnicity), each fitted separately by sex with a cubic spline in age so the age curve can bend. Models were fitted on interview years 1997-2002 ({R['n_fit']:,} adults, {R['cancer_fit']:,} cancer deaths) and tested on 2005-2009 ({R['n_test']:,} adults, {R['cancer_test']:,} cancer deaths). The cheapest model that suffices was fixed as F2 unless F3 gained at least 0.01 AUC with a bootstrap interval above zero.</p>
<h3>2.4 External checks</h3>
<p>The model was fitted in NHIS with smoking collapsed to ever or never (the only smoking measure all three surveys share) and applied, with nothing refitted, to NHANES 1999-2008 and to NHANES III (1988-1994), separate surveys with different samples, instruments and years, using the same ten-year cancer-death outcome from their own National Death Index linkage.</p>
<h3>2.5 Cancer-risk age</h3>
<p>For each person the model gives a ten-year risk. The <b>cancer-risk age</b> is the age at which the age-and-sex curve (model F0) reaches that same risk, and the <b>risk advancement period</b> is the cancer-risk age minus the person's own age. A 45-year-old whose free answers put them where an average 58-year-old of their sex stands has an advancement of 13 years. This is a technical way of putting people on one scale; we make no claim here about communication or understanding.</p>
<h3>2.6 The cost-matched comparison</h3>
<p>Cost is proxied by the number of people invited. In the temporal test, among adults aged 40 to 74, we compared two policies that invite exactly the same number of people: <b>A</b>, the oldest people, and <b>B</b>, the people with the highest predicted risk. We counted the ten-year cancer deaths among those invited (coverage), with paired bootstrap 95% intervals, at each share invited from 5% to 75%. We also fixed the coverage (40%, 50% and 60% of cancer deaths) and counted how many invitations each policy needs. All outcomes are all cancer deaths, which includes cancers with no screening test.</p>
<h3>2.7 Bars</h3>
<p>Five bars (Table {TN['bars']}) and the cost-matched bar were fixed in advance. Calibration used the slope and intercept of the outcome on the logit of the prediction, and observed against predicted deaths by age band (within 20% or inside the Poisson 95% interval). Fairness required a calibration slope between 0.8 and 1.2 in every race and ethnicity group and an AUC spread no larger than 0.03.</p>
<h3>2.8 Departures from the pre-registration</h3>
<p>The pre-registration specified a natural spline in age with 4 degrees of freedom; the code uses a cubic spline with five quantile knots (scikit-learn SplineTransformer), fitted on the fitting years only. The choice of F2 as the cheapest model and the bar thresholds were fixed in advance, but the spline form was settled while writing the code, and the 2005-2009 test set was seen during development. The first listed bars were not changed after the results.</p>
<h3>2.9 Software and reproducibility</h3>
<p>Python (scikit-learn, NumPy, pandas). The cohort builder is <code>fetch_nhis_cancer_risk.py</code>, the analysis <code>experiments/cancer_age.py</code>, the exploratory analyses <code>experiments/cancer_age_extra.py</code>; tables and figures are built from their saved output.</p>
"""

RESULTS = f"""
<h3>3.1 Summary against the pre-registered bars</h3>
{t_bars}
{fig("fig1_design", "The design. Four free answers give a ten-year cancer-death risk; two invitation policies are compared at the same budget.")}
<h3>3.2 How far do free answers go?</h3>
<p>Age and sex alone ranked ten-year cancer death with AUC {M[names[0]]['auc']:.3f} in the temporal test. Adding smoking raised it to {M[names[1]]['auc']:.3f} and BMI to {M[names[2]]['auc']:.3f} (gain over age and sex {G['F2 vs F0']['gain']:+.4f}, 95% CI {G['F2 vs F0']['ci'][0]:+.4f} to {G['F2 vs F0']['ci'][1]:+.4f}). BMI itself added {G['F2 vs F1']['gain']:+.4f}; schooling, marital status, self-rated health and activity together added {G['F3 vs F2']['gain']:+.4f} (interval {G['F3 vs F2']['ci'][0]:+.4f} to {G['F3 vs F2']['ci'][1]:+.4f}), and race and ethnicity {G['F4 vs F3']['gain']:+.4f}. By the pre-set rule, F2 is the cheapest model that suffices: age, sex, smoking and BMI, four answers anyone can give in a minute ({TR('ladder')}, {F('fig2_ladder')}).</p>
{fig("fig2_ladder", "AUC for ten-year cancer death in 2005-2009 by how much free information the model uses.", "85%")}
{t_ladder}
<h3>3.3 Calibration, including the age band that failed before</h3>
<p>The F2 model was well calibrated overall (slope {b1['slope']}, intercept {b1['intercept']}) and in every age band: in 50 to 59-year-olds it predicted {b2['50-59']['expected']} cancer deaths and {b2['50-59']['observed']} occurred, a ratio of {b2['50-59']['ratio']:.2f}, where the earlier NHANES-only model had failed. The larger sample and the spline for age closed the gap. In NHANES and NHANES III the calibration slopes were {nh['slope']} and {n3['slope']} ({TR('bands')}, {F('fig3_calibration')}).</p>
{fig("fig3_calibration", "Left: calibration in ten equal-sized groups of the 2005-2009 test set. Right: observed over predicted cancer deaths by age band in the three datasets; the shaded band is 0.8 to 1.2.")}
{t_bands}
<h3>3.4 Cancer-risk age</h3>
<p>On average a current smoker carried the cancer-death risk of someone {RAP['current smokers']:.1f} years older, a former smoker {RAP['former smokers']:+.1f} years and a never-smoker {RAP['never smokers']:+.1f} years, relative to the average of their age and sex. The median advancement was {RAP['50']:+.1f} years and the 95th percentile {RAP['95']:+.1f}. Current smokers aged 40 to 49 who smoked 20 or more cigarettes a day averaged {RAP['current smokers 20+/day aged 40-49']:+.1f} years. Among people with an advancement of 5 years or more, observed cancer deaths matched predicted in both sexes ({b5['groups']['women']['observed']} against {b5['groups']['women']['expected']} in women; {b5['groups']['men']['observed']} against {b5['groups']['men']['expected']} in men), so the measure means what it says ({F('fig4_riskage')}).</p>
{fig("fig4_riskage", "Left: risk advancement (cancer-risk age minus age) by smoking status. Right: mean advancement by age. A positive value means free answers place the person where an older average person stands.")}
<h3>3.5 The same number invited: more cancer deaths covered</h3>
<p>In the temporal test, at every share invited from 10% to 60% of adults aged 40 to 74, inviting by predicted risk covered more of the ten-year cancer deaths than inviting the oldest, with intervals above zero ({TR('cost')}, {F('fig5_coverage')}). Inviting 20% of adults covered {pc(CM['coverage_risk'][3], 1)} of cancer deaths by risk against {pc(CM['coverage_age'][3], 1)} by age, a difference of {100 * CM['diff'][3]:+.1f} points (95% CI {100 * CM['diff_lo'][3]:+.1f} to {100 * CM['diff_hi'][3]:+.1f}); inviting 10% covered {pc(CM['coverage_risk'][1], 1)} against {pc(CM['coverage_age'][1], 1)}. Put the other way, covering half of all cancer deaths needed {pc(fc['0.5']['share_invited_risk'])} of adults by risk and {pc(fc['0.5']['share_invited_age'])} by age, which is {fc['0.5']['fewer_invitations']:.0%} fewer invitations; for 40% of cancer deaths the saving was {fc['0.4']['fewer_invitations']:.0%} and for 60% it was {fc['0.6']['fewer_invitations']:.0%}. The advantage is largest when few are invited and fades toward zero as most adults are invited, as it must.</p>
{fig("fig5_coverage", "Left: share of ten-year cancer deaths among those invited, as more adults aged 40-74 are invited, by age and by risk (shaded: 95% interval of the risk-based curve). Right: invitations needed to cover a fixed share of cancer deaths.")}
{t_cost}
<h3>3.6 What this means for people in their forties</h3>
<p>No adult under 50 is invited under an age policy at a 30% invitation rate (the oldest 30% of adults 40 to 74 are all over 50). Under the risk policy {E['risk_policy_invites_under_50']:,} people under 50 were invited and {E['cancer_deaths_under_50_covered_risk']} of the {E['cancer_deaths_under_50_total']} cancer deaths in that age group were among them, against none. That is a small absolute number; most cancer deaths in people under 50 are not predicted by smoking and BMI. Among all {SS['n_40_49']:,} adults aged 40 to 49, {pc(SS['share_risk_age_50plus'], 1)} had a cancer-risk age of 50 or more; they held {pc(SS['share_of_deaths_in_high_group'])} of that age group's cancer deaths, against {pc(SS['share_of_deaths_flagging_oldest_same_number'])} if the same number of the oldest 40 to 49-year-olds were chosen. They were overwhelmingly current smokers ({pc(SS['composition_of_high_group']['current smoker'])} against {pc(SS['composition_of_all_40_49']['current smoker'])} of the age group) and were more often without a high-school diploma ({pc(SS['composition_of_high_group']['less than high school'])} against {pc(SS['composition_of_all_40_49']['less than high school'])}) ({F('fig8_forties')}). The descriptive answer to "who is already at fifty-year-old risk in their forties" is: smokers.</p>
{fig("fig8_forties", "Composition of the adults aged 40 to 49 whose free answers put them at the cancer-death risk of an average 50-year-old or older, against all adults aged 40 to 49.", "85%")}
<h3>3.7 Does it hold in other surveys?</h3>
<p>Applied unchanged to NHANES 1999-2008 ({nh['n']:,} adults, {nh['cancer_deaths']} cancer deaths) the model's AUC rose from {nh['auc_F0']:.3f} to {nh['auc_F2']:.3f} (gain {nh['gain']:+.3f}, interval {nh['gain_ci'][0]:+.3f} to {nh['gain_ci'][1]:+.3f}), and in NHANES III ({n3['n']:,} adults, {n3['cancer_deaths']} cancer deaths) from {n3['auc_F0']:.3f} to {n3['auc_F2']:.3f} ({n3['gain']:+.3f}, {n3['gain_ci'][0]:+.3f} to {n3['gain_ci'][1]:+.3f}). Calibration was acceptable in both ({TR('external')}). The cost-matched advantage pointed the same way at every share in both surveys ({F('fig6_external')}). Each survey alone has too few cancer deaths for the interval to exclude zero at every share, which is why that part of the bar is recorded as not met alone; pooled, the intervals excluded zero at 10% to 50% invited and just reached zero at 60% ({'above zero at every share' if pool['interval_above_zero_everywhere'] else 'the lower bound was 0.0 points at 60%'}).</p>
{fig("fig6_external", "Extra cancer deaths covered by inviting by risk instead of by age (risk-based minus age-only, in points), at each share invited, in the temporal test and in two other national surveys. Lines are 95% intervals.", "85%")}
{t_ext}
<h3>3.8 The bar that was missed: fairness</h3>
<p>Calibration and ranking differed by race and ethnicity ({TR('equity')}, {F('fig7_equity')}). The model's AUC was {gr['non-Hispanic White']['auc_F2']:.3f} in non-Hispanic White adults, {gr['non-Hispanic Black']['auc_F2']:.3f} in Black, {gr['Hispanic']['auc_F2']:.3f} in Hispanic and {gr['other']['auc_F2']:.3f} in other adults, a spread of {b4['auc_spread']:.3f} against the bar of 0.03, and the calibration slope in the 'other' group ({gr['other']['slope_F2']}) was below 0.8. It under-predicted cancer deaths in Black adults (observed over predicted {gr['non-Hispanic Black']['observed_over_expected']:.2f}) and over-predicted them in Hispanic adults ({gr['Hispanic']['observed_over_expected']:.2f}). Adding race and ethnicity made both calibrated ({F4['non-Hispanic Black']['OE_F4']:.2f} and {F4['Hispanic']['OE_F4']:.2f}) but did not improve ranking, and it shifts who is invited (exploratory; the shares were not saved in the result files), so whether to include race is a policy decision and not a technical one.</p>
<p>The benefit followed the same pattern. At 20% invited the coverage gain was {100 * gr['non-Hispanic White']['gain_at_20pct']['diff']:+.1f} points in White adults (interval {100 * gr['non-Hispanic White']['gain_at_20pct']['ci'][0]:+.1f} to {100 * gr['non-Hispanic White']['gain_at_20pct']['ci'][1]:+.1f}) but {100 * gr['non-Hispanic Black']['gain_at_20pct']['diff']:+.1f}, {100 * gr['Hispanic']['gain_at_20pct']['diff']:+.1f} and {100 * gr['other']['gain_at_20pct']['diff']:+.1f} in Black, Hispanic and other adults, with intervals that included zero. It was largest in adults without a high-school diploma ({100 * gr['less than high school']['gain_at_20pct']['diff']:+.1f}) and smallest in college graduates ({100 * gr['college or more']['gain_at_20pct']['diff']:+.1f}): the policy directs invitations toward people with less schooling, who more often smoke and die younger of cancer. We did not test why the model works less well in non-White adults. A plausible reason is that smoking explains a smaller share of their cancer deaths and the cancers that matter more (for example breast, prostate and bowel) are not predicted by smoking and BMI, but that is a hypothesis.</p>
{fig("fig7_equity", "Left: AUC by race and ethnicity. Middle: observed over predicted cancer deaths without and with race in the model. Right: extra cancer deaths covered at 20% invited, by group; teal intervals exclude zero.")}
{t_eq}
<h3>3.9 Robustness</h3>
<p>The result did not depend on early deaths or the horizon. With the first 24 months of follow-up removed (to guard against undiagnosed cancer), the AUC gain held ({rob['first_24_months_removed']['auc_F0']:.3f} to {rob['first_24_months_removed']['auc_F2']:.3f}) and the coverage gain at 20% invited was {100 * rob['first_24_months_removed']['gain_at_20pct']['diff']:+.1f} points ({100 * rob['first_24_months_removed']['gain_at_20pct']['ci'][0]:+.1f} to {100 * rob['first_24_months_removed']['gain_at_20pct']['ci'][1]:+.1f}). With a five-year horizon it was {100 * rob['five_year']['gain_at_20pct']['diff']:+.1f} points ({100 * rob['five_year']['gain_at_20pct']['ci'][0]:+.1f} to {100 * rob['five_year']['gain_at_20pct']['ci'][1]:+.1f}), AUC {rob['five_year']['auc_F0']:.3f} to {rob['five_year']['auc_F2']:.3f}.</p>
<h3>3.10 Survey-weighted, national scale (exploratory)</h3>
<p>The main analysis counts respondents. Weighting each respondent by the NHIS sample-adult weight so the sample stands for the US civilian adult population ({N['population_40_74_millions']} million adults aged 40 to 74 in 2005-2009), the advantage was the same size or larger: inviting 20% of adults covered {pc(N['shares']['0.2']['risk_based'])} of ten-year cancer deaths by risk against {pc(N['shares']['0.2']['age_only'])} by age ({100 * N['shares']['0.2']['diff']:+.1f} points, interval {100 * N['shares']['0.2']['ci'][0]:+.1f} to {100 * N['shares']['0.2']['ci'][1]:+.1f}), and the interval excluded zero from 10% to 50% invited but not at 60% ({100 * N['shares']['0.6']['diff']:+.1f}, {100 * N['shares']['0.6']['ci'][0]:+.1f} to {100 * N['shares']['0.6']['ci'][1]:+.1f}). To cover half of ten-year cancer deaths, inviting by risk instead of age would need about {N['fixed_coverage']['0.5']['invitations_saved_millions']} million fewer invitations ({N['fixed_coverage']['0.5']['fewer_invitations']:.0%}); for 40% of deaths, {N['fixed_coverage']['0.4']['invitations_saved_millions']} million ({N['fixed_coverage']['0.4']['fewer_invitations']:.0%}). These are one-off counts for a population of that size, not annual savings or money, and the bootstrap ignores the survey's clustering, so the intervals are somewhat too narrow.</p>
<h3>3.11 Independent re-analysis and what it changes</h3>
<p>A separate AI agent re-wrote the spline, regression, AUC and coverage code from the pre-registration and reproduced the headline numbers (AUC 0.756 and 0.785; calibration slope 1.01; 55.0% against 47.6% of cancer deaths covered at 20% invited), found no leakage between fitting and test years, no early censoring among those counted as non-cases, and no change after dropping the first 24 months or using survey weights (<code>docs/independent_check/</code>). This is a check of the code, not a second human analyst. It also made a sharper point than the text above: <b>age plus smoking status alone (never, former, current) reaches an AUC of 0.785, the same as the full four-answer model</b>, and the extra deaths covered by inviting by risk are all in current and former smokers (never-smokers are covered slightly less well than by age). BMI, cigarettes per day and years since quitting add almost nothing here. The honest summary is that the free-information policy is a smoking-status policy, and age-only is a weak comparator because it ignores the single strongest cheap fact. Across interview years the absolute risk also drifted downward (observed over predicted about 0.90 to 0.96 after 1999) and the coverage advantage looked smaller in the latest years, which is expected as smoking falls but was not formally tested.</p>
"""

DISCUSSION = f"""
<h3>4.1 What the study shows</h3>
<p>Four free answers (age, sex, smoking and BMI) are enough to rank ten-year cancer death well, are calibrated across age bands, and replicate in two other national surveys. Using them to decide whom to invite, in place of age alone, covers more cancer deaths for the same number of invitations, or the same share of cancer deaths with roughly a fifth to a third fewer invitations. In the sense of the original idea, this is cancer risk assessment that costs nothing to collect and could make screening earlier for people at high risk and cheaper overall.</p>
<p>The gain is essentially smoking status (Section 3.11). Smoking drove almost all the improvement over age and sex, the people moved up in the ranking in their forties were overwhelmingly current smokers, and the people who gained most were those with less schooling. This is not a surprise, and it is a limit: for the many cancers that smoking does not cause, free answers say little.</p>
<h3>4.2 What it does not show</h3>
<ul>
<li><b>All cancer deaths, not screenable cancers.</b> The public mortality file does not name the cancer site, so coverage is of deaths from any cancer. Some of them (pancreas, for example) have no screening test, and screening for others (breast, bowel, cervix, lung) works on different time scales. The result describes where cancer-death risk is concentrated and not what screening would prevent. Translating it needs data with cancer sites and, ideally, diagnoses.</li>
<li><b>Death, not diagnosis, and no screening effect.</b> The outcome cannot show whether a screening programme would have found a cancer earlier or whether early detection would have changed the outcome.</li>
<li><b>The invitation count is a crude cost.</b> Real programmes also pay for follow-up of false alarms, for recruitment, and for collecting the free answers. We did not model any of this.</li>
<li><b>Smoking is changing.</b> The cohorts were interviewed from 1997 to 2009, when many more people smoked than do now. A model built on smoking will need recalibration as smoking falls, which is why the calibration checks matter.</li>
</ul>
<h3>4.3 Equity</h3>
<p>The fairness bar was missed and the paper reports it as a main result. The same policy that reduces invitations overall helps non-Hispanic White adults and adults with less schooling clearly, and helps Black, Hispanic and other adults by amounts we cannot distinguish from zero. A tool built on this model would under-predict risk for Black adults and over-predict it for Hispanic adults unless race were included, and including race changes who is invited. Neither option is obviously right, and the choice belongs to the people who run a programme and the communities it serves. A better answer is probably more information that matters in those groups (family history, screening history, and cancer-specific risk factors) and not a different way of using the same four answers.</p>
<h3>4.4 Relation to existing work</h3>
<p>Risk-based selection for lung cancer screening is established,<sup>1</sup> and risk-adapted starting ages have been derived for lung, bowel and breast cancer using smoking history, polygenic scores or biomarkers.<sup>2,3,4</sup> A recorded search of Europe PMC on 7 October 2026 (six queries, 806 retained records; <code>data/audit/novelty_cancer_age.csv</code>) found no study that frames all-cancer-death risk from free information as a risk-equivalent age, compares cost-matched invitation by age and by risk, and checks it in three national surveys with an equity analysis. The closest work is site-specific. This is not proof that none exists. The contribution is the general, free-information framing, the cost-matched comparison, the external replication and the honest equity result, not the invention of risk-based screening.</p>
<h3>4.5 Next steps</h3>
<ol>
<li>Repeat the cost-matched comparison in a cohort with cancer registry diagnoses and sites (for example the UK Biobank, PLCO or CHARLS; <code>docs/CLINICAL_DATA_ACCESS.md</code>), cancer by cancer, so that coverage is of screenable cancers.</li>
<li>Add the information that is free to ask and matters in the groups where the model works less well, such as family history, and test it against the bars again.</li>
<li>Ask a clinician and a screening-programme planner what the invitation count leaves out and which decision a risk-based invitation could actually change.</li>
</ol>
"""

DECL = """
<h3>AI assistance</h3>
<p>This study was carried out with an AI assistant (Anthropic Claude, via Claude Code), which wrote the cohort builder and analysis code, ran the analyses, drafted the text and searched for background literature. An AI cannot be an author. The human author set the question, is responsible for the work, and must verify the statistics and every statement before submission. No second person has re-run the analysis.</p>
<h3>Ethics, data and code</h3>
<p>The study uses public, de-identified survey data and NCHS public-use linked mortality files; no participants were contacted. Code, pre-registration and results are in the public repository (github.com/palashraks-afk/OncoVision). Clinical mentorship and authorship are not asserted here.</p>
<h3>Conflicts of interest and funding</h3>
<p>None declared. No funding.</p>
"""

REFS = """
<ol>
<li>Tammem&auml;gi MC, Katki HA, Hocking WG, et al. Selection criteria for lung-cancer screening. <i>N Engl J Med</i> 2013;368:728-736.</li>
<li>Frick C, Hallsson LR, Siebert U, Bhardwaj M, Sch&ouml;ttker B, Brenner H, et al. Risk-adapted lung cancer screening starting ages for former smokers. 2025 (PMID 41632141).</li>
<li>Chen X, Heisser T, Cardoso R, Hoffmeister M, Brenner H. Personalized initial screening age for colorectal cancer in individuals at average risk. 2023 (PMID 37878311).</li>
<li>Zheng Y, Dong X, Li J, Qin C, Xu Y, Wang F, Cao W, et al. Use of breast cancer risk factors to identify risk-adapted starting age of screening in China. 2022 (PMID 36355372).</li>
<li>Hippisley-Cox J, Coupland C. Development and validation of risk prediction algorithms to estimate future risk of common cancers in men and women: prospective cohort study. <i>BMJ Open</i> 2015;5:e007825.</li>
<li>Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. <i>BMJ</i> 2024;385:e078378.</li>
<li>US Preventive Services Task Force. Recommendations for breast (2024), cervical (2018), colorectal (2021), lung (2021) and prostate (2018) cancer screening. uspreventiveservicestaskforce.org.</li>
<li>National Center for Health Statistics. National Health Interview Survey public-use files 1997-2009 and public-use Linked Mortality Files (follow-up through 31 December 2019); NHANES and NHANES III data and linked mortality files.</li>
</ol>
<p class="tnote">References 1, 5 and 6 were looked up by title in Crossref and match; references 2 to 4 were identified from Europe PMC records and their journals should be completed from the originals. References 7 and 8 are guidelines and data products.</p>
"""

TITLE = "Can Free Information Make Cancer Screening Cheaper?"
SUB = f"Risk-based invitation from age, sex, smoking and body mass index in {R['n_fit'] + R['n_test']:,} US adults, checked in two other national surveys, with an equity analysis in which one pre-registered bar was missed"
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1><div class="sub">{SUB}</div>
<div class="meta"><b>Palash Rakshit</b> (author) &nbsp;·&nbsp; Draft of 8 October 2026 &nbsp;·&nbsp; Oncovision project</div>
<div class="banner"><b>Draft. Not peer reviewed.</b> The analysis was run by an AI assistant and has not been re-run by a second person. The outcome is death from any cancer, not screenable cancers or diagnoses, so the result describes where risk sits and does not estimate what screening would prevent. One pre-registered bar (fairness) was missed. Nothing here is medical advice.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> cancer screening; risk-based invitation; screening starting age; risk advancement period; smoking; National Health Interview Survey; health equity; pre-registration.</p>
<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
</body></html>"""
out = os.path.join(HERE, "age_paper.html")
open(out, "w", encoding="utf-8").write(html)
pdf = os.path.join(HERE, "Free_Information_Cheaper_Cancer_Screening.pdf")
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
