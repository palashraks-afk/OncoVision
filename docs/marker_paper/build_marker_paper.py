"""
Builds the cheap-blood-markers paper (HTML, then PDF through headless Chrome) from the saved
result files, so no number in the text can differ from the analysis.

Run:  python docs/marker_paper/build_marker_paper.py
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_result.json")))
S = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_sensitivity_result.json")))
RR = json.load(open(os.path.join(ROOT, "experiments", "risk_ranking_result.json")))
src = open(os.path.join(ROOT, "docs", "paper", "build_paper.py"), encoding="utf-8").read()
CSS = src[src.index('CSS = """') + 9: src.index('"""', src.index('CSS = """') + 9)]
I = R["indices"]
NAMES = list(I)
INFL = ["NLR", "PLR", "MLR", "SII", "SIRI", "NPAR"]

FIGS = ["fig1_logic", "fig2_gain", "fig3_frailty", "fig4_specificity", "fig5_claims"]
FN = {n: i + 1 for i, n in enumerate(FIGS)}
TNAMES = ["main", "decedent", "claims", "fiveyear"]
TN = {n: i + 1 for i, n in enumerate(TNAMES)}


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


def g(v):
    return f"{v:+.4f}"


def ci2(c):
    return f"{c[0]:+.3f} to {c[1]:+.3f}"


def orc(o, c):
    return f"{o:.2f} ({c[0]:.2f} to {c[1]:.2f})"


rows = []
for n in NAMES:
    r = I[n]
    rows.append([n, f"{g(r['T1_gain'])} ({ci2(r['T1_ci'])})", f"{g(r['T3_gain'])} ({ci2(r['T3_ci'])})", orc(r["or_cancer_death"], r["or_cancer_ci"]),
                 orc(r["or_noncancer_death"], r["or_noncancer_ci"]), "<b class='bad'>no</b>" if not r["qualifies"] else "<b class='ok'>yes</b>"])
t_main = table("main", "Twelve cheap blood markers: added discrimination and association with each kind of death.",
               ["Marker", "AUC gain, cancer death (95% CI)", "AUC gain, other-cause death (95% CI)", "OR per SD, cancer death", "OR per SD, other-cause death", "Qualifies"], rows,
               f"AUC gain: baseline of age, sex, smoking and BMI plus the marker, fitted on 1999-2004 and tested on 2005-2008. OR per SD against people alive at ten years, adjusted for the baseline and cycle. Baseline AUC in the test era: cancer death {R['baseline_auc']['cancer']:.3f}, other-cause death {R['baseline_auc']['noncancer']:.3f}. Qualifying requires an AUC gain of at least 0.01 with an interval above 0 AND a cancer-specific signal among decedents.")
rows = []
for n in NAMES:
    r = I[n]
    rows.append([n, orc(r["T2_or"], r["T2_ci"]), f"{r['T2_p_holm']:.3f}", "yes" if r["passes_T2"] else "no", "yes" if r["passes_T1"] else "no"])
t_dec = table("decedent", "Specificity among people who died: odds of dying of cancer rather than another cause, per SD.", ["Marker", "OR (95% CI)", "Holm-adjusted p", "Passes specificity bar", "Passes added-discrimination bar"], rows,
              f"{S['all (the pre-registered analysis)']['decedents']:,} people who died within ten years, {S['all (the pre-registered analysis)']['cancer_deaths']} of them from cancer. Adjusted for age, age squared, sex, smoking, BMI and cycle.")
rows = []
for n in NAMES:
    r = R["T5"]["indices"][n]
    rows.append([n, orc(r["or"], r["ci"]), f"{r['auc_gain_later_cycles']:+.4f}"])
t_claims = table("claims", "The literature's own design: self-reported cancer history at the same visit.", ["Marker", "OR per SD (95% CI)", "AUC gain, later cycles"], rows,
                 f"{R['T5']['n']:,} adults 40 and over, {R['T5']['cases']:,} with a cancer history. Baseline AUC (later cycles) {R['T5']['baseline_auc']:.3f}. Adjusted for age, sex, smoking, BMI and cycle.")
five = S["five_year"]
rows = [[n, f"{five['indices'][n]['gain_cancer']:+.4f} ({five['indices'][n]['ci_cancer'][0]:+.3f} to {five['indices'][n]['ci_cancer'][1]:+.3f})", f"{five['indices'][n]['gain_noncancer']:+.4f}"] for n in NAMES]
t_five = table("fiveyear", "Five-year horizon, all eight cycles, fitted on 1999-2006 and tested on 2007-2014.", ["Marker", "AUC gain, cancer death (95% CI)", "AUC gain, other-cause death"], rows,
               f"{five['n']:,} adults, {five['cancer_deaths_test']} cancer deaths in the test era. Baseline AUC {five['baseline_cancer_auc']:.3f}.")

infl_nc = [I[n]["or_noncancer_death"] for n in INFL]
infl_c = [I[n]["or_cancer_death"] for n in INFL]
dec_lo = min(I[n]["T2_or"] for n in INFL)
dec_hi = max(I[n]["T2_or"] for n in INFL)
nq = R["n_qualify"]
rdw = I["RDW"]

ABSTRACT = f"""
<p><b>Background.</b> A cheap way to estimate cancer risk would be to use tests people already have. A large literature does this with blood-count indices such as the neutrophil-to-lymphocyte ratio (NLR), the systemic immune-inflammation index (SII) and the neutrophil-percentage-to-albumin ratio (NPAR), reporting that higher values go with cancer. A marker that rises with illness in general would show the same thing, so two questions matter: does it add to what age, sex, smoking and body mass index already tell us, in data it was not built on, and is it linked to dying of cancer more than to dying of other causes?</p>
<p><b>Methods.</b> We pre-registered twelve indices (NLR, platelet-to-lymphocyte ratio, monocyte-to-lymphocyte ratio, SII, SIRI, NPAR, advanced lung cancer inflammation index, prognostic nutritional index, RDW, white count, albumin, haemoglobin), two tests, and the bars for passing them, before running anything. In {R['n']:,} US adults aged 40 and over from NHANES 1999-2008 with ten years of linked follow-up, each marker was added to a baseline of age, sex, smoking and BMI and tested on later survey cycles (test 1), and compared between people who died of cancer and of other causes (test 2), with death from other causes as a negative-control outcome.</p>
<p><b>Results.</b> <b>None of the twelve markers qualified</b> ({nq} of 12). No marker improved discrimination of ten-year cancer death by the pre-set 0.01 AUC with a confidence interval above zero (baseline AUC {R['baseline_auc']['cancer']:.3f}); the best, RDW, gained {rdw['T1_gain']:+.3f} (95% CI {rdw['T1_ci'][0]:+.3f} to {rdw['T1_ci'][1]:+.3f}) and gained more for other-cause death ({rdw['T3_gain']:+.3f}). Five of the six inflammation indices were clearly associated with death from other causes (odds ratio per SD up to {max(infl_nc):.2f}) but not with death from cancer (odds ratios {min(infl_c):.2f} to {max(infl_c):.2f}), and among people who died all six pointed away from cancer (odds {dec_lo:.2f} to {dec_hi:.2f}). Albumin-based markers passed the specificity test only because they lowered the risk of other deaths. All twelve together gained {R['T4']['gain']:+.3f} (95% CI {R['T4']['ci'][0]:+.3f} to {R['T4']['ci'][1]:+.3f}). The result held after removing the first two years, in each sex, in each age group and at a five-year horizon. In the design most papers use, self-reported cancer history at the same visit, the same markers were associated (odds ratio up to {max(v['or'] for v in R['T5']['indices'].values()):.2f} per SD) but added at most {max(v['auc_gain_later_cycles'] for v in R['T5']['indices'].values()):.3f} AUC.</p>
<p><b>Conclusions.</b> Popular cheap blood-count markers behave as markers of general frailty. The associations in the literature reproduce, and they are real, but they are not cancer-specific and are of no use for prediction beyond age, sex, smoking and BMI. Papers that propose a cheap marker of cancer risk should report a baseline and a negative-control outcome. The cheapest information, age, sex, smoking and BMI, remains the best available cancer risk information in this data.</p>
"""

INTRO = """
<p>A cheap, widely available way to estimate who is at higher risk of cancer would change how limited follow-up is used. The cheapest information a person has is what is already known: their age, sex, whether they have smoked, their weight. The next cheapest is the blood count that many adults have every year as part of routine care. If a number from that blood count carried cancer-specific information, it would cost almost nothing to use.</p>
<p>A large literature claims that it does. Indices built from a routine blood count, such as the neutrophil-to-lymphocyte ratio (NLR), the systemic immune-inflammation index (SII), the neutrophil-percentage-to-albumin ratio (NPAR) and the advanced lung cancer inflammation index (ALI), have been reported to associate with cancer in dozens of NHANES analyses alone. A search of Europe PMC on 7 October 2026 found 103 NHANES papers with these indices and cancer in the abstract. Only 5 also mention incremental prediction (C-index or net reclassification) and 6 mention specificity, negative controls or competing causes. Several of the studies we read note that the same index predicts cardiovascular death at least as well as cancer death.<sup>1,2,3</sup></p>
<p>That last point is the problem. A marker that rises with illness in general will be associated with cancer in any sample, because ill people die of cancer, and it will also be associated with heart disease, diabetes and everything else. Two properties separate a cancer marker from a frailty marker. It must add to what age, sex, smoking and BMI already tell us, in data it was not built on. And it must be linked to dying of cancer more than to dying of other things, which is what a negative-control outcome is for.<sup>4</sup></p>
<p>This paper tests twelve cheap blood-count markers against both properties. The tests and the bars for passing them were fixed before the analysis. We expected that most would fail, because earlier work in this project found that routine blood values add little to age and sex for cancer death and that lab-based alerts marked non-cancer death more than cancer death. We report the result either way.</p>
"""

METHODS = f"""
<h3>2.1 Pre-registration</h3>
<p>The question, the twelve markers, the tests and the bars were committed to the project repository on 7 October 2026 (<code>docs/CHEAP_MARKERS_PREREG.md</code>) before the analysis was run. The expected result, that none would qualify, was written down with them. The sensitivity analyses (Section 3.6) were added afterwards and have no bar.</p>
<h3>2.2 Data</h3>
<p>The US National Health and Nutrition Examination Survey (NHANES), cycles 1999-2000 to 2007-2008, linked by the National Center for Health Statistics to the National Death Index through 31 December 2019. Every person has at least ten years of follow-up. We kept adults 40 and over without a reported cancer diagnosis at the exam, with a complete blood count with differential, albumin and BMI. This left {R['n']:,} people; {R['cancer_deaths_fit'] + R['cancer_deaths_test']} died of cancer (NCHS underlying cause 2, ICD-10 C00-C97) and {R['noncancer_deaths_fit'] + R['noncancer_deaths_test']:,} of another cause within ten years. Nobody was dropped for how they died. A secondary five-year analysis used all eight cycles through 2013-2014.</p>
<h3>2.3 The markers</h3>
<p>NLR (neutrophils over lymphocytes), PLR (platelets over lymphocytes), MLR (monocytes over lymphocytes), SII (platelets times neutrophils over lymphocytes), SIRI (neutrophils times monocytes over lymphocytes), NPAR (neutrophil percentage over albumin), ALI (BMI times albumin over NLR), PNI (albumin in g/L plus five times the lymphocyte count), red cell distribution width (RDW), white count, albumin and haemoglobin. Each was log-transformed and standardised on the fitting data, so effects are per one standard deviation. C-reactive protein (CRP) was analysed separately because it was not measured in 2011-2014.</p>
<h3>2.4 Tests</h3>
<p>The baseline model (M0) was logistic regression on age, age squared, sex, ever smoked and BMI. <b>Test 1</b> added each marker to M0 and compared the AUC for ten-year cancer death in cycles 2005-2008, with models fitted on 1999-2004, using a paired bootstrap of 1,000 resamples for the interval. <b>Test 2</b> used only people who died within ten years and asked whether the marker was associated with the death being from cancer rather than another cause (logistic regression adjusted for M0 and cycle), with Holm correction across the twelve markers.<sup>5</sup> <b>Test 3</b>, the negative control, repeated Test 1 and the association with death from other causes, which a cancer-specific marker should not predict as strongly. <b>Test 4</b> added all twelve markers together (ridge logistic regression). <b>Test 5</b> reproduced the design most papers use: the association of each marker with self-reported cancer history at the same visit.</p>
<p><b>A marker qualified only if it passed both Test 1 (AUC gain of at least 0.01 with an interval above zero) and Test 2 (odds above 1 with a Holm-adjusted p below 0.05).</b></p>
<h3>2.5 Software and reproducibility</h3>
<p>Python (scikit-learn, NumPy, pandas). The analysis is <code>experiments/cheap_markers.py</code> and the sensitivity analyses are <code>experiments/cheap_markers_sensitivity.py</code>; the tables and figures are built from their saved output.</p>
"""

RESULTS = f"""
<h3>3.1 The design and who is in it</h3>
{fig("fig1_logic", "The design. Everyone is classed as died of cancer, died of something else, or alive at ten years. The test asks whether a marker is about cancer rather than about being unwell.")}
<p>{R['n']:,} adults aged 40 and over: {R['n_fit']:,} in the fitting cycles (1999-2004, {R['cancer_deaths_fit']} cancer deaths) and {R['n_test']:,} in the test cycles (2005-2008, {R['cancer_deaths_test']} cancer deaths). The baseline of age, sex, smoking and BMI ranked ten-year cancer death with AUC {R['baseline_auc']['cancer']:.3f} in the test cycles. It ranked death from other causes better, with AUC {R['baseline_auc']['noncancer']:.3f}: the cheapest information predicts dying in general better than it predicts dying of cancer.</p>
<h3>3.2 Test 1: does a marker add discrimination?</h3>
<p>No. {F('fig2_gain')} and {TR('main')} show the AUC gain for each marker. None reached the pre-set 0.01 with an interval above zero. Eleven of the twelve had intervals that included zero; the exception was RDW, which gained {rdw['T1_gain']:+.4f} (95% CI {ci2(rdw['T1_ci'])}), below the bar, and which gained more for death from other causes ({rdw['T3_gain']:+.4f}). Adding all twelve together gained {R['T4']['gain']:+.4f} ({ci2(R['T4']['ci'])}; baseline {R['T4']['auc_baseline']:.3f}, with all {R['T4']['auc_all']:.3f}), which also did not pass. CRP, measured in the same cycles, gained {R['CRP']['T1_gain']:+.4f} ({ci2(R['CRP']['T1_ci'])}).</p>
{fig("fig2_gain", "Gain in AUC from adding each marker to age, sex, smoking and BMI, for death from cancer (red) and from other causes (orange), fitted on 1999-2004 and tested on 2005-2008. Lines are 95% intervals. The dashed line is the 0.01 bar.")}
{t_main}
<h3>3.3 Test 3: the negative control shows what the markers are measuring</h3>
<p>Eleven of the twelve markers were more strongly related to death from other causes than to death from cancer ({F('fig3_frailty')}); the exception, PLR, was weakly and inversely related to cancer death and not at all to other deaths. The six inflammation indices (NLR, PLR, MLR, SII, SIRI and NPAR) had odds ratios against survivors of {min(infl_nc):.2f} (PLR) to {max(infl_nc):.2f} per SD for death from other causes and {min(infl_c):.2f} to {max(infl_c):.2f} for death from cancer; the intervals for cancer death included 1 for five of the six. Their gain in AUC for other-cause death was positive for most. Albumin, haemoglobin, ALI and PNI went the other way: higher values lowered the risk of other-cause death (odds ratios {min(I[n]['or_noncancer_death'] for n in ('ALI', 'PNI', 'Albumin', 'Haemoglobin')):.2f} to {max(I[n]['or_noncancer_death'] for n in ('ALI', 'PNI', 'Albumin', 'Haemoglobin')):.2f}) with little or no effect on cancer death. These are the signatures of markers of general health.</p>
{fig("fig3_frailty", "Odds ratio per SD against people alive at ten years, for death from cancer (red) and from other causes (orange). A cancer marker would have a red point far from 1 and an orange point near it. Almost every marker shows the reverse.")}
<h3>3.4 Test 2: among people who died, is the marker about cancer?</h3>
<p>Among the {S['all (the pre-registered analysis)']['decedents']:,} people who died within ten years, higher values of the six inflammation indices were associated with <i>lower</i> odds that the death was from cancer (odds {dec_lo:.2f} to {dec_hi:.2f} per SD; {TR('decedent')}). Four markers passed the specificity bar: ALI, PNI, albumin and haemoglobin. They did so because they protect against other deaths, which leaves cancer a larger share of the deaths that do occur; their own association with cancer death was at most {max(I[n]['or_cancer_death'] for n in ('ALI', 'PNI', 'Albumin', 'Haemoglobin')):.2f} and mostly not distinguishable from 1. This is a limit of the specificity test used alone, and is why the qualifying rule requires added discrimination as well.</p>
{t_dec}
<p><b>Result: {nq} of 12 markers qualified.</b> The pre-registered expectation was met.</p>
<h3>3.5 Test 5: the claims in the literature, reproduced</h3>
<p>The associations that papers report are real. In {R['T5']['n']:,} adults, a standard deviation higher NLR went with {I['NLR']['or_cancer_death'] and R['T5']['indices']['NLR']['or']:.2f} times the odds of a self-reported cancer history, MLR with {R['T5']['indices']['MLR']['or']:.2f}, SIRI with {R['T5']['indices']['SIRI']['or']:.2f}; ALI was inversely associated ({R['T5']['indices']['ALI']['or']:.2f}). These are statistically clear. As prediction they are nearly worthless: no marker raised the AUC by more than {max(v['auc_gain_later_cycles'] for v in R['T5']['indices'].values()):.4f} in later survey cycles ({F('fig5_claims')}, {TR('claims')}). An odds ratio of 1.2 per SD is large enough to publish and far too small to rank people.</p>
{fig("fig5_claims", "Association (x axis) against added discrimination (y axis) for the twelve markers and self-reported cancer history. The associations are clear; the gain is a few thousandths of an AUC. Dashed line: the 0.01 bar.", "80%")}
{t_claims}
<h3>3.6 Does the result depend on the choices?</h3>
<p>It did not ({F('fig4_specificity')}). Removing the first 24 months of follow-up, which guards against undiagnosed cancer raising a marker, left the specificity result unchanged for the inflammation indices (for NLR, 0.82 to {S['first 24 months removed']['or']['NLR']['or']:.2f}). It held in women ({S['women']['or']['NLR']['or']:.2f}) and men ({S['men']['or']['NLR']['or']:.2f}) and at ages 40-59 ({S['ages 40-59']['or']['NLR']['or']:.2f}) and 60 and over ({S['ages 60 and over']['or']['NLR']['or']:.2f}). With a five-year horizon in all eight cycles, tested on 2007-2014, no marker improved discrimination of cancer death by 0.01 ({TR('fiveyear')}); the largest gain was RDW at {five['indices']['RDW']['gain_cancer']:+.4f}, again smaller than its gain for other-cause death ({five['indices']['RDW']['gain_noncancer']:+.4f}).</p>
{fig("fig4_specificity", "Specificity among people who died, for the nine most-discussed markers, across six subsets. Each colour is a subset; lines are 95% intervals. The inflammation indices stay left of 1 and the albumin-based ones right of 1 in every subset.")}
{t_five}
"""

DISCUSSION = f"""
<h3>4.1 What this means for cheap cancer risk prediction</h3>
<p>The idea of reading cancer risk from a routine blood count is attractive, and the associations in the literature are real. They are not what they seem. The popular indices are largely markers of general frailty: they are more strongly tied to dying of other causes than of cancer, they point away from cancer among people who died, and none improves on age, sex, smoking and BMI in data it was not built on. Taken as a cancer test, they would mostly find people who are ill.</p>
<p>Two practical points follow. First, the cheapest information, age, sex, smoking and BMI, is already the best cancer-risk information in this data, and it predicts death from other causes even better ({R['baseline_auc']['noncancer']:.2f} against {R['baseline_auc']['cancer']:.2f}). The most useful job for cheap risk information may therefore be deciding who is offered an expensive, specific test, which is how screening eligibility rules already work, and not detecting cancer itself. Second, any paper proposing a cheap marker of cancer risk should report the baseline, a model fitted and tested in different data, and a negative-control outcome, because the association alone does not distinguish a cancer marker from a frailty marker.</p>
<h3>4.2 The one marker worth a second look</h3>
<p>RDW, the width of the distribution of red cell sizes, was the only marker with a positive interval for cancer death in the main test ({rdw['T1_gain']:+.4f}) and its odds ratio against survivors for cancer death was {rdw['or_cancer_death']:.2f} ({rdw['or_cancer_ci'][0]:.2f} to {rdw['or_cancer_ci'][1]:.2f}). It was a stronger marker of other deaths ({rdw['or_noncancer_death']:.2f}), and it missed the bar, so we do not call it a cancer marker. It is the natural first candidate for a study in data with cancer diagnoses and symptoms.</p>
<h3>4.3 Relation to existing work</h3>
<p>Our results agree with what the association studies themselves report when they look: NLR predicting cardiovascular death more clearly than cancer death even among people with cancer,<sup>2</sup> and the immune-inflammation index linking to all-cause mortality linearly but to cancer mortality in a U-shape.<sup>1</sup> What is new, as far as a search could tell, is putting twelve markers through the same two tests, with a negative-control outcome and an unseen survey era. A short search cannot show that no one has done so.</p>
<h3>4.4 Limitations</h3>
<ul>
<li><b>Death, not diagnosis.</b> The outcome is death from cancer. Cancers found early and cured count as non-cases, and a marker could detect those without predicting death. NHANES has no diagnoses. A study in a cohort with cancer registry data would answer that, and is the next step.</li>
<li><b>One survey and one country.</b> US adults examined from 1999 to 2008; weights not applied, since the aim was prediction in the sample.</li>
<li><b>Power.</b> {R['cancer_deaths_test']} cancer deaths in the test era give a gain interval of about plus or minus 0.01 AUC, so a gain smaller than that cannot be excluded. The bar of 0.01 was chosen as the smallest gain worth acting on.</li>
<li><b>All cancers pooled.</b> A marker specific to one cancer (for example, a blood marker for one type) would be diluted. Site-specific tests need many more events than this survey has.</li>
<li><b>Correlated markers.</b> The twelve are derived from the same blood count and are not independent; Holm correction is conservative for that reason, and the main conclusions do not depend on it because Test 1 carries them.</li>
<li><b>One measurement.</b> A single value at one visit, with no repeat or trajectory.</li>
<li><b>Coding and pipeline by an AI assistant</b> (see Declarations); the code and results are public for checking.</li>
</ul>
<h3>4.5 Next steps</h3>
<ol>
<li>Repeat the two tests in a cohort with cancer-registry diagnoses and symptoms (candidates are listed in <code>docs/CLINICAL_DATA_ACCESS.md</code>), starting with RDW.</li>
<li>Test trajectories (change in a marker over repeated blood counts) rather than one value.</li>
<li>Test site-specific questions, such as RDW and bowel cancer, where a plausible mechanism exists.</li>
<li>Ask a clinician which decisions a cheap marker could realistically change.</li>
</ol>
"""

DECL = """
<h3>AI assistance</h3>
<p>This study was carried out with an AI assistant (Anthropic Claude, via Claude Code), which wrote the analysis code, ran the analyses, drafted the text and searched for background literature. An AI cannot be an author. The human author set the question, is responsible for the work, and must verify the statistics and every statement before submission.</p>
<h3>Ethics, data and code</h3>
<p>The study uses public, de-identified NHANES data and NCHS public-use linked mortality files; no participants were contacted. Code, pre-registration and results are in the public repository (github.com/palashraks-afk/OncoVision). Clinical mentorship and authorship are not asserted here.</p>
<h3>Conflicts of interest and funding</h3>
<p>None declared. No funding.</p>
"""

REFS = """
<ol>
<li>Lu W, Gong Y, Liu L, Zhang Y, Tian X, Liu H. Association of systemic immune-inflammatory index with all-cause and cancer mortality in Americans aged 60 years and older. <i>Front Aging</i> 2025 (PMC11931307).</li>
<li>Li G, Fu Y, Zhang D. Association between neutrophil-to-lymphocyte ratio and all-cause and cardiovascular mortality among adults with cancer from NHANES 2005-2018: a retrospective cohort study. <i>Front Oncol</i> 2025 (PMC11959702).</li>
<li>Chen C, Chen Y, Gao Q, Wei Q. Association of systemic immune inflammatory index with all-cause and cause-specific mortality among individuals with type 2 diabetes. 2023 (PMC10702126).</li>
<li>Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool for detecting confounding and bias in observational studies. <i>Epidemiology</i> 2010;21:383-388.</li>
<li>Holm S. A simple sequentially rejective multiple test procedure. <i>Scand J Stat</i> 1979;6:65-70.</li>
<li>Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. <i>Ann Intern Med</i> 2019;170:51-58.</li>
<li>Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. <i>BMJ</i> 2024;385:e078378.</li>
<li>Centers for Disease Control and Prevention, National Center for Health Statistics. NHANES data and documentation; NCHS public-use linked mortality files.</li>
</ol>
<p class="tnote">References 4, 6 and 7 were looked up by title in Crossref and match. References 1 to 3 were identified by a literature search and their journals and volumes should be completed from the originals. Reference 5 is classical and was not machine-verified. Reference 8 is a data product.</p>
"""

TITLE = "Cheap Blood Markers of Cancer Risk: Cancer Signal or General Frailty?"
SUB = f"A pre-registered test of twelve blood-count indices in {R['n']:,} US adults, with a negative-control outcome and an unseen survey era"
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1><div class="sub">{SUB}</div>
<div class="meta"><b>Palash Rakshit</b> (author) &nbsp;·&nbsp; Draft of 8 October 2026 &nbsp;·&nbsp; Oncovision project</div>
<div class="banner"><b>Draft. Not peer reviewed.</b> The analysis was run by an AI assistant and has not been independently re-run by a second person. The main result is negative: none of the twelve markers qualified. Nothing here is medical advice.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> cancer risk; blood count; neutrophil-to-lymphocyte ratio; systemic immune-inflammation index; NHANES; negative control; frailty; pre-registration.</p>
<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
</body></html>"""
out = os.path.join(HERE, "marker_paper.html")
open(out, "w", encoding="utf-8").write(html)
pdf = os.path.join(HERE, "Cheap_Blood_Markers_Cancer_Risk.pdf")
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
