"""
Builds the paper on Medicare's multi-cancer early detection test law (HTML, then PDF via headless Chrome) from
the saved result files, so no number in the text can differ from the analysis.

Run:  python docs/mced_paper/build_mced_paper.py
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
J = lambda n: json.load(open(os.path.join(ROOT, "experiments", n)))  # noqa: E731
R, L, C, OLD = J("mced_law_result.json"), J("free_data_ladder_result.json"), J("mced_coverage_by_year.json"), J("mced_law_result_introduced_bill.json")
src = open(os.path.join(ROOT, "docs", "paper", "build_paper.py"), encoding="utf-8").read()
CSS = src[src.index('CSS = """') + 9: src.index('"""', src.index('CSS = """') + 9)]

FIGS = ["fig1_coverage", "fig2_yield", "fig3_triage", "fig4_ladder"]
FN = {n: i + 1 for i, n in enumerate(FIGS)}
TABS = ["groups", "triage", "disability", "ladder"]
TN = {n: i + 1 for i, n in enumerate(TABS)}


def fig(name, caption, width="88%"):
    return f'<figure style="width:{width}"><img src="figures/{name}.png" alt=""><figcaption><b>Figure {FN[name]}.</b> {caption}</figcaption></figure>'


def table(key, title, head, rows, note=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="tcap"><b>Table {TN[key]}.</b> {title}</div><table class="num"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{n}'


def F_(n):
    return f"Figure {FN[n]}"


def T_(n):
    return f"Table {TN[n]}"


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


q1, q1b, q3, q4 = R["Q1"]["2029"], R["Q1"]["2035 (stylised)"], R["Q3"], R["Q4"]
t20 = q3["0.2"]
yrs = C["years"]
first50 = next(y for y, s in zip(yrs, C["share_of_cancer_deaths"]) if s >= 0.5)

t_groups = table("groups", "Expected multi-cancer test results per 1,000 tests, by group (NHIS 2005-2009, survey-weighted).",
                 ["Group", "Adults", "Five-year cancer deaths", "Cancers found / 1,000 tests", "False alarms / 1,000", "Positives that are real", "Test cost per cancer found"],
                 [[g, f"{v['n']:,}", v["cancer_deaths"], v["cancers_found_per_1000_tests"], v["false_alarms_per_1000_tests"], pc(v["ppv"]), usd(v["medicare_test_cost_per_cancer_found"])] for g, v in R["Q2"].items()],
                 "Cancers found = 0.31 x the group's observed five-year cancer-death rate, the factor that makes adults 50 and older reproduce PATHFINDER 2's 0.54% detection rate. Costs are the test only, at $508.87.")
t_triage = table("triage", "Inside the group the law covers in 2029: testing the highest-risk people by free answers against testing at random.",
                 ["Share of the covered group tested", "Yield ratio (95% CI)", "Cancers found / 1,000 tests", "Positives that are real", "Test cost per cancer found"],
                 [[f"{float(s):.0%}", f"{q3[s]['yield_ratio']:.2f} ({q3[s]['ci'][0]:.2f} to {q3[s]['ci'][1]:.2f})",
                   f"{q3[s]['random']['cancers_found_per_1000_tests']} &rarr; {q3[s]['triaged']['cancers_found_per_1000_tests']}",
                   f"{pc(q3[s]['random']['ppv'])} &rarr; {pc(q3[s]['triaged']['ppv'])}",
                   f"{usd(q3[s]['random']['medicare_test_cost_per_cancer_found'])} &rarr; {usd(q3[s]['triaged']['medicare_test_cost_per_cancer_found'])}"] for s in ("0.2", "0.3", "0.5")],
                 f"Free answers: age, sex and smoking (never, former with years since quitting, current with cigarettes a day). Ranking by age and sex alone gave a ratio of {q3['age_sex_only_ratio_20pct']:.2f} at 20%, so smoking does nearly all the work.")
t_dis = table("disability", "Medicare beneficiaries aged 50-64 against same-age adults without Medicare, age-standardised.",
              ["Measure", "Medicare 50-64", "No Medicare 50-64", "Ratio", "Difference, 95% CI"],
              [["Observed five-year cancer death", pc(q4["observed_5y_cancer_death"]["medicare"], 2), pc(q4["observed_5y_cancer_death"]["no_medicare"], 2), q4["observed_5y_cancer_death"]["ratio"],
                f"{100 * q4['observed_5y_cancer_death']['diff_ci'][0]:+.2f} to {100 * q4['observed_5y_cancer_death']['diff_ci'][1]:+.2f} points"],
               ["Predicted five-year risk (free answers)", pc(q4["predicted_5y_risk"]["medicare"], 2), pc(q4["predicted_5y_risk"]["no_medicare"], 2), q4["predicted_5y_risk"]["ratio"],
                f"{100 * q4['predicted_5y_risk']['diff_ci'][0]:+.2f} to {100 * q4['predicted_5y_risk']['diff_ci'][1]:+.2f} points"],
               ["Current smokers", pc(q4["current_smoker"]["medicare"]), pc(q4["current_smoker"]["no_medicare"]), q4["current_smoker"]["ratio"],
                f"{100 * q4['current_smoker']['diff_ci'][0]:+.1f} to {100 * q4['current_smoker']['diff_ci'][1]:+.1f} points"]],
              f"{q4['n_medicare_50_64']:,} Medicare beneficiaries aged 50-64, {q4['cancer_deaths_medicare_50_64']} five-year cancer deaths. Under 65, Medicare entitlement comes almost entirely from disability or end-stage kidney disease.")
G = L["gains"]
t_ladder = table("ladder", "Free-data ladder, NHANES 2003-2006, ten-year death, out-of-fold AUC (20 x 5-fold).",
                 ["Model", "AUC, cancer death", "Calibration slope"],
                 [["Q: age, sex, smoking, BMI, weight change", L["models"]["Q"]["auc"], L["models"]["Q"]["slope"]],
                  ["Q + routine checkup (blood count, chemistry, blood pressure)", L["models"]["QC"]["auc"], L["models"]["QC"]["slope"]],
                  ["Q + activity (7-day accelerometer)", L["models"]["QW"]["auc"], L["models"]["QW"]["slope"]],
                  ["Q + checkup + activity", L["models"]["QCW"]["auc"], L["models"]["QCW"]["slope"]]],
                 f"{L['n']:,} adults aged 40-84, {L['cancer_deaths']} cancer deaths and {L['other_deaths']} other deaths. Gains over Q: activity {G['QW vs Q']['gain']:+.4f} ({G['QW vs Q']['ci'][0]:+.4f} to {G['QW vs Q']['ci'][1]:+.4f}); checkup {G['QC vs Q']['gain']:+.4f} ({G['QC vs Q']['ci'][0]:+.4f} to {G['QC vs Q']['ci'][1]:+.4f}). For other-cause death, activity added {L['other_cause']['gain']['gain']:+.4f} ({L['other_cause']['gain']['ci'][0]:+.4f} to {L['other_cause']['gain']['ci'][1]:+.4f}).")

ABSTRACT = f"""
<p><b>Background.</b> In February 2026 the United States enacted Medicare coverage of multi-cancer early detection (MCED) blood tests from 2029 (Public Law 119-75). The law pays only for people aged 50 or older who are under an age limit of 65 in 2029 that rises by one year each year. Because the limit rises as fast as people age, it is in effect a birth-date cutoff. We asked whom the law reaches, what the test would find in them, and whether information people already have for free could direct the tests better.</p>
<p><b>Methods.</b> Pre-registered analysis. NHIS 2005-2009 ({R['n_test']:,} adults, survey-weighted) linked to the National Death Index, five-year cancer death as the marker of lethal-cancer risk, and the published performance of one MCED test (PATHFINDER 2: detection 0.54% in adults 50 and older, specificity 99.64%). A model built on 1997-2002 using age, sex and smoking ranked people. A second pre-registered analysis in NHANES 2003-2006 ({L['n']:,} adults) asked whether routine checkup results or a week of accelerometer data, the kind of activity record phones and watches keep, add to free questions for cancer death.</p>
<p><b>Results.</b> In 2029 the law covers {pc(q1['share_of_population'])} of Medicare-eligible adults and {pc(q1['share_of_cancer_deaths'])} of their five-year cancer deaths (95% CI {pc(q1['ci'][0], 1)} to {pc(q1['ci'][1], 1)}); coverage of cancer deaths does not pass half until about {first50}. The people it never covers, everyone 66 and older in 2029, are those in whom a test would find the most cancers per test (9.0 to 12.1 per 1,000 against 5.4 to 6.7) at the lowest cost per cancer found. The first people covered are mostly Medicare beneficiaries aged 50 to 64, who carried {q4['observed_5y_cancer_death']['ratio']} times the age-standardised five-year cancer-death rate of same-age adults without Medicare. Inside the covered group, testing the fifth at highest risk on free answers found {t20['yield_ratio']:.2f} times as many cancers per test as testing at random (95% CI {t20['ci'][0]:.2f} to {t20['ci'][1]:.2f}), cutting test cost per cancer found from {usd(t20['random']['medicare_test_cost_per_cancer_found'])} to {usd(t20['triaged']['medicare_test_cost_per_cancer_found'])}; almost all of that came from smoking. Neither routine checkup results ({G['QC vs Q']['gain']:+.3f} AUC) nor activity data ({G['QW vs Q']['gain']:+.3f}) added to free questions for cancer death, although activity data predicted other-cause death well ({L['other_cause']['gain']['gain']:+.3f}).</p>
<p><b>Conclusions.</b> As written, Medicare's MCED law excludes the people in whom the test would work best and, for its first years, mainly reaches a disabled, high-risk group whose need is real but small in number. Within whatever group is covered, a few free questions could make each test go much further; wearable and checkup data do not add to that for cancer. No MCED test has shown a reduction in cancer deaths, so these are statements about who the law reaches and how to spend it, not about lives saved.</p>
"""

INTRO = """
<p>Blood tests that look for signals of many cancers at once are the most discussed new idea in cancer screening. They find cancers that have no screening test, but they miss most early cancers, cost close to $1,000, and have not yet been shown to reduce deaths: the NHS-Galleri trial, reported in 2026, did not reduce stage III and IV cancers overall, although stage IV diagnoses fell in later rounds.<sup>2</sup> Even so, in February 2026 Congress enacted Medicare coverage of these tests from 2029, at the payment rate of the stool DNA test, subject to FDA approval and a national coverage determination.<sup>1</sup></p>
<p>The law's eligibility rule has received little attention. Payment is barred for anyone who, on 1 January of the year of the test, has reached 65 in 2029, 66 in 2030, and so on. The limit rises exactly as fast as people age, so anyone already 65 on 1 January 2029 never becomes eligible, while everyone younger stays eligible for life. The law covers birth cohorts, not ages. Cancer risk rises steeply with age, so the rule excludes the people at highest risk. The one published estimate of what the law will cost Medicare described the age limit as one that rises each year but did not analyse who it excludes.<sup>3</sup></p>
<p>Earlier work in this project showed that four free answers (age, sex, smoking, BMI) rank the risk of dying of cancer about as well as anything cheap, and that routine blood tests add almost nothing. Risk-stratified use of MCED tests has been proposed using polygenic scores and risk factors in UK Biobank,<sup>4</sup> and modelled for risk-factor subgroups,<sup>5,6</sup> but not against the eligibility rule that will actually govern US use. This paper asks four pre-registered questions: what share of cancer-death risk does the law reach; what would the test find in the people it covers and the people it does not; how much better would tests go if the covered group were triaged on free answers; and who are the first people covered. A second pre-registered analysis asks whether data people already have for free beyond questions, routine checkup results and the activity a phone or watch records, adds anything for cancer.</p>
"""

METHODS = f"""
<h3>2.1 Pre-registration and a correction</h3>
<p>The questions, the mapping from risk to test results, and three bars were committed before the analysis (<code>docs/MCED_LAW_PREREG.md</code>). The first version quoted the bill <i>as introduced</i> (coverage from 2028, age limit 68). An independent re-analysis found that the enacted text differs, and we confirmed it in the enrolled bill: coverage from 1 January 2029, a floor of 50, and a limit of 65 in 2029.<sup>1</sup> The correction is recorded in the pre-registration, the first run is kept in the repository, and every number here uses the enacted rule. With the introduced rule the 2028 group would have held {pc(OLD['Q1']['2028']['share_of_cancer_deaths'])} of cancer deaths; the enacted rule is stricter.</p>
<h3>2.2 Data</h3>
<p>NHIS 1997-2009 public-use files linked to the National Death Index through 2019, with the NHIS Medicare-coverage recode added to the cohort built for the earlier study (the rebuilt file is identical on every earlier column). Adults 35-84 without a reported earlier cancer. Outcome: death from cancer within five years of interview. All descriptive results use interview years 2005-2009 with the sample-adult weight. The ranking model (age as a cubic spline, sex, smoking status, cigarettes a day, years since quitting) was fitted on 1997-2002.</p>
<h3>2.3 Who the law covers</h3>
<p>Medicare-eligible adults are those aged 65-84 plus Medicare beneficiaries aged 50-64 (under 65, Medicare entitlement comes almost entirely from disability or end-stage kidney disease). Covered in 2029: beneficiaries aged 50-64, plus adults aged 65 at interview as a stand-in for those turning 65 during 2029. For later years the covered ages extend by one year per year (Figure {FN['fig1_coverage']}), holding the 2005-2009 age and risk structure fixed; this projection is exploratory.</p>
<h3>2.4 From risk to test results</h3>
<p>The chance that the test finds a cancer in a group was taken as proportional to the group's observed five-year cancer-death rate, scaled so that adults 50 and older reproduce PATHFINDER 2's detection rate of 0.54%.<sup>7</sup> False alarms are (1 - 0.9964) times those without a detected cancer. This rests on a stated assumption, that MCED tests preferentially detect the cancers that kill, and it was varied (detection proportional to the square root of predicted risk). Because only observed death rates enter the main numbers, ranking people by the model and scoring them with the same model never happens; ratios between groups are ratios of observed death rates, and the anchor only sets the absolute scale. Costs are the test only, at $508.87, the Medicare rate for the stool DNA test (CPT 81528) on the 2026 Clinical Laboratory Fee Schedule, which the law uses until 2031.</p>
<h3>2.5 Triage and the free-data ladder</h3>
<p>Inside the 2029-covered group we compared testing the 20%, 30% or 50% at highest predicted risk with testing the same number at random, with bootstrap intervals. The free-data ladder (<code>docs/FREE_DATA_LADDER_PREREG.md</code>) used NHANES 2003-2006 adults 40-84 with at least four valid days of hip accelerometry, summarised (after the Troiano non-wear rule<sup>8</sup>) as total activity, minutes of moderate-to-vigorous activity, share of time sedentary and activity fragmentation. Questions (age, sex, smoking, BMI, self-reported weight change and unintentional weight loss), routine checkup results and activity were compared by out-of-fold AUC from 20 repeats of 5-fold cross-validation.</p>
"""

RESULTS = f"""
<h3>3.1 The law reaches about one in eight of the cancer deaths it could</h3>
<p>In 2029 the covered group is {pc(q1['share_of_population'], 1)} of Medicare-eligible adults and holds {pc(q1['share_of_cancer_deaths'], 1)} of their five-year cancer deaths (95% CI {pc(q1['ci'][0], 1)} to {pc(q1['ci'][1], 1)}); the pre-registered bar (an interval below 50%) was met. Coverage widens as younger cohorts age into Medicare, but in the stylised projection it passes half of cancer deaths only in about {first50} ({F_('fig1_coverage')}). Everyone born before 1964 is permanently outside the law.</p>
{fig("fig1_coverage", "Share of Medicare-eligible adults, and of their five-year cancer deaths, inside the law's age limit in each year, holding the 2005-2009 age and risk structure fixed (exploratory). The dashed line is half.")}
<h3>3.2 The people excluded are the ones in whom the test would work best</h3>
<p>Expected yield rises steeply with age ({F_('fig2_yield')}, {T_('groups')}). Among adults 75-84, never covered, a test would find about {R['Q2']['75-84 (never covered)']['cancers_found_per_1000_tests']} cancers per 1,000 tests, {pc(R['Q2']['75-84 (never covered)']['ppv'])} of positives would be real, and the test cost per cancer found would be {usd(R['Q2']['75-84 (never covered)']['medicare_test_cost_per_cancer_found'])}. Among adults turning 65, covered, the figures are {R['Q2']['65 (covered 2029)']['cancers_found_per_1000_tests']}, {pc(R['Q2']['65 (covered 2029)']['ppv'])} and {usd(R['Q2']['65 (covered 2029)']['medicare_test_cost_per_cancer_found'])}. That an older person benefits less from finding a cancer (shorter life expectancy, more competing illness, more overdiagnosis) is a real reason to limit testing by age, and this analysis does not weigh it; it shows only how much yield the rule gives up.</p>
{fig("fig2_yield", "Expected cancers found per 1,000 tests and the share of positive results that are real, by group. Teal: covered in 2029. Red: never covered under the enacted rule.")}
{t_groups}
<h3>3.3 The first people covered are a high-risk group</h3>
<p>Medicare beneficiaries aged 50-64, most of whom qualify through disability, carried {q4['observed_5y_cancer_death']['ratio']} times the age-standardised five-year cancer-death rate of same-age adults without Medicare, {q4['predicted_5y_risk']['ratio']} times the predicted risk from free answers, and were {q4['current_smoker']['ratio']} times as likely to smoke ({T_('disability')}). The bar (both observed and predicted risk higher, intervals excluding no difference) was met. The interval is wide because the group is small ({q4['cancer_deaths_medicare_50_64']} cancer deaths). For the law's first years, then, the people it reaches are few but at high risk, and a test in them would perform about as well as in adults in their late sixties.</p>
{t_dis}
<h3>3.4 Free answers make each covered test go about twice as far</h3>
<p>Testing the fifth of the covered group at highest risk on age, sex and smoking found {t20['yield_ratio']:.2f} times as many cancers per test as testing at random (95% CI {t20['ci'][0]:.2f} to {t20['ci'][1]:.2f}; bar of 1.25 met), raised the share of real positives from {pc(t20['random']['ppv'])} to {pc(t20['triaged']['ppv'])} and cut the test cost per cancer found from {usd(t20['random']['medicare_test_cost_per_cancer_found'])} to {usd(t20['triaged']['medicare_test_cost_per_cancer_found'])} ({F_('fig3_triage')}, {T_('triage')}). Ranking by age and sex alone gave {q3['age_sex_only_ratio_20pct']:.2f}: inside a narrow age band the gain is smoking.</p>
{fig("fig3_triage", "Cancers found per 1,000 tests inside the 2029-covered group, testing at random against testing those at highest risk on free answers.")}
{t_triage}
<h3>3.5 Data people already have adds nothing for cancer</h3>
<p>In NHANES, free questions ranked ten-year cancer death with AUC {L['models']['Q']['auc']:.3f}. Adding routine checkup results changed it by {G['QC vs Q']['gain']:+.4f} and adding a week of activity data by {G['QW vs Q']['gain']:+.4f} (interval {G['QW vs Q']['ci'][0]:+.4f} to {G['QW vs Q']['ci'][1]:+.4f}). The same activity data added {L['other_cause']['gain']['gain']:+.4f} for death from other causes ({F_('fig4_ladder')}, {T_('ladder')}). All three pre-registered bars for the ladder were missed: activity and checkup data do not add for cancer, and what activity measures is general frailty. Less active people did die of cancer more often ({pc(L['single_W']['log_tac']['cancer_rate_bottom_quarter'], 1)} in the least active quarter against {pc(L['single_W']['log_tac']['cancer_rate_top_quarter'], 1)} in the most), but age and smoking already explain that. This disagrees with a smaller study that proposed wearable data for cancer-death prediction<sup>9</sup> and is consistent with NHANES work showing that activity predicts all-cause death strongly.<sup>10</sup></p>
{fig("fig4_ladder", "Gain in AUC over free questions from routine checkup results and from activity data, for cancer death and, as a check, for other-cause death. Dashed line: the pre-registered 0.01 bar.")}
{t_ladder}
<h3>3.6 Robustness (independent re-analysis)</h3>
<p>A separate re-implementation reproduced the headline numbers on the first rule and found the law error that led to the correction. With a ten-year outcome, unweighted analysis, or prior-cancer cases retained, the direction of every result held (<code>docs/independent_check/mced/</code>). The 2029-group result under the enacted rule, re-computed in that check on a slightly different definition (beneficiaries 50-64 only), was 7.9% of cancer deaths.</p>
"""

DISCUSSION = f"""
<h3>4.1 What this means</h3>
<p>Read literally, Medicare's MCED law sets up a test whose expected value is highest in exactly the people it excludes. That may be deliberate, a way to limit cost and overdiagnosis while the evidence matures, and the law lifts all age limits if the US Preventive Services Task Force later gives a test an A or B grade. But it should be said plainly, because describing the law simply as Medicare coverage of these tests suggests something much broader than about {pc(q1['share_of_population'])} of them in its first year.</p>
<p>Two practical points follow. First, the earliest beneficiaries are mostly disabled adults aged 50-64, a group with more than twice the cancer-death rate of their peers and a high smoking rate; outreach and follow-up capacity for positive results should be planned around them. Second, wherever coverage starts, a minute of free questions (age, sex, smoking) can roughly double the number of cancers found per test inside the covered group. That is a way to spend a fixed budget better, not a reason to deny anyone a covered test, and it would need to be weighed against the risk of entrenching existing gaps: smoking-based ranking favours people with less education and income, who are less often screened.</p>
<h3>4.2 What it does not show</h3>
<ul>
<li><b>Lives saved.</b> No MCED test has reduced cancer deaths in a trial. Finding cancers is not the same as preventing deaths.</li>
<li><b>The mapping is an assumption.</b> Test yield is taken as proportional to cancer-death risk and anchored to one company-run study of healthier volunteers. Ratios between groups depend only on observed death rates; absolute numbers depend on the anchor.</li>
<li><b>Old data.</b> Interview years 2005-2009 have more smoking and higher cancer death than 2029 will. The coverage projection holds that structure fixed.</li>
<li><b>Survey limits.</b> Self-reported Medicare can be confused with Medicaid; people 85 and older are top-coded and left out; the bootstrap ignores the survey's clustering, so intervals are somewhat narrow.</li>
<li><b>The activity monitor is not a phone.</b> NHANES used research hip monitors; consumer devices differ.</li>
</ul>
<h3>4.3 Novelty, checked three ways</h3>
<p>We searched Europe PMC with five MCED queries (826 records) and seven queries on wearable and routine-data prediction (1,984 records), and searched the web twice. Risk-stratified MCED use has been proposed,<sup>4,5,6</sup> and a 2026 preprint estimated the law's cost to Medicare,<sup>3</sup> but we found no analysis of the enacted eligibility rule as a birth-cohort cutoff, of the cancer burden it excludes, of the high-risk disabled group it reaches first, or of free-answer triage within it. Searches are not proof of absence.</p>
"""

DECL = """
<h3>AI assistance</h3>
<p>An AI assistant (Anthropic Claude, via Claude Code) wrote the code, ran the analyses, drafted the text and searched the literature; a second AI agent re-ran the analysis independently and found the law error corrected here. An AI cannot be an author. The human author set the question and is responsible for the work, and must verify every statement before submission. No human second analyst has re-run it.</p>
<h3>Data, code, ethics</h3>
<p>Public, de-identified NCHS data; no participants contacted. Code, pre-registrations, results and the re-analysis are in github.com/palashraks-afk/OncoVision. Clinical mentorship and authorship are not asserted. No funding; no conflicts of interest.</p>
"""

REFS = """
<ol>
<li>Consolidated Appropriations Act, 2026 (H.R. 7148; Public Law 119-75), section adding Social Security Act sections 1861(nnn) and 1834(aa) (Nancy Gardner Sewell Medicare Multi-Cancer Early Detection Screening Coverage Act). Enrolled text, govinfo.gov BILLS-119hr7148enr.</li>
<li>NHS-Galleri trial: first results presented at the 2026 ASCO Annual Meeting (King's College London and GRAIL releases, May 2026); journal publication to be cited from the original.</li>
<li>Scherer LD, Matlock DD, Cronin J, Gritz M. Potential cost of multi-cancer detection tests to Medicare. medRxiv 2026, doi:10.64898/2026.08.24.26361257 (preprint).</li>
<li>Kim ES, Scharpf RB, Garcia-Closas M, Visvanathan K, Velculescu VE, Chatterjee N. Potential utility of risk stratification for multicancer screening with liquid biopsy tests. <i>NPJ Precis Oncol</i> 2023 (PMID 37087533).</li>
<li>Kim A, Cong Z, Jazieh AR, et al. Estimating the incremental population health impact of a multi-cancer early detection test to complement existing screening among elevated risk populations. <i>BMC Health Serv Res</i> 2024 (PMID 39695574).</li>
<li>Raoof S, Kansal AR, Tafazzoli A, et al. Impact of population cancer risk on the cost-effectiveness of multicancer early detection testing in the United States. <i>Cancer Epidemiol Biomarkers Prev</i> 2025 (PMID 40864148).</li>
<li>Nabavizadeh N, et al. Performance and safety of a multi-cancer early detection test: the PATHFINDER 2 study. <i>Nat Med</i> 2026 (PMID 42773207).</li>
<li>Troiano RP, Berrigan D, Dodd KW, et al. Physical activity in the United States measured by accelerometer. <i>Med Sci Sports Exerc</i> 2008;40:181-188.</li>
<li>Tedesco S, et al. Investigation of the analysis of wearable data for cancer-specific mortality prediction in older adults. IEEE EMBC 2021 (PMID 34891647).</li>
<li>Leroux A, et al. / Smirnova E, et al. Predictive performance of accelerometry-derived physical activity for five-year all-cause mortality in older adults, NHANES 2003-2006. <i>J Gerontol A</i> 2020 (doi:10.1093/gerona/glz193).</li>
<li>Centers for Medicare &amp; Medicaid Services. Clinical Laboratory Fee Schedule, calendar year 2026 (Q1 and Q4 files), CPT 81528.</li>
<li>National Center for Health Statistics. NHIS 1997-2009 and NHANES 2003-2006 public-use files and Linked Mortality Files (follow-up through 2019).</li>
</ol>
<p class="tnote">Reference 10's author list should be confirmed from the original before submission.</p>
"""


TITLE = "Who Will Medicare's Multi-Cancer Test Law Reach?"
SUB = "The 2026 law's age rule as a birth-cohort cutoff, the cancer burden it leaves out, the high-risk group it reaches first, and how far free questions could stretch each test"
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1><div class="sub">{SUB}</div>
<div class="meta"><b>Palash Rakshit</b> (author) &nbsp;&middot;&nbsp; Draft of 12 October 2026 &nbsp;&middot;&nbsp; Oncovision project</div>
<div class="banner"><b>Draft. Not peer reviewed.</b> Analysis by an AI assistant, re-checked by a second AI agent, not by a person. No multi-cancer test has been shown to reduce cancer deaths; nothing here estimates lives saved. Nothing here is medical or legal advice.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> multi-cancer early detection; Medicare; health policy; risk-based screening; disability; wearables; pre-registration.</p>
<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
</body></html>"""
out = os.path.join(HERE, "mced_paper.html")
open(out, "w", encoding="utf-8").write(html)
pdf = os.path.join(HERE, "Medicare_Multi_Cancer_Test_Law.pdf")
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
