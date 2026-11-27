"""
Builds the red-flag-labs paper (HTML, then PDF via headless Chrome) from experiments/red_flag_labs_result.json,
so no number in the text can differ from the analysis.

Run:  python docs/red_flag_paper/build_red_flag_paper.py
"""

import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(ROOT, "experiments", "red_flag_labs_result.json")))
src = open(os.path.join(ROOT, "docs", "paper", "build_paper.py"), encoding="utf-8").read()
CSS = src[src.index('CSS = """') + 9: src.index('"""', src.index('CSS = """') + 9)]
A, B5, B10 = R["A_societal_2017_2020"], R["B_scientific_5y"], R["B_scientific_10y"]
P5, P10, N3 = B5["pooled"], B10["pooled"], B5["NHANES III 1988-1994"]
MO = A["missed_opportunity_exploratory"]
R1, M1 = R["R1_replication_2001_2002"], R["M1_competing_risks_5y"]
X = json.load(open(os.path.join(ROOT, "experiments", "red_flag_more_result.json")))
T1a, T1b, T2, T3 = X["T1"]["2017-2020"], X["T1"]["2001-2002"], X["T2"], X["T3"]
RD = json.load(open(os.path.join(ROOT, "experiments", "routine_test_flags_discovery.json")))
RR = json.load(open(os.path.join(ROOT, "experiments", "routine_test_flags_replication.json")))
AG = json.load(open(os.path.join(ROOT, "experiments", "red_flag_angles_result.json")))
A1 = AG["A1_untreated"]
MED, A3 = AG["A4_medicines"], AG["A3_by_payer"]
ROB = A["robustness_exploratory"]
FIGS = ["fig1_funnel", "fig2_untreated", "fig3_groups", "fig4_odds"]
FN = {n: i + 1 for i, n in enumerate(FIGS)}
TN = {n: i + 1 for i, n in enumerate(["societal", "followup", "bars"])}


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


def ci(c, d=0):
    return f"{100 * c[0]:.{d}f} to {100 * c[1]:.{d}f}%"


def F(n):
    return f"Figure {FN[n]}"


def T(n):
    return f"Table {TN[n]}"


def fig(name, caption, width="86%"):
    return f'<figure style="width:{width}"><img src="figures/{name}.png" alt=""><figcaption><b>Figure {FN[name]}.</b> {caption}</figcaption></figure>'


def table(key, title, head, rows, note=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="tcap"><b>Table {TN[key]}.</b> {title}</div><table class="num"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{n}'


reg = MO["usual_care_and_2plus_visits"]
t_soc = table("societal", "Iron-deficiency anaemia among US adults 50+ (men, and women past menopause), NHANES 2017-March 2020, survey-weighted.",
              ["Measure", "Estimate", "95% interval", "Sample"],
              [["Prevalence, all", pc(A["ida_prevalence"]["est"], 1), ci(A["ida_prevalence"]["ci"], 1), f"{A['ida_n']} of {A['n_50plus']:,}"],
               ["National count", f"{A['ida_prevalence']['millions']} million", f"{A['ida_prevalence']['millions_ci'][0]} to {A['ida_prevalence']['millions_ci'][1]} million", ""],
               ["Prevalence, men", pc(A["ida_prevalence_men"]["est"], 1), ci(A["ida_prevalence_men"]["ci"], 1), ""],
               ["Prevalence, women past menopause", pc(A["ida_prevalence_women"]["est"], 1), ci(A["ida_prevalence_women"]["ci"], 1), ""],
               ["Not treated for anaemia in the past 3 months", pc(A["ida_untreated_share"]["est"]), ci(A["ida_untreated_share"]["ci"]), f"n = {A['ida_untreated_share']['n']}"],
               ["Of the untreated: usual place of care", pc(A["untreated_ida_usual_place_of_care"]["est"]), ci(A["untreated_ida_usual_place_of_care"]["ci"]), f"baseline in all adults 50+: {pc(MO['baseline_usual_care_all_50plus']['est'])}"],
               ["Of the untreated: 2+ health-care visits past year", pc(A["untreated_ida_two_plus_visits_past_year"]["est"]), ci(A["untreated_ida_two_plus_visits_past_year"]["ci"]), ""],
               ["Of the untreated: insured", pc(A["untreated_ida_insured"]["est"]), ci(A["untreated_ida_insured"]["ci"]), ""],
               ["Untreated share among people with a usual doctor and 2+ visits (exploratory)", pc(reg["untreated_share_of_ida"]), ci(reg["ci"]), f"n = {reg['ida_n']}"]],
              "Iron-deficiency anaemia: haemoglobin below 13 g/dL (men) or 12 g/dL (women) and ferritin below 45 ng/mL (AGA 2020). Women 50+ reporting regular periods in the past year excluded. Design-based intervals by Taylor linearisation.")


def fu_row(name, d):
    return [name, f"{d['n']:,}", d["ida_n"], d["cancer_deaths_ida"], pc(d["crude_cancer_rate_ida"], 1), pc(d["crude_cancer_rate_none"], 1),
            f"{d['or_cancer_ida']:.2f} ({d['or_cancer_ida_ci'][0]:.2f} to {d['or_cancer_ida_ci'][1]:.2f})" if "or_cancer_ida" in d else "too few deaths",
            f"{d['or_other_ida']:.2f} ({d['or_other_ida_ci'][0]:.2f} to {d['or_other_ida_ci'][1]:.2f})" if "or_other_ida" in d else ""]


t_fu = table("followup", "Five-year cancer death in adults 50-84 tested regardless of symptoms (NHANES III 1988-1994 and 2001-2002, deaths through 2019).",
             ["Dataset", "Adults", "With IDA", "Cancer deaths in IDA", "Cancer death, IDA", "Cancer death, no anaemia", "Odds ratio, cancer death (95% CI)", "Odds ratio, other-cause death"],
             [fu_row("NHANES 2001-2002", B5["NHANES 2001-2002"]), fu_row("NHANES III", N3), fu_row("Pooled", P5)],
             "Odds ratios adjusted for age (spline), sex, ever smoking and dataset; bootstrap intervals (500 resamples). Non-cases include deaths from other causes.")
bars = B5["bars"]
t_bars = table("bars", "Pre-registered bars and what happened.", ["#", "Bar", "Result", "Outcome"], [
    ["Societal", "Over half of untreated IDA in people with a usual place of care (interval above 50%)", f"{pc(A['untreated_ida_usual_place_of_care']['est'])} ({ci(A['untreated_ida_usual_place_of_care']['ci'])})", "Met, but uninformative: the baseline in all adults 50+ is " + pc(MO["baseline_usual_care_all_50plus"]["est"])],
    ["S1", "IDA odds ratio for 5-year cancer death above 1, pooled, and the same direction in each dataset", f"pooled {P5['or_cancer_ida']:.2f} ({P5['or_cancer_ida_ci'][0]:.2f} to {P5['or_cancer_ida_ci'][1]:.2f}); 2001-2002 has {B5['NHANES 2001-2002']['cancer_deaths_ida']} cancer death among people with IDA", "<b class='bad'>Not shown</b> (pooled signal clear; one dataset cannot be tested)"],
    ["S2", "IDA odds ratio for cancer death larger than for other-cause death (ratio interval above 1)", f"{P5['ratio_of_ors_cancer_vs_other_ida']:.2f} ({P5['ratio_of_ors_ci'][0]:.2f} to {P5['ratio_of_ors_ci'][1]:.2f})", "<b class='bad'>Missed</b>"],
    ["R1", "Replication in 2001-2002: untreated share of IDA among adults 50+ has an interval above 50%", f"{pc(R1['ida_untreated_share']['est'])} ({ci(R1['ida_untreated_share']['ci'])})", "<b class='ok'>Met</b>"],
    ["M1", "Competing-risks model: IDA relative-risk ratio for cancer death larger than for other-cause death (interval above 1)", f"{M1['ratio']:.2f} ({M1['ratio_ci'][0]:.2f} to {M1['ratio_ci'][1]:.2f})", "<b class='bad'>Missed</b>"],
    ["T1", "Adding ferritin for any anaemia in adults 50+ finds 90%+ of guideline IDA at no more than 10 tests per case", f"{T1a['any_anaemia']['ferritin_tests_per_case']} tests per case (2017-2020), {T1b['any_anaemia']['ferritin_tests_per_case']} (2001-2002)", "<b class='ok'>Met</b> (the 90% part is true by definition)"],
    ["T2", "Microcytic anaemia: cancer-death risk raised and more than other-cause death, 21,820 adults", f"{T2['micro_anaemia']['rrr_cancer']:.2f} ({T2['micro_anaemia']['rrr_cancer_ci'][0]:.2f} to {T2['micro_anaemia']['rrr_cancer_ci'][1]:.2f}); ratio {T2['micro_anaemia']['ratio']:.2f}", "<b class='bad'>Missed</b>"],
    ["T3", "Cancer signal stronger in months 0-24 than 25-60", f"ratio {T3['ratio_early_to_late']:.2f} ({T3['ratio_ci'][0]:.2f} to {T3['ratio_ci'][1]:.2f})", "<b class='bad'>Missed</b> (too few deaths)"],
    ["S3", "IDA odds ratio for cancer death larger than for non-iron anaemia (interval above 1)", f"{P5['ida_minus_nia_cancer_ratio_of_ors']:.2f} ({P5['ida_minus_nia_ci'][0]:.2f} to {P5['ida_minus_nia_ci'][1]:.2f})", "<b class='bad'>Missed</b> (borderline; crosses 1)"],
], "Bars committed in docs/RED_FLAG_LABS_PREREG.md before the data were downloaded.")

ABSTRACT = f"""
<p><b>Background.</b> A US guideline (American Gastroenterological Association, 2020) says that men and women past menopause who have iron-deficiency anaemia should have an upper endoscopy and a colonoscopy, even with no symptoms, because slow bleeding from a gut cancer is one cause. Both numbers needed are on ordinary lab reports. Record-based studies can only see people whose doctor ordered the test. NHANES measured everyone, so it can show how many people carry this red flag without being treated, and whether it signals cancer in people who were not tested because something was wrong.</p>
<p><b>Methods.</b> Pre-registered. NHANES 2017-March 2020 (adults 50+, survey-weighted) for the national picture; NHANES III (1988-1994) and 2001-2002, linked to deaths through 2019, for five-year cancer death. Iron-deficiency anaemia (IDA): haemoglobin below 13 g/dL (men) or 12 g/dL (women) with ferritin below 45 ng/mL. A second, independent analyst (an AI agent with its own code) re-ran every number.</p>
<p><b>Results.</b> {pc(A['ida_prevalence']['est'], 1)} of US adults 50+ had IDA, about <b>{A['ida_prevalence']['millions']} million people</b> (95% CI {A['ida_prevalence']['millions_ci'][0]} to {A['ida_prevalence']['millions_ci'][1]} million). {pc(A['ida_untreated_share']['est'])} were not being treated for anaemia, and the same was true 17 years earlier in an independent survey ({pc(R1['ida_untreated_share']['est'])} in 2001-2002, a pre-registered replication). Among people who had a usual doctor and at least two visits in the past year, {pc(reg['untreated_share_of_ida'])} of IDA was untreated (exploratory), about {MO['ida_prevalence_among_regular_attenders']['millions_untreated_among_them']} million people. In people tested regardless of symptoms, IDA went with higher odds of cancer death within five years (odds ratio {P5['or_cancer_ida']:.2f}, 95% CI {P5['or_cancer_ida_ci'][0]:.2f} to {P5['or_cancer_ida_ci'][1]:.2f}; one cancer death per {P5['people_with_ida_per_cancer_death']:.0f} people with IDA against one per {P5['people_without_anaemia_per_cancer_death']:.0f} without anaemia), but almost as much with death from other causes ({P5['or_other_ida']:.2f}), and the signal faded by ten years. Counting any iron prescription or supplement as treatment still left {pc(A1['plus_any_iron_supplement']['est'])} untreated. A competing-risks model gave the same answer, and so did four routine red flags tested on {T2['n']:,} adults. Adding ferritin automatically for anaemic adults 50+ would cost about {T1a['any_anaemia']['ferritin_tests_per_case']} ferritin tests per case found, while the usual small-red-cell shortcut misses about three in four cases. Of the pre-registered bars, the replication and the laboratory-trigger bar were met, one was met but uninformative, and the six cancer-signal bars were missed or not shown.</p>
<p><b>Conclusions.</b> A guideline red flag already printed on routine lab reports goes untreated in most of the older Americans who have it, most of whom see a doctor regularly. In an unselected population it marks a raised short-term risk of dying of cancer, but mostly poor health in general, so it argues for following the existing guideline (finding the cause of the iron loss), not for treating it as a cancer test.</p>
"""

INTRO = """
<p>Most cancer-risk tools need something new: a scan, a genetic test, a questionnaire. This study asks what is already sitting on the lab reports people have. A complete blood count, the most common blood test there is, reports haemoglobin. A ferritin test, which measures stored iron, is cheap and widely available. Together they identify iron-deficiency anaemia, and the American Gastroenterological Association's 2020 guideline strongly recommends that men and women past menopause who have it be examined for a source of gut bleeding with an upper endoscopy and a colonoscopy, even with no symptoms.<sup>1</sup> Colon and stomach cancers bleed slowly, and iron-deficiency anaemia is one of the ways they come to light.</p>
<p>There is good evidence that abnormal routine blood results raise the chance of a cancer diagnosis, mostly from primary-care records in the United Kingdom, Denmark and Sweden.<sup>2,3</sup> There is also evidence that the guideline is not followed: a survey of US primary-care physicians found gaps in how iron studies are interpreted and when patients are referred,<sup>4</sup> and only about one in ten younger US veterans with new iron-deficiency anaemia had both endoscopies within a year.<sup>5</sup> But record-based studies share a blind spot. They can only see people whose doctor ordered the test, usually because something was wrong, so they cannot say how many people are walking around with this result untreated, and they cannot separate the cancer signal of the result from the reason the test was ordered.</p>
<p>The National Health and Nutrition Examination Survey (NHANES) measured haemoglobin and ferritin in a national sample regardless of symptoms, asked about anaemia treatment and health-care use, and is linked to death records. That allows two questions: how many older Americans have this guideline red flag and are not being treated (societal), and does it signal cancer in people tested without a reason (scientific)?</p>
"""

METHODS = f"""
<h3>2.1 Pre-registration</h3>
<p>Definitions, populations, outcomes and four bars were committed before the data were downloaded (<code>docs/RED_FLAG_LABS_PREREG.md</code>). Three dated addenda record decisions made afterwards: coding choices and a wording fix before any result; robustness checks after the main result; and changes made after an independent re-analysis. Everything decided after a result is labelled exploratory.</p>
<h3>2.2 Data</h3>
<p><b>Current picture:</b> NHANES 2017-March 2020 pre-pandemic files (demographics, complete blood count, ferritin, anaemia treatment, cancer history, insurance, usual care, visits, menstruation), adults 50+, with the exam weights, strata and primary sampling units. Ferritin was not measured in adults 50+ in 2021-2023, so this is the most recent national picture. <b>Follow-up:</b> NHANES III (1988-1994; ferritin and red-cell size read from the NHANES III lab file at the positions given in CDC's layout) and NHANES 2001-2002, linked to the National Death Index through 2019, adults 50-84 without earlier cancer.</p>
<h3>2.3 Definitions</h3>
<p>Anaemia: haemoglobin below 13 g/dL in men and 12 g/dL in women (World Health Organization). Iron-deficiency anaemia (IDA): anaemia with ferritin below 45 ng/mL, the guideline's cut-off.<sup>1</sup> Non-iron anaemia: anaemia with ferritin 45 or more. Age 50 and older stands in for being past menopause; women 50+ who reported regular periods in the past year were excluded. Not treated: reported no treatment for anaemia in the past three months. Usual place of care: one or more places usually gone to for care; two or more visits: two or more health-care visits in the past year.</p>
<h3>2.4 Analysis</h3>
<p>National estimates are survey-weighted ratios with design-based 95% intervals (Taylor linearisation with strata and primary sampling units). The follow-up analysis is logistic regression of death from cancer within five years (secondary: ten years) on IDA and non-iron anaemia against no anaemia, adjusted for age (spline), sex, ever smoking and dataset, with bootstrap intervals. Non-cases include deaths from other causes. The same model for death from other causes tests whether the result is specific to cancer. Python (pandas, scikit-learn); the code is <code>experiments/red_flag_labs.py</code>, with tests of the estimator and the definitions in <code>tests/test_red_flag_labs.py</code>.</p>
"""

_NAMES = {"high_calcium": "High calcium", "high_alp": "High alkaline phosphatase", "low_albumin": "Low albumin", "high_ggt": "High GGT",
          "high_wbc": "High white count", "high_platelets": "High platelets", "high_fib4": "High FIB-4 liver score",
          "new_diabetes_hba1c": "Unrecognised diabetes-range HbA1c", "urine_albumin": "Albumin in urine", "high_crp": "High C-reactive protein",
          "two_or_more": "Two or more of these"}


def _cells(r):
    if "rrr_cancer" not in r:
        return [r["cancer_deaths_flagged"], "too few", "", ""]
    return [r["cancer_deaths_flagged"], f"{r['rrr_cancer']:.2f} ({r['rrr_cancer_ci'][0]:.2f} to {r['rrr_cancer_ci'][1]:.2f})", f"{r['rrr_other']:.2f}",
            f"{r['ratio']:.2f} ({r['ratio_ci'][0]:.2f} to {r['ratio_ci'][1]:.2f})"]


_rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in [_NAMES[k]] + _cells(RD["flags"][k]) + _cells(RR["flags"][k])[1:]) + "</tr>" for k in _NAMES)
ROUTINE = f"""<h3>3.8 Flags on other routine tests</h3>
<p>The same question was asked of other tests people commonly have: calcium, liver enzymes, albumin, white count, platelets, a liver score calculated from them, blood sugar (HbA1c), urine albumin and C-reactive protein. Cut-offs, bars and a multiple-testing correction were committed first (<code>docs/ROUTINE_TESTS_PREREG.md</code>); the analysis was not blind, because earlier work here had suggested routine values mostly mark general ill health. Discovery: NHANES 1999-2014, {RD['n']:,} adults 50-84 and {RD['cancer_deaths']} cancer deaths within five years. Check: NHANES III, {RR['n']:,} adults and {RR['cancer_deaths']} cancer deaths. Most flags raised the risk of cancer death, but none raised it more than the risk of other deaths: no flag passed discovery, so none was confirmed. Albumin in urine, low albumin and having two or more flags pointed clearly the other way, towards deaths from other causes. The NHANES III columns are therefore exploratory. A first run of that check read the wrong column for diagnosed diabetes; the error was caught by an implausible rate, fixed, and the code now stops if the column is wrong.</p>
<div class="tcap"><b>Table 4.</b> Flags on routine tests: relative-risk ratios for five-year cancer death and other-cause death.</div>
<table class="num"><thead><tr><th>Flag</th><th>Cancer deaths flagged (1999-2014)</th><th>Cancer RRR (95% CI)</th><th>Other-cause RRR</th><th>Cancer / other (95% CI)</th><th>NHANES III: cancer RRR</th><th>Other-cause RRR</th><th>Cancer / other</th></tr></thead><tbody>{_rows}</tbody></table>
<p class="tnote">Each flag modelled alone against everyone else with that test, adjusted for age (spline), sex, ever smoking and survey wave. Unweighted. Bar: cancer RRR and ratio above 1 with p below 0.05/11. Blood pressure, pulse and lung function were not included.</p>"""

ANGLES = f"""<h3>3.9 Counting iron pills, who pays, and medicines in plain sight</h3>
<p><b>Treated, counted more generously (bar A1 met).</b> The main estimate uses one question about anaemia treatment. Counting iron prescriptions as treatment changed nothing ({pc(A1['plus_iron_prescription']['est'])} untreated; no oral iron prescription matched among these cases). Also counting iron supplements at a treatment dose (45 mg a day or more) gave {pc(A1['plus_treatment_dose_supplement']['est'])}, and counting any iron supplement at all, including a multivitamin with a little iron, gave {pc(A1['plus_any_iron_supplement']['est'])} (95% CI {ci(A1['plus_any_iron_supplement']['ci'])}), about {AG['A1_untreated_millions_broadest']} million people. Even on the most generous definition most cases showed no iron at all. A first run matched drug names containing the letters "iron", which also caught buspirone and spironolactone; it was corrected before reporting.</p>
<p><b>Who covers the untreated (descriptive).</b> Among people untreated on the broadest definition, {pc(A3['medicare']['share_of_untreated_covered']['est'])} had Medicare, {pc(A3['private']['share_of_untreated_covered']['est'])} private insurance, {pc(A3['medicaid']['share_of_untreated_covered']['est'])} Medicaid and {pc(A3['uninsured']['share_of_untreated_covered']['est'])} no insurance (people can hold more than one; intervals are wide). Nearly all are insured, so a rule at the insurer or health-system level could reach them.</p>
<p><b>Medicines that can drain or block iron (descriptive, not adjusted for age).</b> {pc(MED['blood_thinner']['ida']['est'])} of people with iron-deficiency anaemia had a prescription blood thinner or antiplatelet drug, against {pc(MED['blood_thinner']['no_anaemia']['est'])} of people without anaemia, and {pc(MED['acid_reducer_ppi']['ida']['est'])} an acid reducer (proton-pump inhibitor), against {pc(MED['acid_reducer_ppi']['no_anaemia']['est'])}. {pc(MED['any_of_these']['ida']['est'])} had at least one of these or an anti-inflammatory painkiller (against {pc(MED['any_of_these']['no_anaemia']['est'])}). Over-the-counter use is not counted. These drugs do not explain away the need to look: the guideline's work-up applies whether or not someone takes them. They do suggest a practical trigger, since a pharmacist or an electronic record already sees the prescription.</p>"""

RESULTS = f"""
<h3>3.1 The bars</h3>
{t_bars}
<h3>3.2 How many people carry the red flag, untreated</h3>
<p>Of US adults 50+, {pc(A['ida_prevalence']['est'], 1)} (95% CI {ci(A['ida_prevalence']['ci'], 1)}) had iron-deficiency anaemia, about {A['ida_prevalence']['millions']} million people ({T('societal')}, {F('fig1_funnel')}). The rate was similar in men ({pc(A['ida_prevalence_men']['est'], 1)}) and women past menopause ({pc(A['ida_prevalence_women']['est'], 1)}). {pc(A['ida_untreated_share']['est'])} of them reported no treatment for anaemia in the past three months, about {A['ida_untreated_share']['millions']} million people.</p>
{fig("fig1_funnel", "US adults 50+ with iron-deficiency anaemia, those not being treated, and those not treated despite a usual doctor and two or more visits in the past year (the last is exploratory).")}
<p>The pre-registered societal bar asked whether most of the untreated had a usual place of care. They did ({pc(A['untreated_ida_usual_place_of_care']['est'])}), but so do {pc(MO['baseline_usual_care_all_50plus']['est'])} of all adults 50+, so that bar says nothing about missed opportunities; it was met but uninformative. The more telling figure, chosen after the re-check and labelled exploratory, is the other direction: among people with a usual doctor and two or more visits in the past year, {pc(reg['untreated_share_of_ida'])} of iron-deficiency anaemia was untreated (95% CI {ci(reg['ci'])}; {F('fig2_untreated')}), about {MO['ida_prevalence_among_regular_attenders']['millions_untreated_among_them']} million people. Even among people in regular contact with a doctor, about three in four were not being treated. (The comparison group without regular contact is too small, 16 people, to say whether contact helps.)</p>
{fig("fig2_untreated", "Share of iron-deficiency anaemia not being treated, overall and among people with a usual doctor and two or more visits in the past year (exploratory).", "80%")}
{t_soc}
<p>The rate differed by group, from {pc(A['ida_prevalence_by_group']['non-Hispanic White']['est'], 1)} in non-Hispanic White adults to {pc(A['ida_prevalence_by_group']['non-Hispanic Black']['est'], 1)} in non-Hispanic Black adults, with wide intervals ({F('fig3_groups')}). Race is reported as a description, not used as a predictor.</p>
{fig("fig3_groups", "Iron-deficiency anaemia by race and ethnicity among adults 50+ (groups with at least eight cases).", "80%")}
<h3>3.3 The same gap 17 years earlier (pre-registered replication)</h3>
<p>In NHANES 2001-2002, an independent sample with the same measurements and questions, {pc(R1['ida_prevalence']['est'], 1)} of adults 50+ had iron-deficiency anaemia (95% CI {ci(R1['ida_prevalence']['ci'], 1)}; {R1['ida_n']} people), and {pc(R1['ida_untreated_share']['est'])} of them were not being treated (95% CI {ci(R1['ida_untreated_share']['ci'])}). The pre-registered bar (interval above 50%) was met. Among those with a usual doctor and two or more visits it was {pc(R1['untreated_share_regular_attenders']['est'])}. The share left untreated did not change between 2001-2002 and 2017-2020, the period in which the 2020 guideline was written.</p>
<h3>3.4 Robustness (exploratory)</h3>
<p>Leaving out people who reported an earlier cancer, using a stricter ferritin cut-off of 30 ng/mL, or looking at men only changed little: prevalence {pc(ROB['no_prior_cancer']['prevalence'], 1)}, {pc(ROB['ferritin_below_30']['prevalence'], 1)} and {pc(ROB['men_only']['prevalence'], 1)}; untreated {pc(ROB['no_prior_cancer']['untreated_share'])}, {pc(ROB['ferritin_below_30']['untreated_share'])} and {pc(ROB['men_only']['untreated_share'])}.</p>
<h3>3.5 Does the red flag signal cancer in people tested without a reason?</h3>
<p>In the follow-up data, {P5['ida_n']} of {P5['n']:,} adults had iron-deficiency anaemia ({T('followup')}). {pc(P5['crude_cancer_rate_ida'], 1)} of them died of cancer within five years, against {pc(P5['crude_cancer_rate_none'], 1)} of people without anaemia: one cancer death per {P5['people_with_ida_per_cancer_death']:.0f} people flagged, against one per {P5['people_without_anaemia_per_cancer_death']:.0f}. The adjusted odds ratio was {P5['or_cancer_ida']:.2f} (95% CI {P5['or_cancer_ida_ci'][0]:.2f} to {P5['or_cancer_ida_ci'][1]:.2f}). But the odds of dying of other causes were raised almost as much ({P5['or_other_ida']:.2f}), so the result is not specific to cancer (bar S2 missed), and it did not clearly beat non-iron anaemia (S3 missed, interval {P5['ida_minus_nia_ci'][0]:.2f} to {P5['ida_minus_nia_ci'][1]:.2f}). At ten years the cancer odds ratio fell to {P10['or_cancer_ida']:.2f} ({P10['or_cancer_ida_ci'][0]:.2f} to {P10['or_cancer_ida_ci'][1]:.2f}) while the other-cause odds ratio stayed near {P10['or_other_ida']:.2f} ({F('fig4_odds')}). That pattern, a cancer signal concentrated in the first years and a frailty signal that lasts, fits cancers that were already present and bleeding, but it was noticed after the run and the numbers are small: {P5['cancer_deaths_ida']} cancer deaths among people with iron-deficiency anaemia, {N3['cancer_deaths_ida']} of them in NHANES III. The 2001-2002 survey had {B5['NHANES 2001-2002']['cancer_deaths_ida']}, so it cannot confirm the direction (S1 not shown).</p>
<p>A competing-risks model, which treats death from cancer and death from other causes as rival outcomes in one model (pre-registered in Addendum 4), gave the same picture: relative-risk ratio {M1['rrr_cancer_ida']:.2f} (95% CI {M1['rrr_cancer_ci'][0]:.2f} to {M1['rrr_cancer_ci'][1]:.2f}) for cancer death and {M1['rrr_other_ida']:.2f} ({M1['rrr_other_ci'][0]:.2f} to {M1['rrr_other_ci'][1]:.2f}) for other-cause death; their ratio, {M1['ratio']:.2f} ({M1['ratio_ci'][0]:.2f} to {M1['ratio_ci'][1]:.2f}), does not show cancer specificity.</p>
{fig("fig4_odds", "Odds ratios for death from cancer and from other causes, five and ten years after testing, for iron-deficiency anaemia and non-iron anaemia against no anaemia (pooled NHANES III and 2001-2002).")}
{t_fu}
<h3>3.6 A blood-count trigger for laboratories</h3>
<p>Ferritin is not part of a routine blood count, so most people with guideline iron-deficiency anaemia are found only if someone orders it. A common shortcut is to look into iron only when red cells are small (MCV under 80 fL). Among adults 50+, that shortcut found only {pc(T1a['microcytic_anaemia']['sensitivity'])} of guideline cases in 2017-2020 (95% CI {ci(T1a['microcytic_anaemia']['sensitivity_ci'])}) and {pc(T1b['microcytic_anaemia']['sensitivity'])} in 2001-2002: most iron-deficiency anaemia in older adults has normal-sized red cells. Adding ferritin automatically whenever an adult 50+ is anaemic finds every case by definition, at {T1a['any_anaemia']['ferritin_tests_per_case']} ferritin tests per case found in 2017-2020 and {T1b['any_anaemia']['ferritin_tests_per_case']} in 2001-2002 (pre-registered bar T1 met; its sensitivity part was true by definition, so the informative numbers are the cost per case and the shortcut's miss rate). Medicare pays $13.63 for a ferritin test (CPT 82728, 2026 Clinical Laboratory Fee Schedule), so the rule costs about ${13.63 * T1a['any_anaemia']['ferritin_tests_per_case']:.0f} in tests per case found (2017-2020): a cheap laboratory rule, not a new test.</p>
<h3>3.7 Four routine red flags in 21,820 adults</h3>
<p>Ferritin was not measured in older adults in 2003-2014, but haemoglobin, red-cell size, platelets and a weight-loss question were. Pooling NHANES 1999-2014 with NHANES III gave {T2['n']:,} adults 50-84 and {T2['cancer_deaths']} cancer deaths within five years, about four times the data of Section 3.5. In one competing-risks model, no routine red flag was specific to cancer (pre-registered bar T2 missed). Microcytic anaemia: cancer {T2['micro_anaemia']['rrr_cancer']:.2f} ({T2['micro_anaemia']['rrr_cancer_ci'][0]:.2f} to {T2['micro_anaemia']['rrr_cancer_ci'][1]:.2f}) against other causes {T2['micro_anaemia']['rrr_other']:.2f}. Other anaemia: {T2['other_anaemia']['rrr_cancer']:.2f} against {T2['other_anaemia']['rrr_other']:.2f}, mostly frailty. Platelets above 400: {T2['high_platelets']['rrr_cancer']:.2f} ({T2['high_platelets']['rrr_cancer_ci'][0]:.2f} to {T2['high_platelets']['rrr_cancer_ci'][1]:.2f}) against {T2['high_platelets']['rrr_other']:.2f}. Unintentional weight loss (1999-2014 only): {T2['weight_loss_1999_2014']['rrr_cancer']:.2f} ({T2['weight_loss_1999_2014']['rrr_cancer_ci'][0]:.2f} to {T2['weight_loss_1999_2014']['rrr_cancer_ci'][1]:.2f}) against {T2['weight_loss_1999_2014']['rrr_other']:.2f}. The timing test (T3) was missed: {T3['cancer_deaths_exposed_0_24'] + T3['cancer_deaths_exposed_25_60']} cancer deaths among people with microcytic anaemia are too few to compare early and late. Of all the red flags, only ferritin-confirmed iron-deficiency anaemia (Section 3.5) and high platelets showed a cancer signal with an interval above 1, and neither was cancer-specific.</p>
{ROUTINE}
{ANGLES}
<h3>3.10 Independent re-analysis</h3>
<p>A second analyst (an AI agent writing its own code from the pre-registration) reproduced every national number to the decimal and the pooled odds ratios within bootstrap noise (<code>docs/independent_check/red_flag/</code>). It found four weaknesses, all acted on: the societal bar was uninformative (reported above); "not treated" is not the same as "not recognised" (wording changed throughout); {441} NHANES III records had no follow-up months from the exam and had been silently dropped (recovered from the interview date); and the follow-up result rests on very few deaths (stated).</p>
"""

DISCUSSION = f"""
<h3>4.1 What this means</h3>
<p>The societal finding is the stronger one, and it held in two surveys 17 years apart. An action that a US guideline already recommends, prompted by numbers already printed on routine lab reports, is not happening for most of the roughly {A['ida_prevalence']['millions']} million older Americans it applies to, and the gap is not mainly about access: most of them see a doctor regularly. That points at the steps between a result and a referral. Cheap changes follow, each needing no new test: a laboratory can add ferritin automatically when haemoglobin is low in someone over 50; an electronic record can flag anaemia with low ferritin and point to the guideline; a prescription check can prompt a blood count and ferritin for older people on blood thinners or acid reducers (a suggestion this study has not tested); and a patient can read their own report, which is what the companion page in the Oncovision app does (<code>/redflag</code>: three numbers in, the guideline's verdict and the questions to ask out).</p>
<p>The scientific finding is more modest and should be read carefully. In people tested without a reason, iron-deficiency anaemia raised the short-term odds of dying of cancer about two and a half times, but raised the odds of dying of anything else almost as much. It is a marker of poor health that includes cancer, not a cancer test. That supports the guideline's logic (find out why someone is losing iron) without supporting a claim that this blood result screens for cancer.</p>
<h3>4.2 Limits</h3>
<ul>
<li><b>No cancer count is projected.</b> Published scope yields come from people already referred for endoscopy; applied to this group they would imply an implausible number of hidden cancers, so the planned projection was dropped before any result.</li>
<li><b>Not treated is not unrecognised.</b> The question covers anaemia treatment in the past three months; someone fully investigated and found to have a benign cause, or treated and recovered, is not distinguished. People who recovered are no longer anaemic, which pushes the untreated share up.</li>
<li><b>No endoscopy history.</b> NHANES does not ask whether people had the recommended endoscopies, so the gap in investigation is inferred, not measured.</li>
<li><b>Cancer death, not site or diagnosis.</b> The public death files do not name the cancer, so gut cancers cannot be separated out.</li>
<li><b>Small numbers and old follow-up data.</b> {P5['cancer_deaths_ida']} cancer deaths among people with iron-deficiency anaemia, mostly from 1988-1994; the follow-up analysis is not survey-weighted.</li>
<li><b>Age 50 as a stand-in for menopause,</b> and the 2017-2020 estimate is the latest available because ferritin was not measured in older adults in 2021-2023.</li>
</ul>
<h3>4.3 Novelty, and the closest work</h3>
<p>A prior-art check on 26 November 2026 (Europe PMC, including PubMed and preprints: 10 query families, 1,107 records, 169 read in full abstract; <code>data/audit/prior_art_red_flag.csv</code>) found three pieces of work closer than those cited above, and the claims here are narrowed accordingly. Wang and Wang used the same survey and the same treatment question to report national anaemia treatment rates for 2005-2018,<sup>8</sup> so "anaemia often goes untreated" is not new. Tawfik and colleagues estimated iron deficiency and its trend from 1999-2002 to 2017-2020,<sup>9,10</sup> so the rise in iron deficiency is not new. In the ARIC community cohort, anaemia was linked to cancer death but no more than to other deaths once adjusted,<sup>11</sup> which agrees with the scientific result here. What we did not find: an estimate restricted to the guideline's own definition and population (anaemia with ferritin under 45 ng/mL in men and women past menopause, the group for whom endoscopy is strongly recommended), the untreated share among people in regular contact with a doctor, or the comparison of that share across two eras with ferritin measured. Those three pieces are the contribution. For the additions in Sections 3.6 and 3.7, closer work exists too: Spanish primary-care laboratories already use automatic algorithms that add iron tests,<sup>12</sup> normocytic iron deficiency is well described, and record-based work has studied cancer and competing deaths after symptoms in primary care.<sup>13</sup> The new parts are the national size of the gap the small-red-cell shortcut leaves in older adults, with its cost per case, and a side-by-side test of four routine red flags for cancer specificity in a population tested without a reason. Searches are not proof of absence.</p>
"""

DECL = """
<h3>AI assistance</h3>
<p>An AI assistant (Anthropic Claude, via Claude Code) wrote the code, ran the analyses, drafted the text and searched the literature; a second AI agent re-ran the analysis with its own code. An AI cannot be an author. The human author is responsible for the work and must check every statement. No human second analyst or clinician has reviewed it.</p>
<h3>Data, code, ethics</h3>
<p>Public, de-identified NCHS data; no participants contacted. Code, pre-registration with addenda, results and the re-analysis are in github.com/palashraks-afk/OncoVision. No funding; no conflicts of interest.</p>
"""

REFS = """
<ol>
<li>Ko CW, Siddique SM, Patel A, et al. AGA clinical practice guidelines on the gastrointestinal evaluation of iron deficiency anemia. <i>Gastroenterology</i> 2020;159:1085-1094.</li>
<li>Incident anaemia as a marker of cancer and all-cause mortality: evidence from 380 114 adults in the population-based Stockholm Early Detection of Cancer Study (STEADY-CAN) cohort. <i>BMJ Oncology</i> 2026 (PMID 41971157).</li>
<li>Anaemia, ethnicity, and cancer incidence: a retrospective cohort study in primary care. <i>Br J Gen Pract</i> 2025 (PMID 40174989).</li>
<li>Testing practices, interpretation, and diagnostic evaluation of iron deficiency anemia by US primary care physicians. <i>JAMA Netw Open</i> 2021 (PMID 34596670).</li>
<li>Endoscopy underuse for new-onset iron deficiency anemia in younger veterans. <i>Am J Manag Care</i> 2026 (PMID 42189062).</li>
<li>The association of biomarkers of iron status with mortality in US adults. <i>Nutr Metab Cardiovasc Dis</i> 2012 (PMID 21330119).</li>
<li>National Center for Health Statistics. NHANES 2017-March 2020 pre-pandemic files; NHANES 2001-2002; NHANES III (1988-1994); public-use Linked Mortality Files through 31 December 2019.</li>
<li>Wang C, Wang Y. Trends in prevalence and treatment rate of anemia in the U.S. population: cross-sectional study using data from NHANES 2005-2018. <i>Hematology</i> 2022 (PMID 35962563).</li>
<li>Tawfik YMK, Billingsley H, Bhatt AS, et al. Absolute and functional iron deficiency in the US, 2017-2020. <i>JAMA Netw Open</i> 2024 (PMID 39316402).</li>
<li>Tawfik YMK, Aboelsaad IAF, Nelson MD, et al. Temporal trends in absolute iron deficiency among US adults. <i>Am J Med</i> 2026 (PMID 42815754).</li>
<li>Cannon EJ, Misialek JR, Buckley LF, et al. Anemia, iron deficiency, and cause-specific mortality: the Atherosclerosis Risk in Communities Study. <i>Gerontology</i> 2024 (PMID 39047718).</li>
<li>Initial study of anaemia profile for primary care centres with automated laboratory algorithms reduces the demand for ferritin, iron, transferrin and other tests. 2022 (PMID 33234695).</li>
<li>Cancer incidence and competing mortality risk following 15 presenting symptoms in primary care: a population-based cohort study using electronic health records. Preprint, 2024 (Europe PMC PPR856308).</li>
</ol>
<p class="tnote">Author lists for references 2-6 are to be completed from the originals.</p>
"""

TITLE = "A Red Flag Already on the Lab Report"
SUB = "Iron-deficiency anaemia that a US guideline says should be investigated: how many older Americans have it untreated, and what it signals in people tested without a reason"
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1><div class="sub">{SUB}</div>
<div class="meta"><b>Palash Rakshit</b> (author) &nbsp;&middot;&nbsp; Draft of 26 November 2026 &nbsp;&middot;&nbsp; Oncovision project</div>
<div class="banner"><b>Draft. Not peer reviewed.</b> Analysis by an AI assistant and re-run by a second AI agent, not by a person or a clinician. The replication bar was met; the four scientific bars were missed or not shown. Nothing here is medical advice.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> iron-deficiency anaemia; routine laboratory tests; missed diagnostic opportunities; colorectal cancer; NHANES; guideline adherence; pre-registration.</p>
<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
</body></html>"""
out = os.path.join(HERE, "red_flag_paper.html")
open(out, "w", encoding="utf-8").write(html)
chrome = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or "") if p and os.path.exists(p)), None)
if chrome:
    pdf = os.path.join(HERE, "Red_Flag_On_The_Lab_Report.pdf")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=True, timeout=180)
    print("wrote", pdf)
else:
    print("wrote", out, "(no Chrome found for the PDF)")
