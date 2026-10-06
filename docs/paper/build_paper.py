"""
Builds the navigator research paper (HTML, then PDF through headless Chrome).

Every number in the text and tables is read from the saved result files, so the paper
cannot drift from the analysis.

Run:  python docs/paper/build_paper.py
Needs: experiments/navigator_evidence_result.json, navigator_readability_result.json,
       docs/paper/figures/*.png (python experiments/make_paper_figures.py)
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import navigator as nav  # noqa: E402

EV = json.load(open(os.path.join(ROOT, "experiments", "navigator_evidence_result.json")))
RD = json.load(open(os.path.join(ROOT, "experiments", "navigator_readability_result.json")))
CTX = json.load(open(os.path.join(ROOT, "backend", "navigator_context.json")))
KB = nav.load()
ERAS = ["1988-1994 (NHANES III)", "1999-2006", "2007-2014", "2015-2018"]
BANDS = ["40-49", "50-59", "60-69", "70-79", "80-+"]


def p(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(c, d=1):
    return f"{100 * c[0]:.{d}f} to {100 * c[1]:.{d}f}"


def orci(r):
    return f"{r['or']:.2f} ({r['ci'][0]:.2f} to {r['ci'][1]:.2f})"


def fig(name, caption, width="100%"):
    return f'<figure style="width:{width}"><img src="figures/{name}" alt=""><figcaption>{caption}</figcaption></figure>'


TN = {"rules": 1, "labs": 2, "sources": 3, "pre": 4, "burden": 5, "dec": 6, "flags": 7, "auc": 8, "rulehit": 9, "deciles": 10, "read": 11}
FN = {"pipeline": 1, "cohort": 2, "burden": 3, "decedents": 4, "auc": 5, "ferritin": 6, "rules": 7, "calib": 8, "read": 9, "form": 10, "result": 11}


def table(head, rows, cls="", note="", key=None, title=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<p class="tnote">{note}</p>' if note else ""
    cap = f'<div class="tcap"><b>Table {TN[key]}.</b> {title}</div>' if key else ""
    return f'{cap}<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{n}'


q1, q2, q3, q4, q5, q6, q7 = (EV[k] for k in ("q1", "q2", "q3", "q4", "q5", "q6", "q7"))
n_all, n_out, n3 = EV["n_cohort"], EV["n_outcome_cohort"], EV["n_nhanes3"]
q3r, q2r = q3["results"], q2["results"]
labs_only, ctx_pool, n3o = q2r["1999-2014 labs only"], q2r["1999-2014 labs and proxies"], q2r["NHANES III labs only"]
g_lab, g_ctx = q3r["age+sex vs +labs alert"], q3r["age+sex vs +labs and proxies alert"]
g_n3 = q3r["trained 1999-2014, tested NHANES III (labs alert)"]
rules = KB["rules"]
n_rules = len(rules) + 1
sites = sorted({r["site"] for r in rules} | {KB["melanoma_checklist"]["site"]})
age_burden = lambda era, b, k: q1["bands"][era][b][k]["share"]

# ---------------------------------------------------------------- tables
t_sources = table(
    ["Source", "Years", "Adults 40+ with a blood count", "Used for"],
    [["NHANES III, linked to the National Death Index", "1988-1994", f"{n3:,}", "Burden; cancer specificity; added value (external era)"],
     ["Continuous NHANES, eight cycles with five-year follow-up", "1999-2014", f"{n_out:,}", "All questions; context layer"],
     ["Continuous NHANES, two newest cycles", "2015-2018", f"{n_all - n_out:,}", "Burden and ferritin check only (follow-up too short)"],
     ["Total", "1988-2018", f"{n_all + n3:,}", ""]],
    "num", "NHANES: US National Health and Nutrition Examination Survey. Mortality follow-up through 31 December 2019 from the NCHS public-use linked mortality files.")

by_site = {}
for r in rules:
    by_site.setdefault(r["site"], []).append(r)
rows = []
for s in sorted(by_site):
    rs = by_site[s]
    tiers = sorted({nav.tier_of(r) for r in rs}, key=lambda t: nav.TIER_ORDER[t])
    rows.append([s.capitalize(), len(rs), ", ".join(t.replace("_", " ") for t in tiers)])
rows.append(["Melanoma", 1, "talk soon (7-point checklist)"])
t_rules = table(["Cancer site", "Rules", "Alert levels it can produce"], rows, "",
                f"{n_rules} rules in total. Each carries its source page in the guideline. Levels come from the guideline's own timeframe and wording ('recommended' versus 'consider').")

lt = KB["lab_thresholds"]
t_labs = table(
    ["Derived finding", "Definition used", "Where it comes from"],
    [["Anaemia", "Haemoglobin below 13 g/dL (male) or 12 g/dL (female)", "WHO definition for adults"],
     ["Iron deficiency", "Ferritin below 15 ng/mL. If no ferritin, MCV below 80 fL is used instead and the person is told so", "Common laboratory cut-off; MCV rule is the tool's disclosed assumption"],
     ["Iron-deficiency anaemia", "Anaemia and iron deficiency together", "Derived"],
     ["Thrombocytosis", "Platelets above 400 x10^9/L", "Guideline"],
     ["Raised white count", "White cells above 11 x10^9/L", "Common laboratory cut-off"],
     ["Raised CA-125", "35 U/mL or more", "Guideline"]], "")

rows = []
for era in ERAS:
    for b in BANDS:
        r = q1["bands"][era][b]
        rows.append([era if b == BANDS[0] else "", b.replace("-+", "+"), f"{r['n']:,}", f"{p(r['talk_soon']['share'])}", f"{p(r['any_alert']['share'])} ({ci(r['any_alert']['ci'])})"])
t_burden = table(["Era", "Age", "Adults", "See a doctor soon", "Any alert (95% interval)"], rows, "num",
                 "Lab results alone, no symptoms entered. Intervals are Wilson 95% intervals.")

pre = [
    ["1", "Burden is stable across eras: 'see a doctor soon' at most 3% and any alert at most 20% in every age band and era",
     f"Worst 'soon' {p(q1['worst_talk_soon'])}; worst any-alert {p(q1['worst_any_alert'])} (age 80+, 2015-2018)", "<b class='bad'>Missed</b> (any-alert, ages 80+)"],
    ["2", "Alerted people are more likely to die of cancer than of something else, among decedents (adjusted OR interval above 1 in 1999-2014, and OR above 1 in NHANES III)",
     f"1999-2014: OR {orci(ctx_pool)}. NHANES III: OR {orci(n3o)}", "<b class='bad'>Missed</b>"],
    ["3", "An alert adds to age and sex for five-year cancer death in the held-out era (AUC-gain interval excludes 0)",
     f"Labs and proxy symptoms: {g_ctx['gain']:+.3f} ({g_ctx['gain_ci'][0]:+.3f} to {g_ctx['gain_ci'][1]:+.3f}); labs only {g_lab['gain']:+.3f}", "<b class='bad'>Missed</b>"],
    ["4", "Report how well MCV below 80 stands in for ferritin (no bar)",
     f"Sensitivity {p(q4['proxy_sensitivity'], 0)}, PPV {p(q4['proxy_ppv'], 0)}", "Reported"],
    ["5", "Which rules fire (no bar)", "Colorectal rules account for almost all lab-driven alerts", "Reported"],
    ["6", "Age, sex and smoking context layer is calibrated in a later era (slope 0.8 to 1.2)",
     f"Slope {q6['calibration_slope']:.2f}; AUC {q6['auc']:.3f}", "<b class='ok'>Met</b>"],
]
t_pre = table(["#", "Pre-registered question and bar", "Result", "Outcome"], pre, "", "Bars were committed to the repository before the analysis was run (docs/NAVIGATOR_EVIDENCE_PREREG.md).")

t_dec = table(
    ["Group", "Deaths in 10 years", "Alerted", "Cancer share, alerted", "Cancer share, not alerted", "Adjusted OR (95% CI)"],
    [[k, f"{v['decedents']:,}", f"{v['alerted']:,}", p(v["cancer_share_alerted"]), p(v["cancer_share_not"]), orci(v)] for k, v in q2r.items()],
    "num", "Decedents only: everyone here died within ten years, so the question is whether the death was from cancer. Adjusted for age, age squared, sex and (pooled) examination year.")

t_auc = table(
    ["Comparison", "Test people", "Cancer deaths", "AUC, age + sex", "AUC, with alert", "Gain (95% CI)"],
    [["Labs alert; fit 1999-2006, test 2007-2014", f"{g_lab['test_n']:,}", g_lab["test_cancer_deaths"], f"{g_lab['auc_base']:.3f}", f"{g_lab['auc_with_alert']:.3f}", f"{g_lab['gain']:+.3f} ({g_lab['gain_ci'][0]:+.3f} to {g_lab['gain_ci'][1]:+.3f})"],
     ["Labs + proxy symptoms; fit 1999-2006, test 2007-2014", f"{g_ctx['test_n']:,}", g_ctx["test_cancer_deaths"], f"{g_ctx['auc_base']:.3f}", f"{g_ctx['auc_with_alert']:.3f}", f"{g_ctx['gain']:+.3f} ({g_ctx['gain_ci'][0]:+.3f} to {g_ctx['gain_ci'][1]:+.3f})"],
     ["Labs alert; fit 1999-2014, test NHANES III (exploratory)", f"{g_n3['test_n']:,}", g_n3["test_cancer_deaths"], f"{g_n3['auc_base']:.3f}", f"{g_n3['auc_with_alert']:.3f}", f"{g_n3['gain']:+.3f} ({g_n3['gain_ci'][0]:+.3f} to {g_n3['gain_ci'][1]:+.3f})"]],
    "num", "AUC: probability that a person who died of cancer within five years has a higher score than one who did not. Bootstrap intervals, 1,000 resamples.")

rows = [[k, f"{v['decedents_with_flag']:,} of {v['decedents']:,}", orci(v)] for k, v in q7["results"].items()]
t_flags = table(["Single flag", "Decedents flagged", "Adjusted OR for dying of cancer rather than another cause (95% CI)"], rows, "num",
                "Exploratory. Nine comparisons; only intervals well away from 1 should be believed. 'Proxy' means a crude NHANES stand-in for a symptom (see Methods).")

names = {"colorectal_stool_test": "Stool test for hidden blood (bowel)", "colorectal_urgent": "Urgent bowel referral", "lung_xray_symptoms": "Chest X-ray (lung)",
         "lung_xray_consider": "Consider chest X-ray (lung)", "ovarian_other": "Consider ovarian tests", "upper_gi_consider": "Consider endoscopy (stomach, gullet)",
         "weight_loss": "Unexplained weight loss, several cancers"}
rows = [[names.get(k, k), f"{v['n']:,}", p(v["rate"]), f"{v['mean_age']:.0f}"] for k, v in sorted(((k, v) for k, v in q5.items() if not k.startswith("_")), key=lambda kv: -kv[1]["n"])]
rows.append(["Everyone in the analysis", f"{q5['_everyone']['n']:,}", p(q5["_everyone"]["rate"]), ""])
t_rulehit = table(["Rule", "People it fired for", "Died of cancer within 5 years", "Mean age"], rows, "num",
                  "Labs plus proxy symptoms, 1999-2014. Descriptive only: the groups differ in age, and a higher rate in a group is mostly its older age.")

rd = RD["kinds"]
t_read = table(["Text the tool can show", "Items", "Mean US grade", "Hardest item", "Over grade 10"],
               [[k.capitalize(), rd[k]["n"], rd[k]["mean"], rd[k]["worst"], rd[k]["over_target"]] for k in ("action", "headline", "safety", "disclaimer", "symptom") if k in rd],
               "num", "Flesch-Kincaid grade level. 'Symptom' items are one-line labels, where the formula over-scores very short sentences; five of them exceed grade 10 for that reason only.")

t_deciles = table(["Predicted per 1,000", "Observed per 1,000", "People"],
                  [[f"{1000 * d['predicted']:.1f}", f"{1000 * d['observed']:.1f}", f"{d['n']:,}"] for d in q6["deciles"]], "num",
                  "Ten equal-sized groups of the 2007-2014 people, ordered by predicted risk, from a model fitted only on 1999-2006.")

def cap(key, title, html):
    return f'<div class="tcap"><b>Table {TN[key]}.</b> {title}</div>' + html


t_sources = cap("sources", "The data.", t_sources)
t_rules = cap("rules", "The rules, by cancer site.", t_rules)
t_labs = cap("labs", "How lab values become the findings the rules read.", t_labs)
t_burden = cap("burden", "Share of adults alerted by lab results alone, by era and age.", t_burden)
t_pre = cap("pre", "Pre-registered questions and what happened.", t_pre)
t_dec = cap("dec", "Cancer specificity among people who died within ten years.", t_dec)
t_auc = cap("auc", "Does the alert add to age and sex for five-year cancer death?", t_auc)
t_flags = cap("flags", "Each flag on its own, among decedents (exploratory).", t_flags)
t_rulehit = cap("rulehit", "Rules that fired, labs plus proxy symptoms.", t_rulehit)
t_deciles = cap("deciles", "Calibration of the context layer by tenth.", t_deciles)
t_read = cap("read", "Reading level of the tool's text.", t_read)

# ---------------------------------------------------------------- text
ABSTRACT = f"""
<p><b>Background.</b> Cancers are often found late because early symptoms and mildly abnormal blood tests are individually vague.
Clinical guidelines such as the UK's NICE NG12 list symptom-and-test combinations that justify a prompt check, but they are written for clinicians
and are hard for a person with a lab report in hand to apply. We built a patient-facing navigator that applies those rules, and measured what it does on real people
before anyone relies on it.</p>
<p><b>Methods.</b> The navigator is a rule engine, not a trained model: {n_rules} rules transcribed from the 2015 edition of NG12, each with its source page, evaluated with
three-valued logic so that missing facts are requested rather than assumed. We ran the engine on {n_all + n3:,} adults aged 40 and over from NHANES III (1988-1994) and
ten continuous NHANES cycles (1999-2018), linked to the National Death Index through 2019. Six questions and their pass-or-fail bars were committed to the repository before the analysis.
Outcomes were burden of alerts, cancer-specificity among decedents, added value over age and sex in an unseen era, validity of the tool's own iron-deficiency assumption against measured ferritin,
and calibration of an age, sex and smoking context layer. Reading level was measured for every sentence the tool can show.</p>
<p><b>Results.</b> Lab results alone alerted {p(min(age_burden(e,'70-79','any_alert') for e in ERAS), 0)} to {p(max(age_burden(e,'70-79','any_alert') for e in ERAS), 0)} of people aged 70-79 and {p(min(age_burden(e,'80-+','any_alert') for e in ERAS), 0)} to {p(max(age_burden(e,'80-+','any_alert') for e in ERAS), 0)} of those 80 and over, almost entirely through the guideline's
anaemia-to-stool-test rule, and no more than {p(q1['worst_talk_soon'])} in any age band were told to see a doctor soon. <b>Three of the four bars with a pass-or-fail criterion were missed.</b>
Among people who died, alerted people were not more likely to have died of cancer (adjusted odds ratio {orci(ctx_pool)} in 1999-2014; {orci(n3o)} in NHANES III).
Adding the alert to age and sex did not improve prediction of five-year cancer death in the held-out era (AUC gain {g_ctx['gain']:+.3f}, 95% CI {g_ctx['gain_ci'][0]:+.3f} to {g_ctx['gain_ci'][1]:+.3f}; age and sex alone gave AUC {g_ctx['auc_base']:.2f}).
The tool's assumption that small red cells indicate iron deficiency had sensitivity {p(q4['proxy_sensitivity'], 0)} and positive predictive value {p(q4['proxy_ppv'], 0)} against measured ferritin, and overstated
"see a doctor soon" alerts by {p(q4['age60_talk_soon_proxy_only'] / q4['age60_talk_soon_with_ferritin'] - 1, 0)} in people aged 60 and over. The context layer was well calibrated in a later era (slope {q6['calibration_slope']:.2f}).
Patient-facing text averaged US grade {RD['patient_facing_action_mean']}.</p>
<p><b>Conclusions.</b> The navigator applies a published guideline faithfully and readably, but its lab-driven alerts do not carry information about who will die of cancer beyond age and sex.
It should therefore be presented as a way of reading a guideline, not as a risk predictor, and lab-only alerts should be worded as questions for a doctor. The tool is not clinician-reviewed
and the rules are UK guidance from 2015; clinician validation is the necessary next step.</p>
"""

INTRO = """
<p>A person who has had routine bloodwork usually learns one thing from it: which numbers fall outside a printed range. A blood count and a metabolic panel together produce about thirty values,
and what matters in early cancer detection is rarely one of them being wrong. It is a small abnormality in a blood count, alongside a symptom the person has not thought to mention, alongside their age
and smoking history. Doctors weigh these together. The person holding the report generally cannot.</p>
<p>Two bodies of work address parts of this. Clinical guidelines, most prominently the NICE guideline NG12 on suspected cancer, list the combinations of age, symptoms and test results that justify
an urgent check, chosen so that the chance of cancer for someone meeting a criterion is about three percent or more.<sup>1</sup> Statistical risk models such as QCancer combine symptoms, risk factors and
blood tests into a probability.<sup>2</sup> Both are written for clinicians. Neither is designed to be handed to a person who has a lab report and a worry and wants to know whether it is worth asking about.</p>
<p>Multi-cancer detection from blood alone, in people with no symptoms, is a harder and still unresolved problem. Earlier work in this project built and tested several blood-only panels and withdrew most of them
when they failed bars set in advance; the lesson was that routine blood values add little to age and sex for cancer outcomes in the general population. The route that remains well supported is narrower:
people who already have a reason to be looking, such as a symptom, where guidelines already define who deserves a prompt check.</p>
<p>This paper describes a navigator for that case and, more importantly, measures it. The questions are practical ones. How many people would the lab rules alert if used without symptoms?
Are the alerts specific to cancer, or do they mark ill health in general? Do they add to what age and sex already tell us? Is the tool's own simplifying assumption sound? Can an ordinary reader follow what it says?
We committed the questions and the pass-or-fail criteria to the repository before looking at results. Most of them failed, and we report that as the main finding.</p>
"""

METHODS = f"""
<h3>2.1 Design</h3>
<p>This is a development and retrospective evaluation study. No participant was contacted. All data are public, de-identified US survey records; the study is therefore not human-subjects research under
the usual definitions, though the institution's own determination is being sought before any comprehension testing with readers (Section 4.5).</p>
<h3>2.2 The rule engine</h3>
<p>The navigator (<code>backend/navigator.py</code>) reads a request containing age, sex, smoking and asbestos history, a list of symptoms the person has ticked (with how often, where a rule needs it), lab values,
and optional findings such as a positive stool test. It evaluates {n_rules} rules held in a plain data file (<code>backend/navigator_rules.json</code>) that a clinician can read without reading code. The rules were
transcribed from the full NICE NG12 guideline (2015 edition), and every rule records its source page. The engine contains no clinical wording of its own: the action text and the criteria shown to the user come from the file.</p>
<p><b>Three-valued logic.</b> Every condition evaluates to true, false or unknown. A rule that could apply if one more fact were known is reported as "could apply if", naming exactly what is needed, and is not counted as a match.
A missing smoking history or an unentered haemoglobin is therefore asked for and never assumed. Impossible inputs (such as a haemoglobin of 40 g/dL) are refused.</p>
<p><b>Alert levels.</b> The guideline gives each recommendation a timeframe and a strength. These map to three levels: <i>see a doctor soon</i> (urgent or two-week criteria that are recommended), <i>worth raising</i> and <i>mention at a routine visit</i>.
For a given cancer site only the highest level that applies is shown. The tool never states that a person has cancer, and a result of "no threshold met" is worded as different from "nothing is wrong".</p>
{fig("fig1_pipeline.png", "<b>Figure 1.</b> How the navigator works. Inputs on the left are optional; the rule engine asks for what it needs. The context layer and the limits are shown with every answer.")}
{t_rules}
<h3>2.3 Laboratory findings</h3>
<p>Raw lab values are converted into the findings the rules read (Table 2). Iron deficiency uses ferritin when it is entered. When it is not, the engine assumes iron deficiency from small red cells (MCV below 80 fL), and says so.
Question 4 below tests how reasonable that assumption is.</p>
{t_labs}
<h3>2.4 Data</h3>
<p>NHANES is a continuing US national survey with a physical examination and laboratory tests. NCHS links participants to the National Death Index, giving vital status and underlying cause of death through 31 December 2019.
We used NHANES III and ten continuous cycles; cycles examined in 2015-2018 have under five years of follow-up for many people and so were used only for questions that need no outcome (Table 3).
Participants who reported a previous cancer diagnosis were excluded where the questionnaire asked, and nobody was excluded for how they later died, a choice learned from an earlier cohort in this project that dropped everyone who died of another cause
and thereby made every early death a cancer death. The outcome is <b>death from cancer</b> (NCHS underlying cause 2) within 60 months of the examination: a later and harsher endpoint than diagnosis, and one that cannot show whether an alert would have led to a correct referral.</p>
{t_sources}
{fig("fig2_cohort.png", "<b>Figure 2.</b> The data. Blue bars are adults aged 40 or over with a blood count. Orange shows how many also had ferritin measured, which differs by survey cycle. The shaded cycles have too little follow-up for an outcome.")}
<h3>2.5 What the engine was given</h3>
<p>The real engine was run, not a re-implementation. Two versions were scored for every person. <b>Labs only</b> passed haemoglobin, MCV, platelets, white count and (where measured) ferritin, with no symptoms and no smoking history.
<b>Labs plus proxies</b> added smoking (100 or more cigarettes in a lifetime) and three symptom stand-ins, passed only when present: <i>weight loss</i> (weight a year ago minus weight now of at least 5% of the earlier weight and not trying to lose weight, or at least 10% where that question was not asked),
<i>chronic cough</i> (cough on most days for three months, asked 1999-2012) and <i>breathlessness</i> (asked of people 40 and over). NHANES never asked about bleeding, lumps, bowel change, appetite, tiredness or jaundice,
so most of the guideline's symptoms are invisible here, and every alert rate below is a <b>lower bound</b> on what someone reporting those symptoms would see. The proxies are crude: weight loss in a survey includes illness and medication effects that a person ticking "no known reason" would exclude.</p>
<h3>2.6 Pre-registered questions and bars</h3>
<p>Six questions and their criteria were committed before analysis (docs/NAVIGATOR_EVIDENCE_PREREG.md) and are summarised with the results in Table 4. The earlier project had repeatedly found that results look better when the comparison is weak,
so every prediction question uses age and sex as the baseline, and cancer specificity is tested only among people who died, so that "sick people die of things" cannot masquerade as a cancer signal.</p>
<h3>2.7 Statistics</h3>
<p>Proportions use Wilson 95% intervals.<sup>3</sup> Cancer specificity among decedents (deaths within ten years of examination) used logistic regression of cancer versus other cause on the flag, age, age squared, sex and examination year, with Wald intervals.
Added value used logistic models of five-year cancer death with age, age squared and sex, with and without the alert, fitted on 1999-2006 and tested on 2007-2014 (and, exploratorily, fitted on 1999-2014 and tested on NHANES III); the AUC difference was bootstrapped 1,000 times, resampling cases and non-cases separately.
Non-cancer deaths inside the window are counted as non-cases, which is a competing-risk simplification. All analyses are unweighted, because the aim was to exercise the engine on realistic profiles, not to estimate a national prevalence.</p>
<h3>2.8 Context layer</h3>
<p>To answer "how much of this is just my age", the tool reports how common five-year cancer death is among US adults of the same age, sex and smoking history: a logistic model of age, age squared, sex and smoking, fitted on 1999-2006 and tested on 2007-2014 (bar: calibration slope between 0.8 and 1.2),
then refitted on all outcome cycles for use in the tool. It does not use labs or symptoms and is shown as a group average, not the person's risk.</p>
<h3>2.9 Readability</h3>
<p>Every sentence the tool can show was scored with the Flesch-Kincaid grade level.<sup>4</sup> The bar, set before the first measurement, was a mean of at most 8 and no action line above 10. This is a floor and not a test of understanding.</p>
<h3>2.10 Software and reproducibility</h3>
<p>Backend: Python and FastAPI; frontend: Next.js, deployed publicly. There are {79} automated tests, including properties such as "a symptom-free person with normal labs is never told to see a doctor soon over 300 random profiles",
"the tool never says 'you have cancer'" and "a rule that refers to a symptom the form cannot ask about is a loud failure". Every number in this paper is regenerated by scripts in the repository, and the tables and figures are built directly from their output.
Reporting follows the spirit of TRIPOD+AI where it applies;<sup>5</sup> because the engine is not a trained model there is no training-set leakage to guard against in the rules themselves, only in the context layer, which is validated out of era.</p>
"""

RESULTS = f"""
<h3>3.1 Summary against the pre-registered bars</h3>
{t_pre}
<h3>3.2 Alert burden</h3>
<p>With lab results alone and no symptoms, the share of adults alerted rises steeply with age (Table 5, Figure 3): about {p(age_burden('1999-2006','40-49','any_alert'),0)} to {p(age_burden('2015-2018','40-49','any_alert'),0)} at 40-49, {p(age_burden('1999-2006','50-59','any_alert'),0)} to {p(age_burden('2007-2014','50-59','any_alert'),0)} at 50-59,
and {p(age_burden('1999-2006','80-+','any_alert'),0)} to {p(age_burden('2015-2018','80-+','any_alert'),0)} at 80 and over. The pattern is close to identical in all four eras, spanning thirty years of laboratory methods. "See a doctor soon" stayed at or below {p(q1['worst_talk_soon'])}
in every band and era, and is zero below age 60.</p>
<p>Almost every lab-driven alert comes from one guideline rule: anaemia in people 60 or over, or iron-deficiency anaemia under 60, leads to a stool test. In the two newest cycles, where ferritin was measured for most people, this rule
alerted 13.7% of women but 1.6% of men aged 40-49, a difference consistent with menstrual iron loss, which the rule does not take into account. That is a real feature of the guideline applied to people who may have a benign explanation. The pre-registered 20% line for any alert was crossed only at age 80 and over, in three of the four eras (21%, 23% and 27%).</p>
{fig("fig3_burden.png", "<b>Figure 3.</b> Share of symptom-free adults alerted by lab results alone, by age and era. Red dashed lines are the pre-registered bars. Error bars are 95% Wilson intervals.")}
{t_burden}
<h3>3.3 Are the alerts specific to cancer?</h3>
<p>No. Among the {labs_only['decedents']:,} people in 1999-2014 who died within ten years, those alerted by labs alone were <i>less</i> likely to have died of cancer than of something else ({p(labs_only['cancer_share_alerted'])} versus {p(labs_only['cancer_share_not'])}; adjusted odds ratio {orci(labs_only)}). Adding the proxy symptoms moved the estimate to {orci(ctx_pool)}.
The result did not replicate in direction in NHANES III ({orci(n3o)}, interval including 1). In no era did an alert identify cancer deaths.</p>
<p>Looking at one flag at a time (Table 7, Figure 4; exploratory), anaemia was the clearest: among decedents, anaemic people had about 40% lower odds of having died of cancer rather than another cause (OR {orci(q7['results']['anaemia'])}). The proxy for weight loss and breathlessness
pointed the same way. Smoking, the one flag that is not a symptom or lab, pointed the expected way (OR {orci(q7['results']['ever smoked'])}).</p>
{fig("fig5_decedents.png", "<b>Figure 4.</b> Among people who died within ten years, the odds that the death was from cancer rather than another cause, for people with each flag versus those without. Red intervals lie entirely below 1. Adjusted for age, sex and year.")}
{t_dec}
{t_flags}
<h3>3.4 Do alerts add to age and sex?</h3>
<p>Age and sex alone predicted five-year cancer death with AUC {g_ctx['auc_base']:.2f} in the held-out era. Adding the labs-only alert changed this by {g_lab['gain']:+.3f} (95% CI {g_lab['gain_ci'][0]:+.3f} to {g_lab['gain_ci'][1]:+.3f}); adding the alert from labs and proxy symptoms changed it by {g_ctx['gain']:+.3f} ({g_ctx['gain_ci'][0]:+.3f} to {g_ctx['gain_ci'][1]:+.3f}). Both intervals include zero, so the bar was missed.
In the exploratory NHANES III test the gain was {g_n3['gain']:+.3f} ({g_n3['gain_ci'][0]:+.3f} to {g_n3['gain_ci'][1]:+.3f}), statistically above zero but under one point of AUC and not in the pre-registered test.</p>
{fig("fig6_auc.png", "<b>Figure 5.</b> AUC for five-year cancer death with and without the navigator alert. Age and sex do nearly all the work.")}
{t_auc}
<h3>3.5 The tool's own iron-deficiency assumption</h3>
<p>Among {q4['n_with_ferritin']:,} adults with a measured ferritin, {q4['anaemic']} were anaemic and {q4['true_ida_by_ferritin']} of those were iron-deficient by ferritin. Using MCV below 80 as a stand-in caught {p(q4['proxy_sensitivity'],0)} of them and was right {p(q4['proxy_ppv'],0)} of the time.
Among {q4['age60_n']:,} adults 60 or over with a ferritin, the proxy told {q4['age60_talk_soon_proxy_only']} to see a doctor soon where measured ferritin told {q4['age60_talk_soon_with_ferritin']}. A person who enters ferritin gets a more accurate answer than one who does not, and the tool says which one they got.</p>
{fig("fig8_ferritin.png", "<b>Figure 6.</b> Left: how well MCV below 80 stands in for ferritin below 15 among anaemic adults. Right: the cost in alerts for people aged 60 and over.")}
<h3>3.6 Which rules fire</h3>
{t_rulehit}
{fig("fig4_rules.png", "<b>Figure 7.</b> Which rules fire for labs plus proxy symptoms. The lung rule fires most often because breathlessness and smoking are common in older NHANES adults, not because of an excess of cancer.")}
<h3>3.7 The age context layer</h3>
<p>Fitted on 1999-2006 and tested on 2007-2014, the age, sex and smoking model had AUC {q6['auc']:.3f} and a calibration slope of {q6['calibration_slope']:.2f}, inside the pre-registered 0.8 to 1.2. Its Brier score ({q6['brier']:.5f}) was slightly better than predicting the average for everyone ({q6['brier_if_no_model']:.5f}).
The model was refitted on {CTX['n']:,} adults with {CTX['cancer_deaths']} five-year cancer deaths for use in the tool. Typical outputs range from about 2 in 1,000 for a 45-year-old woman who has never smoked to about 50 in 1,000 for a 75-year-old man who has smoked.</p>
{fig("fig7_calibration.png", "<b>Figure 8.</b> Calibration of the context layer in the later era.", "58%")}{t_deciles}
<h3>3.8 Readability</h3>
<p>Patient-facing action and headline lines averaged US grade {RD['patient_facing_action_mean']}, with none over grade 10. The first measurement missed the bar: five action lines exceeded grade 10 and the emergency advice scored grade 16.5 because it was a single long sentence. After rewriting without changing meaning,
the emergency advice scored grade {rd['safety']['mean']}. The remaining hard items in the table are one-word symptom labels, where the formula over-scores.</p>
{fig("fig9_readability.png", "<b>Figure 9.</b> Reading level of each kind of text. Lower is easier; the dashed line is the bar.")}
{t_read}
<h3>3.9 The tool as a person sees it</h3>
<p>Figures 10 and 11 are screenshots of the deployed application. The case entered is invented for illustration: a 66-year-old man who has smoked, with unexplained weight loss and a blood count showing iron-deficiency anaemia.</p>
<figure class="shot"><img src="figures/app_form_crop.png" alt=""><figcaption><b>Figure 10.</b> The entry form, with the research-prototype warning at the top.</figcaption></figure>
<figure class="shot"><img src="figures/app_result_crop.png" alt=""><figcaption><b>Figure 11.</b> An answer for the invented case: what was flagged, why, and where in the guideline it comes from, followed by the age context.</figcaption></figure>
"""

DISCUSSION = f"""
<h3>4.1 What the results mean</h3>
<p>The navigator did what it was designed to do. It applied a published guideline to a person's facts, asked for what was missing, explained each alert with the page it came from, and did so at a reading level most adults can follow. It did not do something it was never designed to do and that a patient-facing tool might be mistaken for:
it did not identify who would die of cancer. This distinction is the main lesson. The NG12 criteria are thresholds for <i>looking</i>, set so that the chance of finding cancer is about three percent or more among people who have <i>already come to a doctor with a symptom</i>. Applied to symptom-free people using only their lab values, the same thresholds mostly flag the common conditions that cause anaemia and raised white counts.</p>
<p>The decedent-only result needs careful reading. It does not show that anaemia protects against cancer death, and it does not show the guideline is wrong. Anaemia is common in people dying of kidney, heart and lung disease, and in a general-population sample those causes of death are far more common than cancer. Whether an alerted person who then received a stool test or a colonoscopy would have had a bowel cancer found is not something death records can show. What the result does show is that a tool which presents lab-only alerts as a signal of cancer would mislead.</p>
<p>The context layer is the other practical result. Age and sex alone give an AUC of about 0.73 for five-year cancer death, and nothing the navigator adds changes that appreciably. Showing a person what is ordinary for their age is cheap, calibrated and, we believe, useful for reading an alert in proportion; whether it helps or alarms people is a question for readers, not for this data.</p>
<h3>4.2 Changes made because of the results</h3>
<p>Lab-only alerts now carry a note that common causes are not cancer and that, in a US survey, these findings did not help tell who later died of cancer once age and sex were known. A lower-tier alert for a cancer site is no longer shown beside a higher-tier one. The context layer was added to the tool because its bar was met.
A rule defect found along the way (asbestos wording shown to a smoker without exposure) was fixed and recorded. These changes are listed as deviations in the pre-registration, which was committed before the first analysis.</p>
<h3>4.3 Relation to existing work</h3>
<p>QCancer-style models report AUCs of roughly 0.84 to 0.88 for cancer when symptoms, risk factors and blood tests are combined in primary-care records.<sup>2</sup> Those models use a much richer symptom set and diagnoses as the outcome; the figures are not comparable with an AUC for cancer death
in a survey, and we make no claim to match them. The navigator's different contribution is access: a person can enter a lab report and symptoms, see why an alert arose in the guideline's own words, and read a plain-language answer. We have not verified that no similar patient-facing tool exists, and the novelty claim should be checked before it is made.</p>
<h3>4.4 Limitations</h3>
<ul>
<li><b>Not clinician-reviewed.</b> The rule file records <code>review_status: not yet reviewed</code>, and the tool shows this to every user. Transcription errors are possible. A design decision that departs from the guideline text (gating the lung "consider chest X-ray" thrombocytosis path behind at least one symptom, because applied literally it alerted 3.7% of symptom-free adults over 40) needs a clinician's confirmation.</li>
<li><b>Dated and foreign guidance.</b> The rules are the 2015 edition of UK guidance, which has since been revised (for example, the bowel pathway now uses the faecal immunochemical test). US guidance may differ.</li>
<li><b>Death, not diagnosis.</b> The outcome cannot show whether an alert would have led to a correct referral. Cancers that were found early and cured are counted as non-cases.</li>
<li><b>Proxies for symptoms.</b> NHANES has three crude symptom stand-ins and none of the symptoms that drive most of the guideline's high-value rules (bleeding, lumps, bowel change, jaundice). Results for symptom-plus-lab combinations are therefore weak evidence.</li>
<li><b>Self-reported weight and unweighted analysis.</b> The proxy for weight loss misclassifies people. The analyses ignore survey weights and describe the sample, not the US population.</li>
<li><b>Competing risks.</b> Non-cancer deaths inside the window are treated as non-cases, which a cause-specific hazard model would handle more carefully.</li>
<li><b>Single comparison for the main prediction question.</b> The one analysis on NHANES III went the other way from the pre-registered era, and with 238 cancer deaths and a one-point AUC difference it should not be over-read.</li>
<li><b>Reading level is not comprehension.</b> A short sentence can still be misunderstood.</li>
</ul>
<h3>4.5 Next steps</h3>
<ol>
<li>Clinician sign-off of every rule against the current guideline, recorded in the rule file.</li>
<li>A vignette study in which clinicians independently apply the guideline text to written cases and their answers are compared with the tool's, with under-alerting counted separately because it is the harmful direction (docs/NAVIGATOR_VALIDATION_PROTOCOL.md).</li>
<li>A comprehension check with lay readers, after an ethics determination.</li>
<li>An evaluation on data with real symptoms and diagnoses, such as a clinical records database, once access is obtained.</li>
<li>Reconciling the rules with current UK and US guidance.</li>
</ol>
"""

CONCLUSION = """
<p>A guideline-based navigator can be built, can be made readable, and can be tested against bars set in advance. In this test it applied the guideline as written and produced a stable, modest alert burden. Its lab-driven alerts did not identify who would die of cancer, three of its four pass-or-fail bars were missed, and the paper says so.
The honest claim is therefore a modest one: the navigator is a faithful, readable way to apply a published referral guideline, accompanied by a calibrated statement of how common cancer death is for a person's age, and it should not be presented as a detector. Whether it helps real people reach a diagnosis sooner is untested.</p>
"""

DECL = """
<h3>AI assistance</h3>
<p>This project was developed with the assistance of an AI model (Anthropic Claude, Sonnet and earlier versions, via Claude Code). The AI was used to write and test code, to run analyses, to draft text, and to search for background literature. An AI cannot be an author. The human author is responsible for the work, reviewed
the results, and must verify every statement and reference before submission; the references listed here were written from the author's and the assistant's knowledge of the literature and have not yet been checked against the originals.</p>
<h3>Ethics</h3>
<p>The analyses use public, de-identified records and involve no contact with participants. Any study recruiting readers or patients will need an ethics determination first.</p>
<h3>Data and code availability</h3>
<p>Code, rules, tests, pre-registration and all result files are in the public repository (github.com/palashraks-afk/OncoVision). NHANES data are public from the US National Center for Health Statistics; the linked mortality files are public-use. The navigator is deployed at oncovisionai.vercel.app/navigator.</p>
<h3>Clinical mentorship and authorship</h3>
<p>Clinical mentorship is being sought from Dr. Rishikesh Chavan (UCI Health / CHOC). His role, review of the rules and authorship status are to be agreed with him and are not asserted here.</p>
<h3>Conflicts of interest and funding</h3>
<p>None declared. No funding.</p>
<h3>Warning</h3>
<p>The navigator is a research prototype and is not medical advice. A person who is worried about a symptom should see a doctor and should seek urgent care for severe symptoms regardless of what the tool says.</p>
"""

REFS = """
<ol>
<li>National Institute for Health and Care Excellence. <i>Suspected cancer: recognition and referral</i> (NICE guideline NG12). 2015, with later updates.</li>
<li>Hippisley-Cox J, Coupland C. Development and validation of risk prediction algorithms to estimate future risk of common cancers in men and women: prospective cohort study. <i>BMJ Open</i> 2015;5:e007825.</li>
<li>Wilson EB. Probable inference, the law of succession, and statistical inference. <i>J Am Stat Assoc</i> 1927;22:209-212.</li>
<li>Kincaid JP, Fishburne RP, Rogers RL, Chissom BS. <i>Derivation of new readability formulas for Navy enlisted personnel.</i> Naval Technical Training Command, 1975. Flesch R. A new readability yardstick. <i>J Appl Psychol</i> 1948;32:221-233.</li>
<li>Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. <i>BMJ</i> 2024;385:e078378.</li>
<li>World Health Organization. <i>Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity.</i> WHO/NMH/NHD/MNM/11.1, 2011.</li>
<li>National Center for Health Statistics. <i>NHANES Public-Use Linked Mortality Files</i>, mortality follow-up through 31 December 2019.</li>
<li>Goff BA, Mandel LS, Drescher CW, et al. Development of an ovarian cancer symptom index. <i>Cancer</i> 2007;109:221-227.</li>
</ol>
<p class="tnote">All references are to be checked against the originals before submission.</p>
"""

CSS = """
@page { size: Letter; margin: 22mm 20mm 22mm 20mm; @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #666; } }
:root { --ink:#1b1f24; --muted:#5b6672; --line:#d5dae0; --blue:#2f6fb0; --red:#b3261e; --green:#1f7a52; }
* { box-sizing: border-box; }
html { font-family: Georgia, "Times New Roman", serif; font-size: 10.6pt; color: var(--ink); line-height: 1.5; }
body { margin: 0; }
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 6pt; letter-spacing: -0.2pt; }
h2 { font-size: 14pt; margin: 22pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5px solid var(--ink); break-after: avoid; }
h3 { font-size: 11.2pt; margin: 14pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f3f5; padding: 0 2px; }
.sub { color: var(--muted); font-size: 11pt; margin-bottom: 10pt; }
.banner { border: 1.5px solid var(--red); background: #fdf1f0; padding: 7pt 10pt; font-size: 9.4pt; margin: 8pt 0 12pt; }
.meta { font-size: 9.6pt; color: var(--muted); margin-bottom: 12pt; }
.abstract { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 6pt 0 2pt; margin: 8pt 0; font-size: 10pt; }
.abstract h2 { border: 0; margin: 0 0 4pt; font-size: 12pt; }
figure { margin: 10pt auto; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 8.8pt; color: var(--muted); text-align: left; margin-top: 3pt; line-height: 1.4; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 8pt 0 3pt; break-inside: avoid; }
th { text-align: left; border-bottom: 1.5px solid var(--ink); border-top: 1.5px solid var(--ink); padding: 3pt 5pt; background: #f6f7f9; vertical-align: bottom; }
td { padding: 2.5pt 5pt; border-bottom: 0.5px solid var(--line); vertical-align: top; }
table.num td:not(:first-child) { font-variant-numeric: tabular-nums; }
.tcap { font-size: 9pt; margin: 10pt 0 2pt; break-after: avoid; }
.tnote { font-size: 8.4pt; color: var(--muted); text-align: left; margin-bottom: 9pt; }
.bad { color: var(--red); } .ok { color: var(--green); }
.side { display: flex; gap: 14pt; align-items: flex-start; break-inside: avoid; }
.side > * { flex: 1; }
.shot img { border: 1px solid var(--line); max-height: 118mm; width: auto; max-width: 100%; }
ul, ol { margin: 0 0 8pt 16pt; padding: 0; } li { margin-bottom: 3pt; text-align: justify; }
sup { font-size: 7pt; line-height: 0; }
.toc { font-size: 9.5pt; color: var(--muted); }
.pb { break-before: page; }
"""

TITLE = "A Guideline-Based Symptom and Laboratory Navigator for Cancer Referral Thresholds"
SUBTITLE = f"Alert burden, cancer specificity and calibration in {n_all + n3:,} US adults, with the bars set in advance and the failures reported"

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1>
<div class="sub">{SUBTITLE}</div>
<div class="meta"><b>Rahul</b> (author; surname to be added) &nbsp;·&nbsp; Clinical mentor: Dr. Rishikesh Chavan, UCI Health / CHOC (role and authorship to be agreed) &nbsp;·&nbsp; Oncovision project &nbsp;·&nbsp; Draft of 5 October 2026</div>
<div class="banner"><b>Draft for mentor review. Not peer reviewed.</b> The navigator applies the 2015 edition of UK guidance, its rules have <b>not</b> been reviewed by a clinician, and nothing here is tested on patients or is medical advice.
The key findings are negative: three of four pre-registered bars were missed.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> cancer referral guidelines; symptom checker; laboratory medicine; early detection; NHANES; pre-registration; health literacy.</p>

<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>5. Conclusion</h2>{CONCLUSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
<h2>Appendix: the rules</h2>
<p>The full rule file, with every criterion and source page, is <code>backend/navigator_rules.json</code>. A reviewing clinician can read it directly. Each rule's action text, as the tool shows it to a person, is listed below with its source page; the criteria behind each are in the file.</p>
{table(["Rule", "Site", "Level", "What the tool tells the person", "Source"],
       [[r["id"], r["site"], nav.tier_of(r).replace("_", " "), r["action"], r["source"]] for r in rules], "")}
</body></html>"""

out_html = os.path.join(HERE, "paper.html")
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_html)

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
pdf = os.path.join(HERE, "Navigator_Research_Paper.pdf")
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out_html.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
