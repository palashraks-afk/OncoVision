"""
Builds the audit paper (HTML, then PDF through headless Chrome) from the saved result files.

Run:  python docs/audit_paper/build_audit_paper.py
"""

import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A = json.load(open(os.path.join(ROOT, "experiments", "audit_result.json")))
RR = json.load(open(os.path.join(ROOT, "experiments", "risk_ranking_result.json")))
RF = json.load(open(os.path.join(ROOT, "experiments", "risk_followups_result.json")))
EX = pd.read_csv(os.path.join(ROOT, "data", "audit", "excluded.csv"))
P1, P2, A3, A4 = A["part1"], A["part2"], A["A3"], A["A4"]
T = pd.DataFrame(P1["table"])
src = open(os.path.join(ROOT, "docs", "paper", "build_paper.py"), encoding="utf-8").read()
CSS = src[src.index('CSS = """') + 9: src.index('"""', src.index('CSS = """') + 9)]


def p(x, d=0):
    return f"{100 * x:.{d}f}%"


FIGS = ["fig1_flow", "fig2_design", "fig3_auc_vs_age", "fig4_reanalysis", "fig5_contrast", "fig6_leak"]
FN = {n: i + 1 for i, n in enumerate(FIGS)}
TN = {n: i + 1 for i, n in enumerate(["hyp", "papers", "reanalysis", "ageonly", "kappa"])}


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


n = P1["included"]
a1, a2 = P1["A1"], P1["A2"]
xs_auc = P1["auc_xs"]
t_any40 = P2["tasks"]["any cancer, 40+"]
t_any20 = P2["tasks"]["any cancer, 20+"]
ob = P2["oversample_before_split"]
hi = int((T["auc"].dropna() >= 0.90).sum())
nauc = int(T["auc"].notna().sum())

hyp = [
    ["A1", "At least half validate only inside one pool of data", f"{a1['no_external']} of {n} ({p(a1['share_no_external'])}; 95% CI {a1['ci'][0]:.2f} to {a1['ci'][1]:.2f}) used a random split, cross-validation or no described validation, with no external cohort and no held-out cycle", "<b class='ok'>Supported</b>"],
    ["A2", "At most 20% report an age-only baseline", f"{a2['age_or_agesex_baseline']} of {n} ({p(a2['share'])}). {a2['any_comparator_logistic_cox_or_clinical']} of {n} compared with some logistic, Cox or clinical-variable model", "<b class='ok'>Supported</b>"],
    ["A3", "Median reported AUC for cancer status is within 0.05 of the age-only AUC", f"Median gap {A3['median_gap_vs_age_only_40plus']:+.3f} against age and sex alone in adults 40+ (the pre-set, conservative baseline), {A3['median_gap_vs_age_only_20plus']:+.3f} in adults 20+", "<b class='bad'>Not supported</b> (depends on the age range, which most papers do not state)"],
    ["A4", "Whole-picture models lose at least 0.03 AUC in a later-cycle split and missing flags identify the cycle", f"Gap {A4['random_vs_later_gap_any_40plus']:+.3f} for cancer status; flags predict the cycle with AUC {A4['era_from_flags_auc']:.2f}", "<b class='bad'>Not supported</b> for cancer status (the cycle shortcut is real but does not hurt here)"],
]
t_hyp = table("hyp", "The four pre-registered claims.", ["#", "Claim", "Result", "Verdict"], hyp, "Written down in docs/AUDIT_PREREG.md before any paper was coded.")

rows = []
for _, r in T.sort_values(["task", "year"]).iterrows():
    v = {"random": "random split", "cv": "cross-validation", "none": "none described", "split_unstated": "test set (unstated)"}[r["validation"]]
    rows.append([r["pmid"], int(r["year"]), r["site"], {"xs": "status", "mort": "mortality", "comorb": "comorbidity"}[r["task"]], v,
                 "yes" if r["ext"] else "no", "yes" if r["comparator"] else "no", f"{r['auc']:.3f}" if pd.notna(r["auc"]) else "not stated"])
t_papers = table("papers", "The 24 included papers, coded.", ["PMID", "Year", "Cancer site", "Task", "Validation inside NHANES", "External cohort or held-out cycle", "Comparator model", "Best AUC"], rows,
                 "AUC is the best test-set value the authors report for the cancer target (C-index for survival models). Papers are identified by PubMed ID so each can be looked up and re-coded.")

order = ["any cancer, 20+", "any cancer, 40+", "breast, 40+", "prostate, 40+", "colorectal, 40+", "digestive, 40+", "uterine, 40+", "bladder, 40+"]
rows = []
for k in order:
    r = P2["tasks"][k]
    row = [k, f"{r['n']:,}", r["cases"]]
    for sp in ("random", "later_cycles"):
        if sp in r:
            q = r[sp]
            row += [f"{q['age + sex']:.3f}", f"{q['whole picture, no missing flags']:.3f}", f"{q['whole picture + missing flags']:.3f}"]
        else:
            row += ["n/a", "n/a", "n/a"]
    rows.append(row)
t_re = table("reanalysis", "Re-analysis of self-reported cancer status, NHANES 1999-2014.", ["Task", "People", "Cases", "Random: age + sex", "Random: all, no flags", "Random: all + flags", "Later cycles: age + sex", "Later: all, no flags", "Later: all + flags"], rows,
             "AUC. Later-cycle split: fit on 1999-2006, test on 2007-2014. 'All' is about 200 laboratory, examination and questionnaire columns, gradient boosting. Cells marked n/a had too few cases in a split.")
ao = P2["age_only_by_site"]
rows = [[k, f"{v['auc']:.3f}", v["cases"], f"{v['n']:,}"] for k, v in ao.items()]
t_ao = table("ageonly", "AUC of age and sex alone, five-fold cross-validated, for each site.", ["Task", "AUC", "Cases", "People"], rows,
             "Used for the per-paper comparison in Figure 3.")
rows = [[k, v["kappa"], v["raw_agreement"]] for k, v in P1["cue_vs_adjudicated"].items()]
t_k = table("kappa", "Agreement between the automatic cue rules and the adjudicated codes.", ["Item", "Cohen's kappa", "Raw agreement"], rows,
            "Kappa is zero or low where almost all papers were coded the same way (a kappa of 0 for the age-only baseline reflects that no paper had one).")

human = T.sample(15, random_state=7)["pmid"].tolist()

ABSTRACT = f"""
<p><b>Background.</b> The US National Health and Nutrition Examination Survey (NHANES) is the data behind a large and growing number of machine-learning papers that claim to predict or detect cancer. Whether those claims are tested against the obvious baseline, age, and validated beyond the pool of data they were built on has not, to our knowledge, been measured.</p>
<p><b>Methods.</b> We pre-registered a fixed search and four claims before coding any paper. A Europe PMC search on 7 October 2026 returned 79 records, 56 with open full text. Every record was screened; {n} US-NHANES papers that built a model with a cancer target were coded for validation design, baselines, calibration, resampling and reported performance. We then re-analysed the task most of them use, predicting self-reported cancer history from other items recorded at the same visit, in {P2['n']:,} adults across eight survey cycles, with age and sex alone as the baseline and a later survey era held out.</p>
<p><b>Results.</b> {a1['no_external']} of {n} papers ({p(a1['share_no_external'])}) validated only inside NHANES with no external cohort and no held-out cycle. None of the {n} reported an age-only or age-and-sex baseline that we could find, and {a2['any_comparator_logistic_cox_or_clinical']} compared with any simpler model. Reported best AUCs for cancer status had a median of {xs_auc['median']:.2f}; {hi} of {nauc} papers reported 0.90 or more. In our re-analysis age and sex alone gave AUC {t_any40['random']['age + sex']:.2f} (adults 40 and over) to {t_any20['random']['age + sex']:.2f} (adults 20 and over), and using about 200 variables raised it by {t_any40['random']['whole picture + missing flags'] - t_any40['random']['age + sex']:.3f} to {t_any40['random']['whole picture + missing flags']:.3f}, and to {t_any40['later_cycles']['whole picture + missing flags']:.3f} in a later era. No analysis we ran reached 0.90 for any cancer. The prediction that missing-value flags would hurt a model in a later era was not supported for cancer status (AUC gap {A4['random_vs_later_gap_any_40plus']:+.3f}) although the flags predict survey cycle with AUC {A4['era_from_flags_auc']:.2f}; in our companion mortality analysis the same shortcut cost 0.13 AUC. Two of four pre-registered claims were supported.</p>
<p><b>Conclusions.</b> In this literature, internal-only validation without an age baseline is the norm, and many reported AUCs are high compared with what a large honest pipeline reproduces on the same survey. We propose an eight-item checklist for NHANES prediction papers, and a protocol that can be extended to the roughly 1,400 open-access papers a broader search returns.</p>
"""

INTRO = """
<p>Public health surveys are attractive to machine-learning researchers: they are free, large and carry hundreds of laboratory, examination and questionnaire items. NHANES in particular supports hundreds of papers a year, and a visible share of them claim that an algorithm can predict, detect or classify cancer. These claims matter because the people who read them include clinicians, funders and students deciding which approaches to build on.</p>
<p>Three problems are well known in prediction research generally. A model must be shown to beat the simplest sensible baseline, and for cancer that baseline is age, which on its own ranks cancer status and cancer death well.<sup>1</sup> It must be validated on data that played no part in building it, ideally from another time or place.<sup>2</sup> And every step that learns from data, such as imputation, feature selection and resampling, must be fitted inside the training data only.<sup>3</sup> A systematic review of machine-learning prediction studies found 87% at high risk of bias, mostly in the analysis.<sup>4</sup> A 2026 scoping review of 16 methodological papers proposed a guideline for the survey design of NHANES-type data, covering sampling weights and design-based validation.<sup>5</sup></p>
<p>What we could not find is a measurement, in one literature, of how often these problems occur, and a check of what they cost on the data in question. NHANES adds two things of its own. Its cancer information is mostly a self-reported history recorded at the same visit as the predictors, so a model can learn what being a cancer survivor looks like rather than what precedes cancer. And it is released in two-year cycles whose questionnaires and laboratory methods change, so patterns of what was asked or measured can identify the cycle.</p>
<p>This paper reports a pre-registered audit of NHANES machine-learning papers with a cancer target and a controlled re-analysis on the task they most often use. We set four claims and their criteria in advance and report them whether or not they held.</p>
"""

METHODS = f"""
<h3>2.1 Pre-registration</h3>
<p>The search, the inclusion rule, the coding items and four claims with the results that would count as support were committed to the project repository on 7 October 2026 (<code>docs/AUDIT_PREREG.md</code>) before any paper was coded and before the re-analysis was run. Two choices were made after that and are listed in Section 4.3.</p>
<h3>2.2 Search and inclusion</h3>
<p>Europe PMC, searched on 7 October 2026 for records from 2015 to 2026 whose abstracts mention NHANES (or its full name), a machine-learning term (machine learning, deep learning, random forest, XGBoost, gradient boosting, neural network, support vector) and a cancer term. The full query is in the pre-registration. It returned 79 records, 56 with open full text. There was no sampling: every record was screened. A paper was included if it used US NHANES and built, trained or reported a prediction or classification model with reported performance whose target was cancer at any site, or death from cancer. Excluded were association-only analyses where machine learning only ranked variables, models of other diseases, studies of the Korean survey (KNHANES), mechanistic and omics studies, and methods papers. Each of the 55 exclusions has a recorded reason.</p>
<h3>2.3 Coding</h3>
<p>Each included paper was coded for the task (cancer status recorded at the same visit, mortality, comorbidity), the validation design (random split, cross-validation only, none described, or held-out NHANES cycles or an outside cohort), whether an age-only or age-and-sex baseline was reported, whether any simpler comparator was reported, the best reported AUC, whether survey weights were used when fitting, calibration, resampling and any stated leakage guard. A first pass applied regular-expression cues to the text; a second pass read the abstract and the sentences the cues returned and decided each code. The second pass is the one reported. Items that need the methods section were marked not assessable for the {int(P1['included'] - P1['open_access_full_text_in_included'])} included papers with only an abstract available.</p>
<p>The cue rules and the adjudicated codes were compared (Table {TN['kappa']}). The adjudication was done by an AI assistant, not by two independent humans, and is the main limitation; the {len(human)} papers listed in Appendix A were chosen at random for a human to re-code.</p>
<h3>2.4 The controlled re-analysis</h3>
<p>We built the task most papers use: predicting a person's self-reported answer to "Has a doctor ever told you that you had cancer?" from the other items recorded at the same visit, in adults from the eight continuous NHANES cycles 1999-2014 ({P2['n']:,} adults, {P2['cancer_history']:,} with a cancer history). Predictors were about 200 laboratory, examination and questionnaire columns, plus an indicator for each column that was partly missing. Cancer-history questions were never predictors. For each task we compared (a) age and sex, (b) all variables without missing flags, and (c) all variables with missing flags (gradient boosting, depth 3, 300 trees), in a random 70/30 split, the usual choice in the audited papers, and in a split that fits on 1999-2006 and tests on 2007-2014. Site-specific tasks (breast, prostate, colorectal, digestive, uterine, bladder) used the cancer type recorded in the survey, with people without cancer as controls.</p>
<p>The age-only AUC for each site, five-fold cross-validated in adults 40 and over and in adults 20 and over, was compared with each audited paper's best reported AUC for that site. The verdict for claim A3 uses the adults 40 and over baseline, which is lower and so favours the papers. We also measured two mechanisms: how well the missing-value flags alone predict the survey cycle, and the inflation from duplicating cases before the train-test split.</p>
"""

RESULTS = f"""
<h3>3.1 What was found</h3>
{fig("fig1_flow", f"Records and exclusions. Of {P1['records_found']} records, {n} met the inclusion rule.")}
<p>{n} papers from {P1['years'][0]} to {P1['years'][1]} were included ({int(P1['open_access_full_text_in_included'])} with full text). {P1['tasks'].get('xs', 0)} predicted cancer status at the same visit, {P1['tasks'].get('comorb', 0)} a diabetes-cancer or heart disease-cancer comorbidity, and {P1['tasks'].get('mort', 0)} cancer death or survival. Sites were any cancer ({P1['sites'].get('any', 0)}), breast ({P1['sites'].get('breast', 0)}), prostate ({P1['sites'].get('prostate', 0)}), colorectal ({P1['sites'].get('colorectal', 0)}), digestive ({P1['sites'].get('digestive', 0)}), and uterine and bladder (one each).</p>
{t_hyp}
<h3>3.2 How the papers validated and compared</h3>
<p>Of {n} papers, {a1['validation_types'].get('random', 0)} used a single random split, {a1['validation_types'].get('cv', 0)} cross-validation only, {a1['validation_types'].get('none', 0)} described no validation, and {a1['validation_types'].get('split_unstated', 0)} mentioned a test set without saying how it was made. Three had any external cohort, in each case a Chinese hospital cohort, and one of those three also used earlier NHANES cycles as a held-out set. {a1['full_text_only']['n']} papers had full text; among them {p(a1['full_text_only']['share_no_external'])} had no external cohort or held-out cycle.</p>
<p>No paper had an age-only or age-and-sex-only baseline. Ten compared with some other simple model: logistic regression, a Cox model, or a clinical-variables model that usually included age but also other things. Calibration was reported in {P1['calibration_reported']['yes']} of {P1['calibration_reported']['assessable']} assessable papers. Resampling such as SMOTE was used in {P1['resampling_used']['yes']} of {P1['resampling_used']['assessable']}; of those, {P1['leakage_guard_among_resampling_papers']['guard_reported']} of {P1['leakage_guard_among_resampling_papers']['papers']} stated that resampling was confined to the training data. Survey weights were used in model fitting in none of the papers where this could be assessed; several used weights only for the descriptive or association analysis.</p>
{fig("fig2_design", "What the included papers did. Bars are percentages; counts are beside each bar. Calibration and resampling are out of papers where the item could be assessed.")}
<h3>3.3 What they reported</h3>
<p>Best reported AUCs ranged from {P1['auc_all']['min']:.2f} to {P1['auc_all']['max']:.3f}. For papers predicting cancer status the median was {xs_auc['median']:.3f} (interquartile range {xs_auc['iqr'][0]:.3f} to {xs_auc['iqr'][1]:.3f}). {hi} of {nauc} papers reported 0.90 or more, for dietary antioxidants and comorbidity, uterine cancer, heart disease and cancer comorbidity, a social-determinants model and caffeine and prostate cancer. {F('fig3_auc_vs_age')} places each against the AUC that age and sex alone give for the same site in the same survey.</p>
{fig("fig3_auc_vs_age", "Each dot is one paper's best reported AUC for cancer status. Lines are the AUC of age and sex alone, cross-validated, in adults 40 and over (orange) and 20 and over (red). Most papers sit near or only modestly above the lines; a few are far above.")}
{t_papers}
<h3>3.4 The re-analysis: what do all the variables add?</h3>
<p>For any cancer in adults 40 and over, age and sex alone gave AUC {t_any40['random']['age + sex']:.3f} in a random split. Adding about 200 variables raised it to {t_any40['random']['whole picture + missing flags']:.3f} (bootstrap 95% interval for the gain {t_any40['random']['gain_ci_whole_picture_flags'][0]:+.3f} to {t_any40['random']['gain_ci_whole_picture_flags'][1]:+.3f}), and in the later-cycle split from {t_any40['later_cycles']['age + sex']:.3f} to {t_any40['later_cycles']['whole picture + missing flags']:.3f}. In adults 20 and over the baseline was higher ({t_any20['random']['age + sex']:.3f}) and the full model reached {t_any20['random']['whole picture + missing flags']:.3f}. So other things recorded at the same visit do help predict cancer history, by about 0.05 to 0.06 AUC, which is plausible: cancer survivors differ from other people in their health, medicines and use of care. The sites told the same story ({TR('reanalysis')}, {F('fig4_reanalysis')}); for colorectal and bladder cancer the full model did not beat age and sex.</p>
{fig("fig4_reanalysis", "AUC for cross-sectional cancer status by site. Grey is age and sex alone. The full model with and without missing-value flags is nearly identical because the flags add nothing for this outcome.")}
{t_re}
{t_ao}
<p>Against these benchmarks, the highest AUC our full pipeline reached for any site and split was {max(r[sp][s] for r in P2['tasks'].values() for sp in ('random', 'later_cycles') if sp in r for s in ('whole picture + missing flags',)):.3f} (prostate cancer, random split). Reported AUCs of 0.90 or more for any cancer, uterine cancer or comorbidity are therefore well above what this much larger honest analysis produces on the same survey. That does not show those results are wrong: the papers used different samples, age ranges and definitions. It shows they need explaining, and almost none of the papers reported a baseline that would let a reader judge.</p>
<h3>3.5 When does the survey-cycle shortcut matter?</h3>
<p>The missing-value flags alone predict which survey cycle a person is from with AUC {A4['era_from_flags_auc']:.3f}, so the cycle is encoded in them. For cancer status, however, that did not hurt a later-cycle test: the whole-picture model scored {t_any40['random']['whole picture + missing flags']:.3f} in a random split and {t_any40['later_cycles']['whole picture + missing flags']:.3f} in the later-cycle split, a gap of {A4['random_vs_later_gap_any_40plus']:+.3f}. Claim A4 was therefore not supported. In our earlier work on death from cancer within five years the same kind of model fell from {RF['F2']['whole_picture_random_split_auc']:.3f} in a random split to {RF['F2']['whole_picture_era_split_auc']:.3f} in a later-cycle split. The difference is that survival outcomes depend on follow-up and baseline mortality, which differ by cycle, while a lifetime history of cancer does not. The shortcut costs something only when the outcome varies with the cycle.</p>
{fig("fig5_contrast", "The same whole-picture model, random split against a later-cycle split, for two outcomes. For cancer status the gap is small; for death within five years it is large.")}
<h3>3.6 A familiar pitfall, measured</h3>
<p>Duplicating cases to balance the classes before the train-test split puts copies of the same person on both sides. In a controlled test on the same task this raised test AUC from {ob['auc_split_then_oversample_train_only']:.3f} (oversampling the training set only) to {ob['auc_oversample_then_split']:.3f}, an inflation of {ob['auc_oversample_then_split'] - ob['auc_split_then_oversample_train_only']:.3f}. It is modest here because the model was regularised; with a more flexible model, or other leakage routes such as selecting variables on all the data, it could be larger. We did not determine the order of steps in any audited paper, apart from the {P1['leakage_guard_among_resampling_papers']['guard_reported']} that stated a guard.</p>
{fig("fig6_leak", "Test AUC when cases are duplicated before the split compared with after. The inflation is the part of the apparent performance that is not real.", "70%")}
<h3>3.7 Reliability of the coding</h3>
<p>The automatic cue rules were a poor substitute for reading the text (Table {TN['kappa']}): they agreed well with the adjudicated codes only for calibration and moderately for random splits and resampling, and poorly for external validation and survey weights. This is why the audit relies on the adjudicated codes and why a human second coder matters.</p>
{t_k}
"""

DISCUSSION = f"""
<h3>4.1 What the audit shows and does not show</h3>
<p>The two claims about practice were supported strongly. Almost nine in ten of the audited papers validated their models only inside NHANES, and none reported the baseline that would show what the model adds. The median reported AUC for predicting a person's cancer history from the rest of the visit was {xs_auc['median']:.2f}; age and sex alone gave 0.71 to 0.77, and a much larger honest analysis reached 0.76 to 0.82. A reader seeing 0.84 with no baseline cannot tell whether that is a good model or a good guess about age.</p>
<p>The two claims about the mechanism were not supported as stated. The survey-cycle shortcut exists, but for lifetime cancer status it did not cost anything in a later cycle. Reported AUCs were not "no better than age" under the pre-set baseline; the answer depends on the age range of each paper's sample, which most did not state clearly enough to match. We report this plainly: the stronger version of our hypothesis, that published performance is largely an age effect, was not borne out by this audit, and what is left is the narrower finding that baselines and external validation are absent and that some reported values are hard to reproduce.</p>
<h3>4.2 A checklist</h3>
<p>From the audit and the re-analysis we propose eight items for papers that predict cancer from NHANES-type data. They are not new principles; they are the ones that the audited papers most often did not meet.</p>
<ol>
<li><b>Report the baseline.</b> Give the AUC of age alone, and of age and sex, on the same people.</li>
<li><b>State the age range</b> and the outcome definition, so the baseline can be matched.</li>
<li><b>Say when the outcome is measured.</b> Cancer history reported at the same visit is not a prediction of cancer; call it classification.</li>
<li><b>Hold out whole survey cycles</b>, or an outside cohort, at least for the headline result.</li>
<li><b>Fit everything inside the training data:</b> imputation, scaling, feature selection and resampling. State the order.</li>
<li><b>Check whether missing-value flags identify the cycle</b> (a one-line test), and drop them if the outcome depends on the cycle.</li>
<li><b>Report calibration</b>, not only discrimination.</li>
<li><b>State how survey weights and design were handled</b> in fitting and in evaluation, or say they were not.</li>
</ol>
<h3>4.3 Deviations from the pre-registration</h3>
<ol>
<li>The pre-registration said the benchmark would be the age-only AUC "computed in Part 2" without saying for which ages. We reported both adults 20 and over and adults 40 and over and used the lower for the verdict, which favours the papers.</li>
<li>Papers whose targets were not clearly cancer were excluded after reading, with reasons recorded (Appendix B); the rule "target is cancer or death from cancer" was applied, so comorbidity papers that include cancer were kept.</li>
</ol>
<h3>4.4 Limitations</h3>
<ul>
<li><b>Small and one literature.</b> {n} papers is a census of one search, not a sample of all NHANES machine learning. A broader search (not restricted to a cancer term in the abstract) returns about 1,400 open-access records; the same protocol could be run on them.</li>
<li><b>One AI coder.</b> Adjudication was done by an AI assistant reading the text, with weak agreement between the cue rules and the final codes. A human second coder is needed, and {len(human)} papers are listed for that.</li>
<li><b>Open access only for methods items.</b> {int(P1['included'] - P1['open_access_full_text_in_included'])} included papers were coded from abstracts, so items such as the order of preprocessing are not assessable for them. If anything this makes the audit conservative for problems and generous for good practice.</li>
<li><b>Reported AUCs are not comparable.</b> Papers use different samples, ages and definitions, and report training, validation or test values. We took the best test or validation value stated for the cancer target.</li>
<li><b>Not a verdict on any paper.</b> The audit measures what was reported, not whether a result is true.</li>
<li><b>Search and database.</b> Abstract-based search in one database, one date.</li>
</ul>
<h3>4.5 Relation to other work</h3>
<p>The risk-of-bias review of machine-learning prediction studies found most at high risk of bias in the analysis;<sup>4</sup> this audit is consistent with that and narrows it to one dataset and one outcome. The survey-aware guideline<sup>5</sup> addresses weights and design; our checklist adds baselines, age-range matching, outcome timing and cycle-held-out validation. Leakage as a cause of unreproducible machine-learning results has been documented across fields;<sup>3</sup> here we measured one such route on this survey. We did not find a prior audit of this kind for NHANES and cancer, but a short search does not show that none exists.</p>
"""

APPX = f"""
<h2>Appendix A. Papers listed for human re-coding</h2>
<p>Fifteen included papers chosen at random (seed 7) from the {n}. PubMed IDs: {", ".join(str(x) for x in human)}. A reviewer who codes them independently would allow a human-versus-AI kappa to be added.</p>
<h2>Appendix B. Reasons for exclusion</h2>
<table class="num"><thead><tr><th>PMID</th><th>Reason</th></tr></thead><tbody>{"".join(f"<tr><td>{r['pmid']}</td><td>{r['reason']}</td></tr>" for _, r in EX.iterrows())}</tbody></table>
"""

DECL = """
<h3>AI assistance</h3>
<p>This study was carried out with an AI assistant (Anthropic Claude, via Claude Code), which wrote the search, extraction and analysis code, read and coded the papers, ran the re-analysis and drafted the text. An AI cannot be an author. The human author set the question, is responsible for the work, and must verify the coding and every statement before submission.</p>
<h3>Ethics, data and code</h3>
<p>The re-analysis uses public, de-identified NHANES data and NCHS mortality files; no participants were contacted. Search records, adjudicated codes, exclusion reasons, code and results are in the public repository (github.com/palashraks-afk/OncoVision). Copyrighted full text is not redistributed. Clinical mentorship and authorship are not asserted here.</p>
<h3>Conflicts of interest and funding</h3>
<p>None declared. No funding.</p>
"""

REFS = """
<ol>
<li>Hippisley-Cox J, Coupland C. Development and validation of risk prediction algorithms to estimate future risk of common cancers in men and women: prospective cohort study. <i>BMJ Open</i> 2015;5:e007825.</li>
<li>Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. <i>Ann Intern Med</i> 2019;170:51-58.</li>
<li>Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i> 2023;4:100804.</li>
<li>Andaur Navarro CL, Damen JAA, Takada T, et al. Risk of bias in studies on prediction models developed using supervised machine learning techniques: systematic review. <i>BMJ</i> 2021;375:n2281.</li>
<li>Oh Y, Zheng HW, Feng J, Bui AAT. Survey-aware machine learning: a guideline for valid population health inference based on scoping review. arXiv:2605.08963, 2026 (preprint).</li>
<li>Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. <i>BMJ</i> 2024;385:e078378.</li>
<li>Centers for Disease Control and Prevention, National Center for Health Statistics. National Health and Nutrition Examination Survey (NHANES) data and documentation; NCHS public-use linked mortality files.</li>
</ol>
<p class="tnote">References 1 to 4 and 6 were looked up by title in Crossref and match. Reference 5 is a preprint read from its arXiv abstract page. Reference 7 is a data product.</p>
"""

TITLE = "How Much of NHANES Machine-Learning Cancer Prediction Survives Honest Validation?"
SUB = "A pre-registered audit of the literature and a controlled re-analysis, with the failures reported"
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>
<h1>{TITLE}</h1><div class="sub">{SUB}</div>
<div class="meta"><b>Palash Rakshit</b> (author) &nbsp;·&nbsp; Draft of 7 October 2026 &nbsp;·&nbsp; Oncovision project</div>
<div class="banner"><b>Draft. Not peer reviewed.</b> Coding was done by an AI assistant and has not been checked by a second human coder. Two of the four pre-registered claims were not supported, and the paper reports that. Nothing here is medical advice.</div>
<div class="abstract"><h2>Abstract</h2>{ABSTRACT}</div>
<p class="meta"><b>Keywords:</b> NHANES; machine learning; cancer prediction; meta-research; validation; baseline; data leakage; pre-registration.</p>
<h2>1. Introduction</h2>{INTRO}
<h2>2. Methods</h2>{METHODS}
<h2 class="pb">3. Results</h2>{RESULTS}
<h2 class="pb">4. Discussion</h2>{DISCUSSION}
<h2>Declarations</h2>{DECL}
<h2>References</h2>{REFS}
{APPX}
</body></html>"""
out = os.path.join(HERE, "audit_paper.html")
open(out, "w", encoding="utf-8").write(html)
pdf = os.path.join(HERE, "NHANES_ML_Cancer_Audit.pdf")
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
