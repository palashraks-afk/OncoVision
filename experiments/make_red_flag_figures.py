"""
Figures for the red-flag-labs paper, from experiments/red_flag_labs_result.json only.
Writes docs/red_flag_paper/figures/fig1..fig4.

Run:  python experiments/make_red_flag_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "experiments", "red_flag_labs_result.json")))
A, B5, B10 = R["A_societal_2017_2020"], R["B_scientific_5y"], R["B_scientific_10y"]
FIG = os.path.join(ROOT, "docs", "red_flag_paper", "figures")
os.makedirs(FIG, exist_ok=True)
INK, MUTED, ACC, FLAG, RULE = "#1d2530", "#6b7480", "#2f7f87", "#b5523f", "#d5d9de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def main():
    # Fig 1: from 2.1 million to 1.4 million
    p = A["ida_prevalence"]
    mo = A["missed_opportunity_exploratory"]
    vals = [p["millions"], A["ida_untreated_share"]["millions"], mo["ida_prevalence_among_regular_attenders"]["millions_untreated_among_them"]]
    labels = ["Have iron-deficiency\nanaemia (AGA 2020)", "Not being treated\nfor anaemia", "Not treated, despite a usual\ndoctor and 2+ visits last year"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.barh(range(3), vals, color=[INK, FLAG, FLAG], height=0.55)
    ax.errorbar(vals[0], 0, xerr=[[vals[0] - p["millions_ci"][0]], [p["millions_ci"][1] - vals[0]]], color=MUTED, capsize=3)
    for i, v in enumerate(vals):
        ax.text((p["millions_ci"][1] if i == 0 else v) + 0.06, i, f"{v:.1f} million", va="center", fontsize=9)
    ax.set_yticks(range(3), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("US adults 50+ (men, and women past menopause), millions")
    save(fig, "fig1_funnel")

    # Fig 2: untreated share by contact with care, and the baseline
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    reg = {"est": mo["usual_care_and_2plus_visits"]["untreated_share_of_ida"], "ci": mo["usual_care_and_2plus_visits"]["ci"]}
    allu = A["ida_untreated_share"]
    for i, (lab, e) in enumerate((("All with iron-deficiency anaemia", allu), ("Usual doctor and 2+ visits last year", reg))):
        ax.errorbar(100 * e["est"], i, xerr=[[100 * (e["est"] - e["ci"][0])], [100 * (e["ci"][1] - e["est"])]], fmt="o", color=FLAG, capsize=3)
        ax.text(100 * e["ci"][1] + 1.5, i, f"{100 * e['est']:.0f}%", va="center", fontsize=9)
    ax.set_yticks(range(2), ["All with iron-deficiency anaemia", "Usual doctor and 2+ visits last year"], fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_xlabel("Share not being treated for anaemia, % (95% interval)")
    save(fig, "fig2_untreated")

    # Fig 3: prevalence by race and ethnicity (stratifier only)
    g = A["ida_prevalence_by_group"]
    names = [k for k in g if g[k]["ida_n"] >= 8]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, n in enumerate(names):
        e = g[n]
        ax.errorbar(100 * e["est"], i, xerr=[[100 * (e["est"] - e["ci"][0])], [100 * (e["ci"][1] - e["est"])]], fmt="o", color=INK, capsize=3)
    ax.axvline(100 * p["est"], color=RULE, ls="--")
    ax.set_yticks(range(len(names)), [f"{n} (n={g[n]['n']})" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Iron-deficiency anaemia, % of adults 50+ (dashed: all)")
    save(fig, "fig3_groups")

    # Fig 4: odds ratios for cancer and other-cause death, 5 and 10 years
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rows = []
    for yrs, b in (("5-year", B5["pooled"]), ("10-year", B10["pooled"])):
        rows.append((f"{yrs} cancer death, iron-deficiency anaemia", b["or_cancer_ida"], b["or_cancer_ida_ci"], ACC))
        rows.append((f"{yrs} other-cause death, iron-deficiency anaemia", b["or_other_ida"], b["or_other_ida_ci"], MUTED))
        rows.append((f"{yrs} cancer death, non-iron anaemia", b["or_cancer_nia"], b["or_cancer_nia_ci"], INK))
    for i, (lab, est, ci, col) in enumerate(rows):
        ax.errorbar(est, i, xerr=[[est - ci[0]], [ci[1] - est]], fmt="o", color=col, capsize=3)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio vs no anaemia (95% interval)")
    save(fig, "fig4_odds")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
