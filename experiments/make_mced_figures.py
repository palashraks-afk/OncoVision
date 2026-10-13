"""
Figures for the multi-cancer test law paper (docs/mced_paper/figures/), and one exploratory number set that the
paper also reports: the share of Medicare-eligible five-year cancer deaths covered in each year 2029-2045 under
the enacted rule, holding the 2005-2009 NHIS age and risk structure fixed (written to mced_coverage_by_year.json).

Run:  python experiments/make_mced_figures.py
"""

import json
import os
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cancer_age import Fit  # noqa: E402
from mced_law import load, wmean  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "mced_paper", "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "experiments", "mced_law_result.json")))
L = json.load(open(os.path.join(ROOT, "experiments", "free_data_ladder_result.json")))
INK, MUTED, ACC, FLAG, RULE = "#1d2530", "#6b7480", "#2f7f87", "#b5523f", "#d5d9de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def main():
    d = load()
    te = d[(d["year"] >= 2005) & d["weight"].notna()].reset_index(drop=True)
    y, w, age, mc = te["y"].to_numpy(float), te["weight"].to_numpy(float), te["age"].to_numpy(), te["medicare"].to_numpy()
    dis = (age >= 50) & (age < 65) & (mc == 1)
    elig = (age >= 65) | dis
    years = list(range(2029, 2046))
    share_d, share_p = [], []
    for yr in years:
        cov = dis | ((age >= 65) & (age <= 65 + (yr - 2029)))
        share_d.append(float((y * w * cov * elig).sum() / (y * w * elig).sum()))
        share_p.append(float((w * cov * elig).sum() / (w * elig).sum()))
    json.dump({"years": years, "share_of_cancer_deaths": [round(x, 4) for x in share_d], "share_of_population": [round(x, 4) for x in share_p],
               "note": "stylised: holds the 2005-2009 age and risk structure fixed; exploratory"},
              open(os.path.join(ROOT, "experiments", "mced_coverage_by_year.json"), "w"), indent=2)

    # Fig 1: coverage over time
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(years, [100 * s for s in share_p], color=MUTED, lw=1.6, label="share of Medicare-eligible adults covered")
    ax.plot(years, [100 * s for s in share_d], color=ACC, lw=2.2, label="share of their five-year cancer deaths covered")
    ax.axhline(50, color=RULE, lw=1, ls="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_xlabel("Year")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig1_coverage")

    # Fig 2: yield per 1,000 tests by group
    g = R["Q2"]
    names = list(g)
    short = ["50-64, no Medicare\n(not eligible)", "50-64, Medicare\n(covered 2029)", "65\n(covered 2029)", "66-74\n(never)", "75-84\n(never)"]
    vals = [g[n]["cancers_found_per_1000_tests"] for n in names]
    cols = [MUTED, ACC, ACC, FLAG, FLAG]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}\nPPV {g[names[i]]['ppv']:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(vals)), short, fontsize=7.5)
    ax.set_ylabel("Cancers found per 1,000 tests")
    ax.set_ylim(0, max(vals) * 1.35)
    save(fig, "fig2_yield")

    # Fig 3: triage inside the covered group
    q = R["Q3"]
    sh = ["0.2", "0.3", "0.5"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(sh))
    ax.bar(x - 0.18, [q[s]["random"]["cancers_found_per_1000_tests"] for s in sh], 0.36, color=MUTED, label="test at random")
    ax.bar(x + 0.18, [q[s]["triaged"]["cancers_found_per_1000_tests"] for s in sh], 0.36, color=ACC, label="test the highest risk by free answers")
    for i, s in enumerate(sh):
        ax.text(i + 0.18, q[s]["triaged"]["cancers_found_per_1000_tests"] + 0.2, f"x{q[s]['yield_ratio']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{float(s):.0%} of covered group tested" for s in sh], fontsize=8)
    ax.set_ylabel("Cancers found per 1,000 tests")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig3_triage")

    # Fig 4: the free-data ladder, cancer vs other-cause death
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = ["Activity data\n(cancer death)", "Checkup labs\n(cancer death)", "Activity data\n(other-cause death)"]
    gains = [L["gains"]["QW vs Q"], L["gains"]["QC vs Q"], L["other_cause"]["gain"]]
    for i, gg in enumerate(gains):
        ax.errorbar(gg["gain"], i, xerr=[[gg["gain"] - gg["ci"][0]], [gg["ci"][1] - gg["gain"]]], fmt="o", color=ACC if i == 2 else INK, capsize=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(0.01, color=RULE, lw=1, ls="--")
    ax.set_yticks(range(3), labels, fontsize=8)
    ax.set_xlabel("AUC gain over free questions (95% interval)")
    ax.invert_yaxis()
    save(fig, "fig4_ladder")
    print("figures written; coverage 2029", share_d[0], "2035", share_d[6], "2045", share_d[-1])


if __name__ == "__main__":
    main()
