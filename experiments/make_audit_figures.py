"""
Figures for the audit paper, drawn from experiments/audit_result.json and
experiments/risk_ranking_result.json / risk_followups_result.json so they cannot disagree
with the tables.

Run:  python experiments/make_audit_figures.py     writes docs/audit_paper/figures/*.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "audit_paper", "figures")
os.makedirs(FIG, exist_ok=True)
A = json.load(open(os.path.join(ROOT, "experiments", "audit_result.json")))
RR = json.load(open(os.path.join(ROOT, "experiments", "risk_ranking_result.json")))
RF = json.load(open(os.path.join(ROOT, "experiments", "risk_followups_result.json")))
P1, P2 = A["part1"], A["part2"]
EX = pd.read_csv(os.path.join(ROOT, "data", "audit", "excluded.csv"))
TABLE = pd.DataFrame(P1["table"])

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, ORANGE, TEAL, RED, GREY = "#2f6fb0", "#d9822b", "#2a9d8f", "#c0392b", "#9aa5b1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 220, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})


def save(name):
    plt.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close()
    print("  wrote", name)


def fig_flow():
    reasons = EX["reason"].str.lower()
    cats = {
        "Association or mechanism study, no model performance": int(reasons.str.contains("association|mechanistic|omics|mendelian|clustering|statistical method|psa association|single-marker|single-index|no model performance|no ml|ranks variables").sum()),
        "Target is not cancer (another disease, ageing, other group)": int(reasons.str.contains("target is|targets|chronic disease|biological age|microbiome|obesity|hyperuric|hospital data|multimorbidity").sum()),
        "Different survey (Korea NHANES)": int(reasons.str.contains("korea").sum()),
    }
    cats["Unclear or other"] = len(EX) - sum(cats.values())
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 40)

    def box(x, y, w, h, text, col, size=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=col, ec="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc="none", ec=col, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.55)
    box(1, 24, 24, 14, "79 records\nfrom the fixed Europe PMC\nsearch, 7 October 2026\n(56 with open full text)", BLUE)
    box(18, 1, 68, 15, "55 excluded\n\n" + "\n".join(f"{v}   {k}" for k, v in cats.items() if v), RED, 8.6)
    box(79, 24, 24, 14, f"{P1['included']} included\nUS NHANES prediction\nmodels with a cancer\ntarget", TEAL)
    for a, b in (((26, 31), (78, 31)), ((52, 31), (52, 17.4))):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.set_title("How many papers were found, and how many were included")
    save("fig1_flow.png")


def fig_design():
    t = TABLE
    n = len(t)
    items = [
        ("Validated only inside NHANES\n(random split, cross-validation, or none)", int((~t["ext"]).sum()), n),
        ("Held-out survey cycles or an outside cohort", int(t["ext"].sum()), n),
        ("Reported an age-only or age + sex baseline", int(t["agesex"].sum()), n),
        ("Compared with any logistic, Cox or clinical model", int((t["agesex"] | t["comparator"]).sum()), n),
        ("Reported calibration", int(t["calib"].fillna(False).sum()), int(t["calib"].notna().sum())),
        ("Used resampling (SMOTE or similar)", int(t["resample"].fillna(False).sum()), int(t["resample"].notna().sum())),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(items))[::-1]
    for yi, (lab, k, d) in zip(y, items):
        ax.barh(yi, 100 * k / d, color=BLUE if "Validated only" in lab else TEAL, height=0.6)
        ax.text(100 * k / d + 1.5, yi, f"{k} of {d}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of included papers")
    ax.grid(axis="y", visible=False)
    ax.set_title("What the 24 included papers did")
    save("fig2_design.png")


def fig_auc():
    xs = TABLE[(TABLE["task"] == "xs") & TABLE["auc"].notna()].copy()
    sites = ["any", "breast", "prostate", "colorectal", "digestive", "uterine", "bladder"]
    ao = P2["age_only_by_site"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for i, s in enumerate(sites):
        v = xs[xs["site"] == s]["auc"].astype(float).to_numpy()
        if len(v):
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)) * (len(v) > 1), v, color=BLUE, s=42, zorder=3, label="a published paper's best AUC" if i == 0 else None)
        for key, col, off, lab in ((f"{s}, 40+", ORANGE, -0.28, "age + sex alone, adults 40+"), (f"{s}, 20+", RED, 0.28, "age + sex alone, adults 20+")):
            if key in ao:
                ax.plot([i + off - 0.18, i + off + 0.18], [ao[key]["auc"]] * 2, color=col, lw=3, zorder=2, label=lab if i == 0 else None)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([f"{s}\n(n = {int((xs['site'] == s).sum())})" for s in sites], fontsize=9)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("AUC for self-reported cancer history")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.set_title("Published AUCs against what age and sex alone give in the same survey")
    save("fig3_auc_vs_age.png")


def fig_reanalysis():
    tasks = ["any cancer, 20+", "any cancer, 40+", "breast, 40+", "prostate, 40+", "digestive, 40+"]
    sets = ["age + sex", "whole picture, no missing flags", "whole picture + missing flags"]
    cols = [GREY, TEAL, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    for ax, sp, title in zip(axes, ("random", "later_cycles"), ("Random split (what most papers do)", "Fit on 1999-2006, test on 2007-2014")):
        x = np.arange(len(tasks))
        w = 0.26
        for i, (s, c) in enumerate(zip(sets, cols)):
            v = [P2["tasks"][t][sp][s] for t in tasks]
            ax.bar(x + (i - 1) * w, v, w, color=c, label=s)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace(", ", "\n") for t in tasks], fontsize=8.5)
        ax.set_ylim(0.5, 0.95)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("AUC, cross-sectional cancer status")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Re-analysis: how much do all the variables add to age and sex, and does it survive a later era?", x=0.06, ha="left", y=1.03, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig4_reanalysis.png")


def fig_contrast():
    xs_t = P2["tasks"]["any cancer, 40+"]
    rows = [("Cancer status\nat the same visit", xs_t["random"]["whole picture + missing flags"], xs_t["later_cycles"]["whole picture + missing flags"]),
            ("Death from cancer\nwithin 5 years", RF["F2"]["whole_picture_random_split_auc"], RF["F2"]["whole_picture_era_split_auc"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=GREY, label="random split of all cycles")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=ORANGE, label="fit on earlier cycles, test on later")
    for xi, r in zip(x, rows):
        ax.text(xi - w / 2, r[1] + 0.006, f"{r[1]:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, r[2] + 0.006, f"{r[2]:.3f}", ha="center", fontsize=9)
        ax.text(xi, 0.52, f"gap {r[1] - r[2]:+.3f}", ha="center", fontsize=9, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("AUC of the whole-picture model with missing flags")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("The survey-cycle shortcut only bites when the outcome depends on the cycle")
    save("fig5_contrast.png")


def fig_leak():
    ob = P2["oversample_before_split"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    v = [ob["auc_split_then_oversample_train_only"], ob["auc_oversample_then_split"]]
    ax.bar(["Split first,\noversample training only", "Oversample cases,\nthen split"], v, color=[TEAL, RED], width=0.5)
    for i, a in enumerate(v):
        ax.text(i, a + 0.006, f"{a:.3f}", ha="center")
    ax.set_ylim(0.6, 0.86)
    ax.set_ylabel("AUC on the test set")
    ax.set_title("A familiar pitfall, measured: copies of the same person on both sides")
    save("fig6_leak.png")


if __name__ == "__main__":
    for f in (fig_flow, fig_design, fig_auc, fig_reanalysis, fig_contrast, fig_leak):
        f()
