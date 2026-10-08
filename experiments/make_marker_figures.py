"""
Figures for the cheap-blood-markers paper, drawn from the saved result files.

Run:  python experiments/make_marker_figures.py     writes docs/marker_paper/figures/*.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "marker_paper", "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_result.json")))
S = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_sensitivity_result.json")))
NAMES = list(R["indices"])
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, ORANGE, TEAL, RED, GREY = "#2f6fb0", "#d9822b", "#2a9d8f", "#c0392b", "#9aa5b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})


def save(name):
    plt.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close()
    print("  wrote", name)


def fig_logic():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)

    def box(x, y, w, h, text, col, size=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=col, ec="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc="none", ec=col, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.5)
    box(1, 14, 22, 12, "12,645 US adults 40+\nblood count at the exam\nfollowed up to ten years", BLUE)
    box(38, 28, 25, 9, "died of cancer", RED)
    box(38, 15, 25, 9, "died of something else", ORANGE)
    box(38, 2, 25, 9, "alive at ten years", GREY)
    box(72, 25, 27, 13, "Test 1: does the marker add to\nage, sex, smoking and BMI,\nin an era it was not fitted on?", TEAL, 8.6)
    box(72, 8, 27, 13, "Test 2: among those who died,\nis it linked to cancer more\nthan to the other causes?", TEAL, 8.6)
    for y in (32.5, 19.5, 6.5):
        ax.annotate("", xy=(37.4, y), xytext=(23.6, 20), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("", xy=(71.4, 31.5), xytext=(63.6, 32.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("", xy=(71.4, 14.5), xytext=(63.6, 19.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.set_title("The design: a marker has to be about cancer, not about being unwell")
    save("fig1_logic.png")


def fig_gain():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y = np.arange(len(NAMES))[::-1]
    for yi, n in zip(y, NAMES):
        r = R["indices"][n]
        ax.plot(r["T1_ci"], [yi + 0.14] * 2, color=RED, lw=2)
        ax.plot(r["T1_gain"], yi + 0.14, "o", color=RED, ms=6, label="death from cancer" if n == NAMES[0] else None)
        ax.plot(r["T3_ci"], [yi - 0.14] * 2, color=ORANGE, lw=2)
        ax.plot(r["T3_gain"], yi - 0.14, "s", color=ORANGE, ms=6, label="death from another cause" if n == NAMES[0] else None)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(0.01, color=MUTED, lw=1, ls="--")
    ax.text(0.0105, len(NAMES) - 0.35, "bar: +0.01", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("gain in AUC from adding the marker to age, sex, smoking and BMI (fit 1999-2004, test 2005-2008)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("None of the twelve markers improves prediction of cancer death")
    save("fig2_gain.png")


def fig_frailty():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    y = np.arange(len(NAMES))[::-1]
    for yi, n in zip(y, NAMES):
        r = R["indices"][n]
        ax.plot(r["or_cancer_ci"], [yi + 0.14] * 2, color=RED, lw=2)
        ax.plot(r["or_cancer_death"], yi + 0.14, "o", color=RED, ms=6, label="death from cancer" if n == NAMES[0] else None)
        ax.plot(r["or_noncancer_ci"], [yi - 0.14] * 2, color=ORANGE, lw=2)
        ax.plot(r["or_noncancer_death"], yi - 0.14, "s", color=ORANGE, ms=6, label="death from another cause" if n == NAMES[0] else None)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6])
    ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6"])
    ax.set_yticks(y)
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("odds ratio per one standard deviation, against people still alive at ten years")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Markers move with other causes of death far more than with cancer death")
    save("fig3_frailty.png")


def fig_specific():
    subs = ["all (the pre-registered analysis)", "first 24 months removed", "women", "men", "ages 40-59", "ages 60 and over"]
    names = ["NLR", "SII", "SIRI", "NPAR", "MLR", "PLR", "ALI", "PNI", "RDW"]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    cols = [INK, TEAL, ORANGE, BLUE, "#8e6bbf", GREY]
    y = np.arange(len(names))[::-1]
    for k, (s, c) in enumerate(zip(subs, cols)):
        for yi, n in zip(y, names):
            o = S[s]["or"][n]
            off = (k - 2.5) * 0.11
            ax.plot(o["ci"], [yi + off] * 2, color=c, lw=1.6)
            ax.plot(o["or"], yi + off, "o", color=c, ms=4, label=s.replace(" (the pre-registered analysis)", "") if n == names[0] else None)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6])
    ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("odds that a death within ten years was from cancer rather than another cause, per SD (left of 1: less cancer-specific)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Among people who died, the result holds in every subset")
    save("fig4_specificity.png")


def fig_claims():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    t5 = R["T5"]["indices"]
    for n, v in t5.items():
        ax.scatter(v["or"], v["auc_gain_later_cycles"], color=BLUE, s=48, zorder=3)
        ax.text(v["or"] + 0.006, v["auc_gain_later_cycles"] + 0.0002, n, fontsize=8.5)
    ax.axvline(1, color=INK, lw=1)
    ax.axhline(0.01, color=MUTED, lw=1, ls="--")
    ax.text(0.83, 0.0102, "bar: +0.01", fontsize=8, color=MUTED)
    ax.set_xlabel("odds ratio per SD for self-reported cancer history (what most papers report)")
    ax.set_ylabel("AUC gain over age, sex, smoking, BMI\n(later survey cycles)")
    ax.set_ylim(-0.002, 0.012)
    ax.set_title("The claims reproduce as associations, and add almost nothing as prediction")
    save("fig5_claims.png")


def fig_replication():
    Z = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_replication_result.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), sharey=True)
    y = np.arange(len(NAMES))[::-1]
    for ax, key, title in ((axes[0], "T1", "Added discrimination for cancer death (AUC gain)"), (axes[1], "OR", "Association with each kind of death (odds ratio per SD)")):
        for yi, n in zip(y, NAMES):
            a, b = R["indices"][n], Z["indices"][n]
            if key == "T1":
                ax.plot(a["T1_ci"], [yi + 0.15] * 2, color=BLUE, lw=2)
                ax.plot(a["T1_gain"], yi + 0.15, "o", color=BLUE, ms=5, label="NHANES 1999-2008 (test cycles)" if n == NAMES[0] else None)
                ax.plot(b["T1_ci"], [yi - 0.15] * 2, color=TEAL, lw=2)
                ax.plot(b["T1_gain"], yi - 0.15, "s", color=TEAL, ms=5, label="NHANES III 1988-94 (external)" if n == NAMES[0] else None)
            else:
                ax.plot(b["or_cancer_ci"], [yi + 0.15] * 2, color=RED, lw=2)
                ax.plot(b["or_cancer_death"], yi + 0.15, "o", color=RED, ms=5, label="death from cancer" if n == NAMES[0] else None)
                ax.plot(b["or_noncancer_ci"], [yi - 0.15] * 2, color=ORANGE, lw=2)
                ax.plot(b["or_noncancer_death"], yi - 0.15, "s", color=ORANGE, ms=5, label="death from another cause" if n == NAMES[0] else None)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(0, color=INK, lw=1)
    axes[0].axvline(0.01, color=MUTED, lw=1, ls="--")
    axes[1].axvline(1, color=INK, lw=1)
    axes[1].set_xscale("log")
    from matplotlib.ticker import NullFormatter
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks([0.7, 0.85, 1, 1.2, 1.4])
    axes[1].set_xticklabels(["0.7", "0.85", "1", "1.2", "1.4"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(NAMES)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Independent replication in NHANES III: same picture", x=0.06, ha="left", y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig6_replication.png")


def fig_decision():
    Z = json.load(open(os.path.join(ROOT, "experiments", "cheap_markers_replication_result.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    for ax, key, title in ((axes[0], "test_cycles_2005_2008", "NHANES 2005-2008, adults 60+"), (axes[1], "nhanes3", "NHANES III, adults 60+")):
        rules = Z["decision"][key]["rules"]
        names = list(rules)
        short = [n.replace("NLR-analogue at the cutoff that flags the same share as NLR >= 3", "NLR >= 3 (matched)").replace("top fifth of ", "top fifth: ") for n in names]
        x = np.arange(len(names))
        w = 0.36
        ax.bar(x - w / 2, [100 * rules[n]["age_only"]["sensitivity"] for n in names], w, color=GREY, label="flag the oldest, same number")
        ax.bar(x + w / 2, [100 * rules[n]["marker"]["sensitivity"] for n in names], w, color=BLUE, label="flag by the marker")
        for xi, n in zip(x, names):
            lo, hi = rules[n]["sens_diff_ci"]
            ax.text(xi, 2, f"diff {100 * rules[n]['sens_diff_marker_minus_age']:+.1f}\n[{100 * lo:+.0f}, {100 * hi:+.0f}]", ha="center", fontsize=7.2, color="white" if False else INK)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7.8, rotation=12)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 42)
    axes[0].set_ylabel("% of cancer deaths found")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("A marker rule finds no more cancer deaths than flagging the oldest, except possibly RDW", x=0.06, ha="left", y=1.03, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig7_decision.png")


if __name__ == "__main__":
    for f in (fig_logic, fig_gain, fig_frailty, fig_specific, fig_claims, fig_replication, fig_decision):
        f()
