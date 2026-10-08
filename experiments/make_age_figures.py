"""
Figures for the cancer-risk-age / cost-matched screening paper, from the saved result files
(and a quick refit for the per-person risk ages).

Run:  python experiments/make_age_figures.py     writes docs/cancer_age_paper/figures/*.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cancer_age import Fit, load_nhis, risk_age  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "cancer_age_paper", "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_result.json")))
X = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_extra_result.json")))
F4 = json.load(open(os.path.join(ROOT, "experiments", "cancer_age_race_F4.json")))
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, ORANGE, TEAL, RED, GREY = "#2f6fb0", "#d9822b", "#2a9d8f", "#c0392b", "#9aa5b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})


def save(name):
    plt.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close()
    print("  wrote", name)


def fig_design():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)

    def box(x, y, w, h, text, col, size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=col, ec="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc="none", ec=col, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.5)
    box(1, 12, 20, 14, "Four free answers\nage, sex, smoking, BMI\n(no test, no cost)", BLUE)
    box(29, 12, 22, 14, "Ten-year risk of\ndying of cancer\n(fitted 1997-2002)", TEAL)
    box(60, 21, 38, 12, "Policy A: invite the oldest people\nuntil the budget is used", GREY)
    box(60, 3, 38, 12, "Policy B: invite the highest-risk people\nuntil the same budget is used", ORANGE)
    for a, b in (((21.6, 19), (28.4, 19)), ((51.6, 19), (59.4, 26)), ((51.6, 19), (59.4, 10))):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(79, 0.3, "Same number invited. Which covers more cancer deaths? (tested on 2005-2009)", ha="center", fontsize=8.6, color=MUTED)
    ax.set_title("The question: can free information make screening invitations cheaper?")
    save("fig1_design.png")


def fig_ladder():
    m = R["models"]
    names = list(m)
    short = ["Age + sex", "+ smoking", "+ BMI", "+ schooling,\nmarital, health,\nactivity", "+ race and\nethnicity"]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(names))
    cols = [GREY, BLUE, TEAL, "#c9d3de", "#c9d3de"]
    ax.bar(x, [m[n]["auc"] for n in names], 0.6, color=cols)
    for xi, n in zip(x, names):
        ax.text(xi, m[n]["auc"] + 0.0015, f"{m[n]['auc']:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.74, 0.795)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.6)
    ax.set_ylabel("AUC, ten-year cancer death (2005-2009)")
    g = R["gains"]
    ax.annotate(f"+{g['F2 vs F0']['gain']:.3f}\n(95% CI {g['F2 vs F0']['ci'][0]:.3f} to {g['F2 vs F0']['ci'][1]:.3f})", xy=(1, 0.7875), xytext=(1.0, 0.7905), ha="center", fontsize=8.4, color=TEAL)
    ax.set_title("Almost everything the free questions know is in smoking; schooling and the rest add 0.001")
    save("fig2_ladder.png")


def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    dec = R["deciles"]
    pr, ob = np.array([d["predicted"] for d in dec]) * 1000, np.array([d["observed"] for d in dec]) * 1000
    m = max(pr.max(), ob.max()) * 1.1
    ax.plot([0, m], [0, m], color=MUTED, ls="--", lw=1)
    ax.plot(pr, ob, "o-", color=BLUE)
    ax.set_xlabel("predicted, per 1,000 people (model fitted 1997-2002)")
    ax.set_ylabel("observed, per 1,000 people (2005-2009)")
    ax.text(m * 0.98, m * 0.04, f"slope {R['bar1']['slope']}\nintercept {R['bar1']['intercept']}", ha="right", fontsize=9)
    ax.set_title("Calibration in ten equal groups")
    ax = axes[1]
    bands = list(R["bar2"]["bands"])
    x = np.arange(len(bands))
    w = 0.26
    vals = {"NHIS 2005-2009": [R["bar2"]["bands"][b]["ratio"] for b in bands],
            "NHANES 1999-2008": [R["external"]["NHANES 1999-2008"]["bands"][b]["ratio"] for b in bands],
            "NHANES III 1988-94": [R["external"]["NHANES III 1988-1994"]["bands"][b]["ratio"] for b in bands]}
    for i, (k, col) in enumerate(zip(vals, (BLUE, TEAL, ORANGE))):
        ax.bar(x + (i - 1) * w, vals[k], w, color=col, label=k)
    ax.axhline(1, color=INK, lw=1)
    ax.axhspan(0.8, 1.2, color=GRID, alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylim(0.6, 1.4)
    ax.set_xlabel("age band")
    ax.set_ylabel("observed / predicted cancer deaths")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("By age band: the 50-59 gap is closed")
    save("fig3_calibration.png")


def fig_riskage():
    d = load_nhis()
    tr, te = d[d["year"] <= 2002].reset_index(drop=True), d[d["year"] >= 2005].reset_index(drop=True)
    f0, f2 = Fit(tr, 0), Fit(tr, 2)
    ages = np.arange(35, 96, 0.5)
    r0 = {sx: f0.predict(pd.DataFrame({"age": ages, "male": sx})) for sx in (0.0, 1.0)}
    p = f2.predict(te)
    te["rap"] = risk_age(p, te["male"].to_numpy(), ages, r0) - te["age"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    groups = [("never smoked", te["smoke_status"] == 0, GREY), ("former smoker", te["smoke_status"] == 1, BLUE), ("current smoker", te["smoke_status"] == 2, RED)]
    bp = ax.boxplot([te["rap"][m] for _, m, _ in groups], vert=True, patch_artist=True, showfliers=False, widths=0.55)
    for patch, (_, _, c) in zip(bp["boxes"], groups):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_xticklabels([g[0] for g in groups])
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("cancer-risk age minus actual age (years)")
    ax.set_title("A current smoker carries about nine extra years")
    ax = axes[1]
    ba = pd.cut(te["age"], [35, 45, 55, 65, 75, 85], right=False)
    for lab, m, c in groups:
        s = te[m].groupby(pd.cut(te["age"][m], [35, 45, 55, 65, 75, 85], right=False), observed=True)["rap"].mean()
        ax.plot([f"{int(i.left)}-{int(i.right) - 1}" for i in s.index], s.values, "o-", color=c, label=lab)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("age")
    ax.set_ylabel("mean risk advancement (years)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("The advance is largest in midlife")
    save("fig4_riskage.png")


def fig_coverage():
    c = R["cost_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    s = np.array(c["shares"]) * 100
    ax.plot(s, np.array(c["coverage_age"]) * 100, "o-", color=GREY, label="invite the oldest")
    ax.plot(s, np.array(c["coverage_risk"]) * 100, "o-", color=ORANGE, label="invite the highest risk")
    lo = np.array(c["coverage_risk"]) - np.array(c["diff"]) + np.array(c["diff_lo"])
    hi = np.array(c["coverage_risk"]) - np.array(c["diff"]) + np.array(c["diff_hi"])
    ax.fill_between(s, lo * 100, hi * 100, color=ORANGE, alpha=0.18)
    ax.plot([0, 100], [0, 100], color=MUTED, ls=":", lw=1)
    ax.set_xlabel("% of adults 40-74 invited")
    ax.set_ylabel("% of ten-year cancer deaths among those invited")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("More cancer deaths covered")
    ax = axes[1]
    fc = c["fixed_coverage"]
    tg = [0.4, 0.5, 0.6]
    x = np.arange(3)
    ax.bar(x - 0.18, [fc[str(t)]["share_invited_age"] * 100 for t in tg], 0.36, color=GREY, label="invite the oldest")
    ax.bar(x + 0.18, [fc[str(t)]["share_invited_risk"] * 100 for t in tg], 0.36, color=ORANGE, label="invite the highest risk")
    for xi, t in zip(x, tg):
        ax.text(xi, fc[str(t)]["share_invited_age"] * 100 + 1.2, f"{fc[str(t)]['fewer_invitations']:.0%}\nfewer", ha="center", fontsize=8.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cover {int(t * 100)}% of\ncancer deaths" for t in tg])
    ax.set_ylabel("% of adults 40-74 who must be invited")
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Or fewer invitations")
    save("fig5_coverage.png")


def fig_external():
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    sets = [("NHIS 2005-2009\n(temporal test)", R["cost_matched"]["shares"], R["cost_matched"]["diff"], R["cost_matched"]["diff_lo"], R["cost_matched"]["diff_hi"], BLUE),
            ("NHANES 1999-2008", R["external"]["NHANES 1999-2008"]["cost_matched"]["shares"], None, R["external"]["NHANES 1999-2008"]["cost_matched"]["diff_lo"], R["external"]["NHANES 1999-2008"]["cost_matched"]["diff_hi"], TEAL),
            ("NHANES III 1988-94", R["external"]["NHANES III 1988-1994"]["cost_matched"]["shares"], None, R["external"]["NHANES III 1988-1994"]["cost_matched"]["diff_lo"], R["external"]["NHANES III 1988-1994"]["cost_matched"]["diff_hi"], ORANGE)]
    for k, (lab, shares, diff, lo, hi, col) in enumerate(sets):
        cm = R["cost_matched"] if k == 0 else R["external"][["", "NHANES 1999-2008", "NHANES III 1988-1994"][k]]["cost_matched"]
        if k == 0:
            idx = [i for i, s in enumerate(shares) if round(s * 100) % 10 == 0 and s <= 0.6]
            sh = [shares[i] for i in idx]
            dd = [(cm["coverage_risk"][i] - cm["coverage_age"][i]) for i in idx]
            lo2, hi2 = [lo[i] for i in idx], [hi[i] for i in idx]
        else:
            sh = shares
            dd = [cm["coverage_risk"][i] - cm["coverage_age"][i] for i in range(len(shares))]
            lo2, hi2 = lo, hi
        xs = np.array(sh) * 100 + (k - 1) * 1.2
        ax.errorbar(xs, np.array(dd) * 100, yerr=[(np.array(dd) - np.array(lo2)) * 100, (np.array(hi2) - np.array(dd)) * 100], fmt="o-", color=col, capsize=2, label=lab, lw=1.4)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("% of adults 40-74 invited")
    ax.set_ylabel("extra deaths covered (points)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The advantage points the same way in three national datasets")
    save("fig6_external.png")


def fig_equity():
    g = X["groups"]
    races = ["non-Hispanic White", "non-Hispanic Black", "Hispanic", "other"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1))
    ax = axes[0]
    x = np.arange(4)
    ax.bar(x - 0.18, [g[r]["auc_F0"] for r in races], 0.36, color=GREY, label="age + sex")
    ax.bar(x + 0.18, [g[r]["auc_F2"] for r in races], 0.36, color=TEAL, label="+ smoking, BMI")
    ax.set_xticks(x)
    ax.set_xticklabels(["White", "Black", "Hispanic", "Other"], fontsize=9)
    ax.set_ylim(0.70, 0.82)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Ranking power")
    ax = axes[1]
    ax.bar(x - 0.18, [g[r]["observed_over_expected"] for r in races], 0.36, color=TEAL, label="without race")
    ax.bar(x + 0.18, [F4[r]["OE_F4"] for r in races], 0.36, color=BLUE, label="with race")
    ax.axhline(1, color=INK, lw=1)
    ax.axhspan(0.8, 1.2, color=GRID, alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["White", "Black", "Hispanic", "Other"], fontsize=9)
    ax.set_ylim(0.6, 1.4)
    ax.set_ylabel("observed / predicted deaths")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Calibration")
    ax = axes[2]
    labs = ["White", "Black", "Hispanic", "Other", "< high\nschool", "High\nschool", "Some\ncollege", "College+"]
    keys = races + ["less than high school", "high school or GED", "some college", "college or more"]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        gg = g[k]["gain_at_20pct"]
        col = TEAL if gg["ci"][0] > 0 else GREY
        ax.plot([gg["ci"][0] * 100, gg["ci"][1] * 100], [yi, yi], color=col, lw=2)
        ax.plot(gg["diff"] * 100, yi, "o", color=col, ms=5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([l.replace("\n", " ") for l in labs], fontsize=8.6)
    ax.set_xlabel("extra deaths covered (points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Who the policy helps")
    fig.suptitle("The fairness bar was missed: the free-information policy helps some groups more than others", x=0.06, ha="left", y=1.03, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig7_equity.png")


def fig_forties():
    s = R["screening_start"]
    hi, al = s["composition_of_high_group"], s["composition_of_all_40_49"]
    keys = [("current smoker", "Current smoker"), ("BMI 30+", "BMI 30 or more"), ("less than high school", "Less than high school"), ("college+", "College or more"), ("male", "Men")]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = np.arange(len(keys))[::-1]
    ax.barh(y + 0.18, [hi[k] * 100 for k, _ in keys], 0.34, color=ORANGE, label=f"cancer-risk age 50 or more ({s['share_risk_age_50plus']:.0%} of 40-49-year-olds)")
    ax.barh(y - 0.18, [al[k] * 100 for k, _ in keys], 0.34, color=GREY, label="all adults aged 40-49")
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in keys])
    ax.set_xlabel("%")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Who is already at fifty-year-old risk at 40-49: smokers ({s['share_of_deaths_in_high_group']:.0%} of this age's cancer deaths)")
    save("fig8_forties.png")


if __name__ == "__main__":
    for f in (fig_design, fig_ladder, fig_calibration, fig_riskage, fig_coverage, fig_external, fig_equity, fig_forties):
        f()
