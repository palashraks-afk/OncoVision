"""
Every figure in the navigator paper, drawn from the saved result files so a number on a
figure cannot differ from a number in the tables.

Run:  python experiments/make_paper_figures.py      writes docs/paper/figures/*.png
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
FIG = os.path.join(ROOT, "docs", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
EV = json.load(open(os.path.join(ROOT, "experiments", "navigator_evidence_result.json")))
RD = json.load(open(os.path.join(ROOT, "experiments", "navigator_readability_result.json")))

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, ORANGE, TEAL, RED, GOLD = "#2f6fb0", "#d9822b", "#2a9d8f", "#c0392b", "#b8962e"
ERA_COL = {"1988-1994 (NHANES III)": "#9aa5b1", "1999-2006": BLUE, "2007-2014": TEAL, "2015-2018": ORANGE}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def save(name):
    plt.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close()
    print("  wrote", name)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=color, ec="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="none", ec=color, lw=1.4))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 2.5, body, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.45)

    box(1, 24, 23, 17, "What the person gives", "age, sex, smoking\nsymptoms ticked\nlab values from a report\n(blood count, ferritin, CA-125)", BLUE)
    box(31, 24, 27, 17, "Rule engine", "~34 rules from NICE NG12\nthree-valued logic:\nyes, no, UNKNOWN\nmissing facts are asked for", TEAL)
    box(65, 24, 34, 17, "What comes back", "plain-language action and timeframe\nwhy it was flagged, with the page\nwhat could change the answer\nsafety text, never \"you have cancer\"", ORANGE)
    box(31, 2, 27, 17, "Context layer", "how common cancer death is\nfor this age, sex, smoking\n(calibrated on a later era)", GOLD)
    box(65, 2, 34, 17, "Honest limits shown", "rules not clinician-reviewed\n2015 UK guidance\nresearch prototype, not advice", RED)
    for (x0, y0, x1, y1) in ((24.6, 32.5, 30.4, 32.5), (58.6, 32.5, 64.4, 32.5), (44.5, 23.4, 44.5, 19.6), (58.6, 10.5, 64.4, 10.5)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.set_title("Figure 1. How the navigator works", loc="left", pad=6)
    save("fig1_pipeline.png")


def fig_cohort():
    d = pd.read_csv(os.path.join(ROOT, "data", "nhanes_navigator.csv.gz"))
    d = d[(d["age"] >= 40)].dropna(subset=["hemoglobin", "platelets"])
    n3 = pd.read_csv(os.path.join(ROOT, "data", "nhanes3_mortality_full.csv"))
    n3 = n3[n3["age"] >= 40].dropna(subset=["hemoglobin", "platelets"])
    g = d.groupby("cycle").agg(n=("SEQN", "size"), deaths=("died", lambda s: int((s == 1).sum())),
                               ferritin=("ferritin", lambda s: int(s.notna().sum())))
    rows = [("1988-94\nNHANES III", len(n3), int((n3["died"] == 1).sum()), 0)] + \
           [(c.replace("-", "-\n"), int(r.n), int(r.deaths), int(r.ferritin)) for c, r in g.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color=BLUE, alpha=0.85, label="adults 40+ with a blood count")
    ax.bar(x, [r[3] for r in rows], color=ORANGE, width=0.5, label="of whom ferritin was measured")
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 60, f"{r[1]:,}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.axvspan(8.5, 10.5, color=GRID, alpha=0.6, zorder=0)
    ax.text(9.5, max(r[1] for r in rows) * 0.95, "too little follow-up\nfor an outcome:\nburden only", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_ylabel("people")
    ax.set_title("Figure 2. The data: eleven survey cycles across three decades")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 0.97), ncol=1, fontsize=8.5)
    save("fig2_cohort.png")


def fig_burden():
    bands = ["40-49", "50-59", "60-69", "70-79", "80-+"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True)
    for ax, key, title in zip(axes, ("talk_soon", "any_alert"), ("Told to see a doctor soon", "Any alert (soon, raise or mention)")):
        w = 0.2
        for i, (era, col) in enumerate(ERA_COL.items()):
            vals = [100 * EV["q1"]["bands"][era][b][key]["share"] for b in bands]
            lo = [100 * EV["q1"]["bands"][era][b][key]["ci"][0] for b in bands]
            hi = [100 * EV["q1"]["bands"][era][b][key]["ci"][1] for b in bands]
            x = np.arange(len(bands)) + (i - 1.5) * w
            ax.bar(x, vals, w, color=col, label=era)
            ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, lw=0.7, capsize=1.5)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.replace("-+", "+") for b in bands])
        ax.set_xlabel("age")
        ax.set_title(title)
    axes[0].set_ylabel("% of adults, lab results alone")
    axes[0].axhline(3, color=RED, ls="--", lw=1)
    axes[0].text(4.45, 3.1, "bar: 3%", color=RED, fontsize=8, ha="right", va="bottom")
    axes[1].axhline(20, color=RED, ls="--", lw=1)
    axes[1].text(4.45, 20.4, "bar: 20%", color=RED, fontsize=8, va="bottom", ha="right")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.97))
    fig.suptitle("Figure 3. How many symptom-free adults the lab rules alert, by age and era", x=0.06, ha="left", y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig3_burden.png")


def fig_rules():
    q5 = {k: v for k, v in EV["q5"].items() if not k.startswith("_")}
    names = {"colorectal_stool_test": "Stool test (bowel)", "colorectal_urgent": "Urgent bowel referral",
             "lung_xray_symptoms": "Chest X-ray (lung)", "lung_xray_consider": "Consider chest X-ray",
             "ovarian_other": "Consider ovarian tests", "upper_gi_consider": "Consider endoscopy",
             "weight_loss": "Unexplained weight loss"}
    items = sorted(q5.items(), key=lambda kv: kv[1]["n"])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh([names.get(k, k) for k, _ in items], [v["n"] for _, v in items], color=TEAL)
    for i, (k, v) in enumerate(items):
        ax.text(v["n"] + 60, i, f"{v['n']:,}  (5-yr cancer death {100 * v['rate']:.1f}%, mean age {v['mean_age']:.0f})", va="center", fontsize=8)
    ax.set_xlim(0, max(v["n"] for _, v in items) * 1.65)
    ax.set_xlabel("people in whom the rule fired (labs + proxy symptoms)")
    ax.set_title("Figure 4. Which rules fire. Descriptive only: age differs between groups")
    save("fig4_rules.png")


def fig_forest():
    rows = []
    for name, r in EV["q7"]["results"].items():
        rows.append((name, r["or"], r["ci"][0], r["ci"][1], r["decedents_with_flag"]))
    order = ["ever smoked", "high platelets (> 400)", "raised white count (> 11)", "small red cells (MCV < 80)",
             "chronic cough (proxy)", "breathless (proxy)", "weight loss (proxy)", "anaemia", "any alert, labs only"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (name, o, lo, hi, n) in zip(y, rows):
        col = RED if hi < 1 else (TEAL if lo > 1 else MUTED)
        ax.plot([lo, hi], [yi, yi], color=col, lw=2)
        ax.plot(o, yi, "o", color=col, ms=6)
        ax.text(2.05, yi, f"n flagged {n:,}", va="center", fontsize=8, color=MUTED)
    ax.axvline(1, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3.2)
    ax.set_xticks([0.4, 0.6, 0.8, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.4", "0.6", "0.8", "1", "1.5", "2", "3"])
    ax.set_xlabel("odds that a death within 10 years was from cancer, flagged vs not\n(left of 1: more likely another cause)")
    ax.set_title("Figure 5. Among people who died, is a flag specific to cancer?")
    ax.grid(axis="y", visible=False)
    save("fig5_decedents.png")


def fig_auc():
    res = EV["q3"]["results"]
    labels = [("age+sex vs +labs alert", "Labs alert\n(1999-2006 fit,\n2007-2014 test)"),
              ("age+sex vs +labs and proxies alert", "Labs + proxy symptoms\n(1999-2006 fit,\n2007-2014 test)"),
              ("trained 1999-2014, tested NHANES III (labs alert)", "Labs alert\n(1999-2014 fit,\nNHANES III test)")]
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    w = 0.34
    for i, (k, lab) in enumerate(labels):
        r = res[k]
        ax.bar(i - w / 2, r["auc_base"], w, color="#9aa5b1", label="age + sex only" if i == 0 else None)
        ax.bar(i + w / 2, r["auc_with_alert"], w, color=BLUE, label="age + sex + navigator alert" if i == 0 else None)
        ax.text(i, 0.835, f"gain {r['gain']:+.3f}\n95% CI {r['gain_ci'][0]:+.3f} to {r['gain_ci'][1]:+.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.6, 0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l for _, l in labels], fontsize=8.5)
    ax.set_ylabel("AUC, five-year cancer death")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    ax.set_title("Figure 6. Does an alert add anything to age and sex?")
    save("fig6_auc.png")


def fig_calibration():
    q6 = EV["q6"]
    pred = np.array([d["predicted"] for d in q6["deciles"]]) * 1000
    obs = np.array([d["observed"] for d in q6["deciles"]]) * 1000
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    m = max(pred.max(), obs.max()) * 1.1
    ax.plot([0, m], [0, m], color=MUTED, ls="--", lw=1, label="perfect")
    ax.plot(pred, obs, "o-", color=BLUE, lw=1.6, label="ten groups of people")
    ax.set_xlabel("predicted, per 1,000 people (fitted on 1999-2006)")
    ax.set_ylabel("observed, per 1,000 people (2007-2014)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(m * 0.98, m * 0.04, f"calibration slope {q6['calibration_slope']}\nAUC {q6['auc']}", ha="right", fontsize=9)
    ax.set_title("Figure 7. The age context layer on a later era")
    save("fig7_calibration.png")


def fig_ferritin():
    q4 = EV["q4"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    vals = [q4["proxy_sensitivity"] * 100, q4["proxy_ppv"] * 100]
    ax.bar(["Sensitivity\n(iron-deficient people\nthe proxy catches)", "PPV\n(proxy positives that\nare iron-deficient)"], vals, color=[BLUE, TEAL], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title(f"MCV < 80 as a stand-in for ferritin < 15\n({q4['true_ida_by_ferritin']} iron-deficient of {q4['anaemic']} anaemic)", fontsize=9.5)
    ax = axes[1]
    ax.bar(["Proxy only\n(no ferritin)", "Measured ferritin"], [q4["age60_talk_soon_proxy_only"], q4["age60_talk_soon_with_ferritin"]],
           color=[ORANGE, TEAL], width=0.5)
    for i, v in enumerate([q4["age60_talk_soon_proxy_only"], q4["age60_talk_soon_with_ferritin"]]):
        ax.text(i, v + 0.8, str(v), ha="center", fontsize=10)
    ax.set_ylabel(f"people told 'see a doctor soon'\n(of {q4['age60_n']:,} aged 60+ with ferritin)")
    ax.set_title("What the assumption costs", fontsize=9.5)
    fig.suptitle("Figure 8. Testing the tool's own disclosed assumption", x=0.06, ha="left", y=1.03, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save("fig8_ferritin.png")


def fig_readability():
    kinds = RD["kinds"]
    order = ["safety", "headline", "action", "disclaimer", "symptom"]
    order = [k for k in order if k in kinds]
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    means = [kinds[k]["mean"] for k in order]
    ax.barh(order[::-1], means[::-1], color=BLUE)
    for i, v in enumerate(means[::-1]):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=9)
    ax.axvline(8, color=RED, ls="--", lw=1)
    ax.text(8.1, len(order) - 0.55, "bar: mean 8", color=RED, fontsize=8)
    ax.set_xlabel("Flesch-Kincaid US grade level (lower is easier)")
    ax.set_title("Figure 9. Reading level of everything the tool can say")
    save("fig9_readability.png")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_cohort, fig_burden, fig_rules, fig_forest, fig_auc, fig_calibration, fig_ferritin, fig_readability):
        f()
