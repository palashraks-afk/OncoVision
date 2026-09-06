"""
Does triaging on free bloodwork save money against sending everyone?

The question this project is actually for
-----------------------------------------
Oncovision exists to lower the cost around expensive diagnosis. Every number
this project has produced so far is an AUC or a count of people flagged per
case. Neither answers the question the tool was built to answer, which is
whether reading bloodwork somebody has already paid for lets a health system
spend less to find the same cancers.

That is a different question from "is the model accurate", and a model can win
one and lose the other. This measures it.

The comparison
--------------
The marginal cost of running a panel is zero. The blood has been drawn, the
analyser has run, the report exists. So the comparison is not "test versus no
test", it is:

    UNIVERSAL   send everyone eligible for the confirmatory procedure
    TRIAGED     send only the people the panel flags

Per 100,000 eligible people at real incidence:

    universal cost = 100,000 x procedure
    triaged cost   = (TP + FP) x procedure  +  FN x (late-stage - early-stage)

The last term is the one that matters and the one a naive cost argument leaves
out. Triage saves procedures and buys missed cancers, and a missed cancer is
found later at a worse stage and a higher price. A panel only saves money if the
procedures it avoids are worth more than the cancers it misses.

Cost inputs
-----------
US dollars, from published sources, listed here so they can be argued with:

    colonoscopy             $2,412   national average, range $1,856-$4,616
    low-dose chest CT         $300   typical self-pay screening price
    prostate MRI            $1,200   typical self-pay price
    colorectal treatment  $49,000 early / $116,000 advanced, first year
    lung treatment        $60,000 early / $120,000 advanced, first year

The treatment figures are first-year costs and therefore understate the true
difference, which continues for years. Understating it makes triage look BETTER
than it is, so the bias runs against the conclusion this file reaches.

This is an illustrative model, not a formal cost-effectiveness analysis. It has
no discounting, no quality-adjusted life years, and no account of the harm of an
unnecessary procedure beyond its price. Every input is swept in a sensitivity
analysis because the conclusion should not rest on any single one.

Run:  python experiments/cost_model.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "experiments/cost_model_result.json"
COHORT = 100_000

# panel -> (confirmatory procedure, its cost, early treatment, late treatment,
#           incidence in the eligible group, source note)
SETTINGS = {
    "colorectal": {
        "procedure": "colonoscopy",
        "procedure_cost": 2412.0,
        "early_cost": 49_000.0,
        "late_cost": 116_000.0,
        "incidence": 0.0040,   # the cohort's own 8-year rate, 96 in 23,794
        "life_years_lost": 15,
        "note": "incidence is this cohort's 8-year rate, not annual SEER",
    },
    "lung": {
        "procedure": "low-dose chest CT",
        "procedure_cost": 300.0,
        "early_cost": 60_000.0,
        "late_cost": 120_000.0,
        "incidence": 0.0047,   # 104 in 21,916, adults with tobacco exposure
        "life_years_lost": 15,
        "note": "eligible group is adults with measurable tobacco exposure",
    },
    "liver": {
        # Not a cancer panel. It detects clinical liver disease, and the
        # confirmatory step is transient elastography rather than a biopsy or a
        # scope. Included because it is the panel with the largest gain over age
        # and sex in the whole project, and leaving it out of the cost analysis
        # because it is awkward would be the wrong reason.
        "procedure": "FibroScan (transient elastography)",
        "procedure_cost": 500.0,      # published range $200 to $1,500
        "early_cost": 5_280.0,        # compensated cirrhosis, per year
        "late_cost": 20_651.0,        # decompensated cirrhosis, per year
        "incidence": 0.0404,          # 1,436 in 35,511
        # Missing liver disease is not the same loss as missing a cancer. Late
        # cirrhosis costs years, not usually fifteen of them, and the figure is
        # set low deliberately: if the panel pays even on a conservative
        # valuation, the conclusion is robust.
        "life_years_lost": 5,
        "note": "endpoint is clinical liver disease, not liver cancer",
    },
}

# Panels with no confirmatory procedure to triage FOR, which is a finding rather
# than an omission.
#
# The general panel predicts a diagnosis of ANY cancer within four years. There
# is no single test that confirms or excludes that, so there is no procedure for
# it to save and no cost model to build. A positive result routes to nothing in
# particular.
#
# That is worth stating plainly: the panel's problem is not only that it adds
# 0.006 over age and sex, nor only that the gain fails to survive external
# validation. It is that even a good version of it would have nowhere to send
# the person it flagged.
NO_CONFIRMATORY_TEST = {
    "general": ("predicts a diagnosis of any cancer within four years, and no "
                "single procedure confirms or excludes that, so there is no test "
                "for it to save"),
    "breast": ("an interpretation panel: the biopsy it reads has already been "
               "performed, so there is no downstream procedure to triage"),
    "prostate": ("an interpretation panel over an MRI that has already been "
                 "done"),
    "ovarian": ("a triage panel for a mass already found, and the operative "
                "decision it informs is not a screening procedure"),
    "pancreatic": ("a triage panel run on someone already under investigation"),
}


def roc_for(panel):
    """
    The panel's real ROC curve, from out-of-fold predictions on its own cohort.

    Necessary because the operating-point search below has to know the actual
    sensitivity-specificity trade this model offers. A curve inferred from the
    shipped point and a guessed shape gave sensitivity 1.0 at specificity 0.0,
    which is not a trade-off, it is an artefact.
    """
    import train_models as tm
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cfg = next(c for c in tm.DATASETS if c["name"] == panel)
    X, y, _ = tm.prepare(cfg)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = pd.Series(y).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(
        CalibratedClassifierCV(tm.build_ensemble(len(y), float(y.mean())),
                               method="isotonic", cv=cv),
        X, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def outcomes(n, prevalence, sens, spec):
    cases = n * prevalence
    tp = cases * sens
    fn = cases - tp
    fp = (n - cases) * (1 - spec)
    return tp, fp, fn


def evaluate(cfg, sens, spec, n=COHORT):
    tp, fp, fn = outcomes(n, cfg["incidence"], sens, spec)
    universal = n * cfg["procedure_cost"]
    triaged = ((tp + fp) * cfg["procedure_cost"]
               + fn * (cfg["late_cost"] - cfg["early_cost"]))
    return {
        "procedures_universal": int(n),
        "procedures_triaged": int(round(tp + fp)),
        "cancers_found": round(tp, 1),
        "cancers_missed": round(fn, 1),
        "cost_universal": round(universal),
        "cost_triaged": round(triaged),
        "saving": round(universal - triaged),
        "saving_per_person": round((universal - triaged) / n, 2),
        "cost_per_cancer_missed": (round((universal - triaged) / fn) if fn > 0 else None),
    }


def main():
    held = json.load(open("evaluation.json", encoding="utf-8"))
    results = {}

    for panel, cfg in SETTINGS.items():
        c = held[panel]["calibrated"]
        sens, spec = c["sensitivity"], c["specificity"]
        base = evaluate(cfg, sens, spec)
        print(f"=== {panel} ===  confirmatory procedure: {cfg['procedure']} "
              f"at ${cfg['procedure_cost']:,.0f}")
        print(f"  panel operating point: sensitivity {sens}, specificity {spec}")
        print(f"  per {COHORT:,} eligible people at {cfg['incidence']:.2%} incidence\n")
        print(f"    send everyone      {base['procedures_universal']:>7,} procedures   "
              f"${base['cost_universal']:>13,}")
        print(f"    send only flagged  {base['procedures_triaged']:>7,} procedures   "
              f"${base['cost_triaged']:>13,}")
        print(f"    finds {base['cancers_found']} of "
              f"{base['cancers_found'] + base['cancers_missed']:.0f} cases, "
              f"misses {base['cancers_missed']}")
        verdict = ("SAVES" if base["saving"] > 0 else "COSTS MORE")
        print(f"\n    -> triage {verdict} ${abs(base['saving']):,} per {COHORT:,} people"
              f"  (${abs(base['saving_per_person'])}/person)")
        if base["saving"] > 0 and base["cancers_missed"] > 0:
            print(f"       and buys that saving with {base['cancers_missed']} missed "
                  f"cases, at ${base['cost_per_cancer_missed']:,} saved per case missed")

        # Sensitivity analysis. The conclusion must not rest on one input.
        sweeps = {}
        for label, key, factors in [
            ("procedure cost", "procedure_cost", [0.5, 0.75, 1.0, 1.5, 2.0]),
            ("late-stage penalty", "late_cost", [0.75, 1.0, 1.5, 2.0, 3.0]),
            ("incidence", "incidence", [0.5, 1.0, 2.0, 5.0, 10.0]),
        ]:
            row = {}
            for f in factors:
                alt = dict(cfg)
                if key == "late_cost":
                    alt[key] = cfg["early_cost"] + (cfg["late_cost"] - cfg["early_cost"]) * f
                else:
                    alt[key] = cfg[key] * f
                row[f"x{f}"] = evaluate(alt, sens, spec)["saving"]
            sweeps[label] = row
            signs = {"saves" if v > 0 else "costs" for v in row.values()}
            print(f"       {label:<20} " +
                  "  ".join(f"x{f}: ${row[f'x{f}']:>+12,}" for f in factors))
            if len(signs) > 1:
                print(f"          ^ the sign FLIPS across this range, so the answer "
                      f"depends on this input")

        results[panel] = {"settings": cfg, "sensitivity": sens, "specificity": spec,
                          "base_case": base, "sweeps": sweeps}
        print()

    # The break-even, which is the number that decides whether any of this is a
    # real argument.
    #
    # The model above prices a missed cancer at the extra treatment it costs
    # when found late. That is the only part of the loss denominated in dollars,
    # and it is not the part that matters. A missed colorectal cancer at 65
    # costs on the order of fifteen life-years, and the willingness-to-pay
    # threshold used in US health economics is roughly $100,000 to $150,000 per
    # quality-adjusted life year.
    #
    # So the honest way to present this is not "triage saves money". It is:
    # triage saves money only if you value a missed cancer BELOW the figure
    # printed here. Everything hinges on that comparison.
    WTP_PER_QALY = 150_000

    print("=" * 84)
    print("  BREAK-EVEN: what a missed case has to be worth before triage stops paying")
    print()
    print(f"  {'panel':<12}{'break-even per missed case':>30}"
          f"{'a case, valued':>18}{'':>6}verdict")
    for panel, r in results.items():
        b = r["base_case"]
        if b["cancers_missed"] <= 0:
            continue
        cfg = r["settings"]
        societal = WTP_PER_QALY * cfg["life_years_lost"]
        breakeven = (b["saving"] / b["cancers_missed"]) + (cfg["late_cost"] - cfg["early_cost"])
        r["break_even_per_missed_cancer"] = round(breakeven)
        r["societal_cost_of_a_missed_cancer"] = societal
        r["saves_money_once_a_life_is_priced"] = bool(breakeven > societal)
        verdict = ("still saves" if breakeven > societal
                   else "NO LONGER SAVES")
        print(f"  {panel:<12}{'$' + format(round(breakeven), ','):>30}"
              f"{'$' + format(societal, ','):>18}{'':>6}{verdict}")

    print()
    print(f"  A missed case is priced above at that panel's own life-years lost x "
          f"${WTP_PER_QALY:,} per QALY, which is the")
    print("  conventional US willingness-to-pay threshold. Where the break-even sits "
          "below that,")
    print("  the saving is an artefact of counting only treatment dollars and "
          "ignoring the person.")

    # If the shipped operating point does not pay once a life is priced, the
    # useful question is whether ANY operating point does. Sweeping the
    # threshold trades specificity for sensitivity: fewer procedures avoided,
    # fewer cancers missed. Somewhere on that curve is the point that maximises
    # net benefit, and it is not where these panels currently sit.
    print()
    print("=" * 84)
    print("  Is there an operating point that pays once a missed case is priced "
          "at a life?")
    print()
    print(f"  {'panel':<12}{'best sens':>11}{'spec':>8}{'procedures':>13}"
          f"{'missed':>9}{'net benefit':>16}")
    for panel, r in results.items():
        cfg = r["settings"]
        societal = WTP_PER_QALY * cfg["life_years_lost"]
        fpr, tpr = roc_for(panel)
        best = None
        for sens, spec in zip(tpr, 1.0 - fpr):
            tp, fp, fn = outcomes(COHORT, cfg["incidence"], float(sens), float(spec))
            avoided = (COHORT - (tp + fp)) * cfg["procedure_cost"]
            net = avoided - fn * societal
            if best is None or net > best["net"]:
                best = {"sens": float(sens), "spec": float(spec),
                        "procs": tp + fp, "missed": fn, "net": net}
        r["best_operating_point"] = {
            "sensitivity": round(best["sens"], 3),
            "specificity": round(best["spec"], 3),
            "procedures_per_100k": int(round(best["procs"])),
            "cancers_missed": round(best["missed"], 1),
            "net_benefit": round(best["net"]),
            "pays_once_a_life_is_priced": bool(best["net"] > 0),
            "shipped_sensitivity": r["sensitivity"],
        }
        print(f"  {panel:<12}{best['sens']:>11.3f}{best['spec']:>8.3f}"
              f"{int(round(best['procs'])):>13,}{best['missed']:>9.1f}"
              f"{'$' + format(round(best['net']), ','):>16}")

    print()
    print("  Sensitivity and specificity are read off each panel's real ROC curve,")
    print("  computed from out-of-fold predictions on its own cohort. An earlier")
    print("  version inferred the curve from the shipped point with a guessed shape;")
    print("  it produced sensitivity 1.0 at specificity 0.0 and was discarded.")

    print()
    print("=" * 84)
    print("  Panels with no confirmatory procedure to triage for:\n")
    for panel, why in NO_CONFIRMATORY_TEST.items():
        print(f"    {panel:<12} {why}")
    print("\n  For these the cost question is not unanswered, it is malformed. The "
          "general panel")
    print("  is the one that matters: even a good version of it would have nowhere "
          "to send")
    print("  the person it flagged, which is a limit no amount of accuracy repairs.")

    print()
    savers = [p for p, r in results.items() if r["base_case"]["saving"] > 0]
    print(f"  panels where triaging on free bloodwork beats sending everyone: "
          f"{savers or 'none'}")
    print("  this is an illustrative model, not a cost-effectiveness analysis: no "
          "discounting,")
    print("  no quality-adjusted life years, and no price on the harm of an "
          "unnecessary procedure")

    with open(OUT, "w") as f:
        json.dump({"cohort_size": COHORT, "panels": results,
                   "panels_that_save": savers,
                   "no_confirmatory_test": NO_CONFIRMATORY_TEST}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
