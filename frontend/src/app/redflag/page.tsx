"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";

// Reads three numbers already printed on an ordinary blood report and checks them against the
// American Gastroenterological Association 2020 guideline on iron-deficiency anaemia. Runs entirely in the
// browser; nothing is sent anywhere. The national figures come from experiments/red_flag_labs_result.json
// (NHANES 2017-March 2020 and NHANES III / 2001-2002 with deaths followed through 2019).

const NATIONAL = {
  millions: 2.1, millionsLo: 1.6, millionsHi: 2.6,
  untreated: 81, untreatedRegular: 77,
  perCancerDeathIda: 16, perCancerDeathNone: 41,
};

const field = "w-full bg-[var(--paper)] border border-[var(--rule-strong)] px-3 py-2 text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--stamp-line)]";

type Verdict = "ida" | "ida-young" | "anaemia-check-iron" | "anaemia-not-iron" | "low-iron-no-anaemia" | "normal";

function RefBar({ label, value, unit, lo, hi, cut, cutLabel, low }: { label: string; value: number; unit: string; lo: number; hi: number; cut: number; cutLabel: string; low: boolean }) {
  const pct = (v: number) => Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100));
  return (
    <div className="mb-5">
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="field-label">{label}</span>
        <span className="data text-sm">
          {value} <span className="text-[var(--ink-3)]">{unit}</span>{" "}
          {low && <span className="flag-stamp text-[var(--flag)] ml-1">L</span>}
        </span>
      </div>
      <div className="refbar">
        <div className="refbar-normal" style={{ left: `${pct(cut)}%`, width: `${100 - pct(cut)}%` }} />
        <div className="refbar-tick" data-flag={low ? "high" : undefined} style={{ left: `${pct(value)}%` }} />
      </div>
      <p className="refbar-caption text-[10px] mt-1 data">{cutLabel}</p>
    </div>
  );
}

export default function RedFlagPage() {
  const [sex, setSex] = useState("female");
  const [age, setAge] = useState("");
  const [periods, setPeriods] = useState("no");
  const [hgb, setHgb] = useState("");
  const [fer, setFer] = useState("");
  const [mcv, setMcv] = useState("");

  const out = useMemo(() => {
    const a = Number(age), h = Number(hgb);
    if (!age.trim() || !hgb.trim() || !Number.isFinite(a) || !Number.isFinite(h) || a < 18 || a > 110 || h < 3 || h > 25) return null;
    const male = sex === "male";
    const cutH = male ? 13 : 12;
    const anaemia = h < cutH;
    const f = fer.trim() ? Number(fer) : NaN;
    const m = mcv.trim() ? Number(mcv) : NaN;
    const hasF = Number.isFinite(f) && f >= 0;
    const lowIron = hasF && f < 45;
    const guidelineGroup = male || periods === "no";
    let v: Verdict = "normal";
    if (anaemia && lowIron) v = guidelineGroup ? "ida" : "ida-young";
    else if (anaemia && !hasF) v = "anaemia-check-iron";
    else if (anaemia) v = "anaemia-not-iron";
    else if (lowIron) v = "low-iron-no-anaemia";
    return { a, h, cutH, anaemia, f, hasF, lowIron, m, small: Number.isFinite(m) && m < 80, v, male };
  }, [sex, age, periods, hgb, fer, mcv]);

  const letter = out && (out.v === "ida" || out.v === "ida-young" || out.v === "anaemia-check-iron")
    ? [
        `My blood test shows haemoglobin ${out.h} g/dL${out.hasF ? ` and ferritin ${out.f} ng/mL` : ""}${Number.isFinite(out.m) ? `, MCV ${out.m} fL` : ""}.`,
        out.v === "anaemia-check-iron"
          ? "I am anaemic but my iron (ferritin) was not on the report. Could we check it?"
          : "By the American Gastroenterological Association 2020 guideline this is iron-deficiency anaemia (anaemia with ferritin under 45 ng/mL).",
        out.v === "ida" ? "The guideline recommends looking for a source of blood loss with an upper endoscopy and a colonoscopy, even without symptoms. Do I need these, and what is the reason I am iron deficient?" : "",
        out.v === "ida-young" ? "The guideline suggests considering an upper endoscopy and a colonoscopy for this too. What do you think is causing it?" : "",
        "Should I be tested for coeliac disease and H. pylori first, as the guideline suggests?",
      ].filter(Boolean).join("\n")
    : "";

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] font-sans">
      <main className="max-w-2xl mx-auto px-5 py-10 fade-in">
        <Link href="/" className="inline-flex items-center gap-2 text-xs text-[var(--stamp)] mb-8"><ArrowLeft className="w-4 h-4" /> Back to Oncovision</Link>

        <h1 className="display text-3xl leading-tight mb-3">Is there a red flag hiding on your blood test?</h1>
        <p className="text-sm text-[var(--ink-2)] leading-relaxed mb-3">
          A routine blood count shows your haemoglobin. If it is low and your ferritin (stored iron) is also low, US gut specialists say
          that is worth looking into even if you feel fine, because slow bleeding from the gut, sometimes from a cancer, is one cause.
          About <span className="data text-[var(--ink)]">{NATIONAL.millions}</span> million Americans over 50 have this result right now, and
          about <span className="data text-[var(--ink)]">{NATIONAL.untreated}%</span> are not being treated for it. Even among people who had
          seen a doctor at least twice in the past year, about <span className="data text-[var(--ink)]">{NATIONAL.untreatedRegular}%</span> were not.
        </p>
        <p className="text-xs text-[var(--ink-3)] leading-relaxed mb-8 border-l-2 border-[var(--warn-line)] pl-3">
          Research prototype, not medical advice, not reviewed by a clinician. Most people with this result do not have cancer. It reads three
          numbers against one published guideline. Nothing you type leaves this page.
        </p>

        <form className="border-y border-[var(--rule-strong)] py-6 mb-8" onSubmit={e => e.preventDefault()}>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div><label className="field-label block mb-1" htmlFor="sex">Sex at birth</label>
              <select id="sex" className={field} value={sex} onChange={e => setSex(e.target.value)}>
                <option value="female">Female</option><option value="male">Male</option></select></div>
            <div><label className="field-label block mb-1" htmlFor="age">Age</label>
              <input id="age" className={`${field} data`} inputMode="numeric" value={age} onChange={e => setAge(e.target.value)} placeholder="e.g. 58" /></div>
            {sex === "female" && (
              <div><label className="field-label block mb-1" htmlFor="periods">Still have periods</label>
                <select id="periods" className={field} value={periods} onChange={e => setPeriods(e.target.value)}>
                  <option value="no">No</option><option value="yes">Yes</option></select></div>)}
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div><label className="field-label block mb-1" htmlFor="hgb">Haemoglobin, g/dL</label>
              <input id="hgb" className={`${field} data`} inputMode="decimal" value={hgb} onChange={e => setHgb(e.target.value)} placeholder="HGB, e.g. 11.8" /></div>
            <div><label className="field-label block mb-1" htmlFor="fer">Ferritin, ng/mL</label>
              <input id="fer" className={`${field} data`} inputMode="decimal" value={fer} onChange={e => setFer(e.target.value)} placeholder="if on report" /></div>
            <div><label className="field-label block mb-1" htmlFor="mcv">MCV, fL</label>
              <input id="mcv" className={`${field} data`} inputMode="decimal" value={mcv} onChange={e => setMcv(e.target.value)} placeholder="optional" /></div>
          </div>
          <p className="text-xs text-[var(--ink-3)] mt-3">Haemoglobin (HGB) and MCV are on every complete blood count. Ferritin is a separate test and may not be on yours.</p>
        </form>

        {!out && (
          <p className="text-sm text-[var(--ink-2)] leading-relaxed">
            Enter your age and the haemoglobin from your blood report (it is usually the second or third line of a complete blood count, labelled HGB
            or Hb). Add ferritin if your report has it. The reading appears here as you type.
          </p>
        )}

        {out && (
          <div className="space-y-8">
            <section>
              <h2 className="display text-lg mb-4 rule-out">Your numbers against the guideline</h2>
              <RefBar label="Haemoglobin" value={out.h} unit="g/dL" lo={6} hi={18} cut={out.cutH} cutLabel={`anaemia below ${out.cutH} g/dL (${out.male ? "men" : "women"})`} low={out.anaemia} />
              {out.hasF && <RefBar label="Ferritin" value={out.f} unit="ng/mL" lo={0} hi={300} cut={45} cutLabel="iron deficiency, if anaemic, below 45 ng/mL (AGA 2020)" low={out.lowIron} />}
              {Number.isFinite(out.m) && <RefBar label="MCV (red cell size)" value={out.m} unit="fL" lo={60} hi={110} cut={80} cutLabel="small red cells below 80 fL, often a sign of low iron" low={out.small} />}
            </section>

            <section className={`border p-5 ${out.v === "ida" || out.v === "ida-young" ? "border-[var(--flag-line)] bg-[var(--flag-bg)]" : out.v === "normal" ? "border-[var(--ok-line)] bg-[var(--ok-bg)]" : "border-[var(--warn-line)] bg-[var(--warn-bg)]"}`}>
              <h2 className="display text-xl mb-2">
                {out.v === "ida" && "This is the guideline red flag"}
                {out.v === "ida-young" && "Iron-deficiency anaemia"}
                {out.v === "anaemia-check-iron" && "You are anaemic: ask for a ferritin test"}
                {out.v === "anaemia-not-iron" && "Anaemia without low iron"}
                {out.v === "low-iron-no-anaemia" && "Low iron, not anaemic"}
                {out.v === "normal" && "No red flag in these numbers"}
              </h2>
              <p className="text-sm text-[var(--ink-2)] leading-relaxed">
                {out.v === "ida" && "Anaemia with ferritin under 45 is iron-deficiency anaemia. For men and for women past menopause, the American Gastroenterological Association strongly recommends an upper endoscopy and a colonoscopy, even with no symptoms, to find where blood is being lost. Most causes are not cancer, but this is how bowel and stomach cancers are often found."}
                {(out.v === "ida" || out.v === "ida-young") && Number.isFinite(out.m) && !out.small && " Your red cells are a normal size. That is common: in US adults over 50, about three in four people with this result have normal-sized red cells, so a check that only looks for small red cells would miss it."}
                {out.v === "ida-young" && "Anaemia with ferritin under 45 is iron-deficiency anaemia. Heavy periods are the most common cause before menopause. The guideline still suggests considering an upper endoscopy and a colonoscopy, a decision to make with your doctor."}
                {out.v === "anaemia-check-iron" && `Your haemoglobin is under ${out.cutH}. Without a ferritin result there is no way to tell whether this is the iron-deficiency kind that the guideline says should be investigated.${out.small ? " Your small red cells (MCV under 80) make low iron more likely." : ""}`}
                {out.v === "anaemia-not-iron" && "Your iron stores are not low, so this is not the iron-deficiency kind. Anaemia has many causes, including kidney disease, inflammation and vitamin deficiencies. It is worth asking your doctor what is causing it."}
                {out.v === "low-iron-no-anaemia" && "Your ferritin is low but you are not anaemic. The guideline's work-up applies to iron-deficiency anaemia; ask your doctor whether your low iron needs explaining."}
                {out.v === "normal" && "Your haemoglobin is not low, so the iron-deficiency anaemia red flag does not apply. This page checks one guideline only; it is not a general all-clear."}
              </p>
            </section>

            {(out.v === "ida" || out.v === "ida-young") && (
              <section>
                <h2 className="display text-lg mb-2 rule-out">What the national data show</h2>
                <p className="text-sm text-[var(--ink-2)] leading-relaxed">
                  In US survey data that followed people for years, adults over 50 with this result died of cancer within five years more often: about 1 in{" "}
                  <span className="data text-[var(--ink)]">{NATIONAL.perCancerDeathIda}</span>, against 1 in <span className="data text-[var(--ink)]">{NATIONAL.perCancerDeathNone}</span> without
                  anaemia. They also died more often of other causes, so the result marks poor health generally, not only cancer. These are older surveys
                  (1988-1994 and 2001-2002), and the numbers are small.
                </p>
              </section>
            )}

            {letter && (
              <section>
                <h2 className="display text-lg mb-2 rule-out">Take this to your doctor</h2>
                <pre className="data text-xs text-[var(--ink-2)] whitespace-pre-wrap border border-[var(--rule)] p-4 leading-relaxed">{letter}</pre>
                <button type="button" onClick={() => navigator.clipboard?.writeText(letter)}
                  className="mt-3 text-xs border border-[var(--stamp-line)] text-[var(--stamp)] px-3 py-1.5 hover:bg-[var(--stamp-bg)]">Copy these questions</button>
              </section>
            )}

            <footer className="border-t border-[var(--rule)] pt-5 text-xs text-[var(--ink-3)] space-y-2 leading-relaxed">
              <p>Guideline: Ko CW et al. AGA clinical practice guidelines on the gastrointestinal evaluation of iron deficiency anemia. Gastroenterology 2020;159:1085-1094.
                Anaemia uses the World Health Organization cut-offs (13 g/dL men, 12 g/dL women).</p>
              <p>National figures: NHANES 2017-March 2020, adults 50+ (men and women past menopause), survey-weighted; 95% interval for the count {NATIONAL.millionsLo} to {NATIONAL.millionsHi} million.
                &ldquo;Not treated&rdquo; means they reported no treatment for anaemia in the past three months. Pre-registered analysis and code are in the project repository.</p>
            </footer>
          </div>
        )}
      </main>
    </div>
  );
}
