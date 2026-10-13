"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import model from "./model.json";

// Everything here runs in the browser from a lookup table exported by experiments/export_mced_app.py.
// The law rule, the test numbers and the model are explained on the page and in
// docs/MCED_LAW_PREREG.md; nothing is sent to a server.

type Model = { k: number; specificity: number; price: number; covered_percentiles: number[]; risk: Record<string, Record<string, Record<string, number>>> };
const M = model as Model;

const field = "w-full bg-[var(--paper)] border border-[var(--rule-strong)] px-3 py-2 text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--stamp-line)]";
const LAW_URL = "https://www.govinfo.gov/content/pkg/BILLS-119hr7148enr/html/BILLS-119hr7148enr.htm";

const snap = (v: number, step: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(v / step) * step));

// Pub. L. 119-75: Medicare pays for a multi-cancer test from 1 January 2029 for people aged 50 or older who,
// on 1 January of that year, have not reached 65 + (year - 2029). The limit rises one year per year, as fast
// as people age, so it reduces to a birth-date cutoff: anyone 65 or older on 1 January 2029 never qualifies.
function lawStatus(birthYear: number, onMedicareEarly: boolean) {
  if (birthYear <= 1963) {
    return {
      covered: false,
      head: "Medicare would never pay for this test for you",
      text: `Born in ${birthYear}, you will be 65 or older on 1 January 2029, when coverage starts. The age limit then rises by one year every year, exactly as fast as you age, so under the law as passed you never fall inside it.`,
    };
  }
  if (birthYear === 1964) {
    return {
      covered: true,
      head: "You are right at the line",
      text: "Born in 1964, you qualify unless your birthday is 1 January (anyone who has turned 65 by 1 January 2029 is out). If you qualify, you stay eligible every later year because the limit rises with you.",
    };
  }
  const when = onMedicareEarly
    ? `Because you are already on Medicare, it could pay for one test a year from ${Math.max(2029, birthYear + 50)}${birthYear + 50 > 2029 ? " (the law sets a floor of age 50)" : ""}`
    : `When you join Medicare at 65, in ${birthYear + 65}, it could pay for one test a year`;
  return {
    covered: true,
    head: "The law would cover you",
    text: `Born in ${birthYear}, you stay under the age limit for good. ${when}, once a test is approved by the FDA and Medicare decides to cover it.`,
  };
}

export default function MultiCancerPage() {
  const thisYear = new Date().getFullYear();
  const [birth, setBirth] = useState("1966");
  const [sex, setSex] = useState("female");
  const [smoke, setSmoke] = useState("never");
  const [cigs, setCigs] = useState("20");
  const [quit, setQuit] = useState("10");
  const [early, setEarly] = useState(false);

  const out = useMemo(() => {
    const by = Number(birth);
    if (!birth.trim() || !Number.isInteger(by) || by < 1900 || by > thisYear) return null;
    const age = thisYear - by;
    const law = lawStatus(by, early);
    if (age < 35 || age > 84) return { law, age, risk: null };
    const key = smoke === "never" ? "never:0" : smoke === "former" ? `former:${snap(Number(quit) || 0, 5, 0, 50)}` : `current:${snap(Number(cigs) || 0, 5, 5, 40)}`;
    const p = M.risk[sex === "male" ? "m" : "f"][String(age)][key];
    const det = Math.min(M.k * p, 0.5);
    const fp = (1 - M.specificity) * (1 - det);
    let rank = 0;
    while (rank < 100 && M.covered_percentiles[rank] < p) rank++;
    return { law, age, risk: { p, det, fp, ppv: det / (det + fp), rank, cost: M.price / det } };
  }, [birth, sex, smoke, cigs, quit, early, thisYear]);

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] font-sans">
      <main className="max-w-2xl mx-auto px-5 py-10 fade-in">
        <Link href="/" className="inline-flex items-center gap-2 text-xs text-[var(--stamp)] mb-8"><ArrowLeft className="w-4 h-4" /> Back to Oncovision</Link>

        <h1 className="display text-3xl leading-tight mb-3">Would Medicare pay for a multi-cancer blood test for you, and would it be worth it?</h1>
        <p className="text-sm text-[var(--ink-2)] leading-relaxed mb-3">
          In February 2026 Congress passed a law (Public Law 119-75) that lets Medicare pay, from 2029, for blood tests that look for
          many cancers at once, such as Galleri. The law only covers people who are under 65 when it starts. Four answers tell you whether you
          are one of them, and what the test would be likely to find in someone like you.
        </p>
        <p className="text-xs text-[var(--ink-3)] leading-relaxed mb-8 border-l-2 border-[var(--warn-line)] pl-3">
          Research prototype, not medical advice. No multi-cancer blood test has been shown to reduce deaths from cancer, and none is
          FDA-approved for screening. The estimates below come from a US national survey and one published study of the test.
        </p>

        <form className="border-y border-[var(--rule-strong)] py-6 mb-8" onSubmit={e => e.preventDefault()}>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div><label className="field-label block mb-1" htmlFor="birth">Year born</label>
              <input id="birth" className={`${field} data`} inputMode="numeric" value={birth} onChange={e => setBirth(e.target.value)} placeholder="e.g. 1966" /></div>
            <div><label className="field-label block mb-1" htmlFor="sex">Sex at birth</label>
              <select id="sex" className={field} value={sex} onChange={e => setSex(e.target.value)}>
                <option value="female">Female</option><option value="male">Male</option></select></div>
            <div><label className="field-label block mb-1" htmlFor="smoke">Smoking</label>
              <select id="smoke" className={field} value={smoke} onChange={e => setSmoke(e.target.value)}>
                <option value="never">Never smoked</option><option value="former">Used to smoke</option><option value="current">Smoke now</option></select></div>
            {smoke === "current" && (
              <div><label className="field-label block mb-1" htmlFor="cigs">Cigarettes a day</label>
                <input id="cigs" className={`${field} data`} inputMode="numeric" value={cigs} onChange={e => setCigs(e.target.value)} /></div>)}
            {smoke === "former" && (
              <div><label className="field-label block mb-1" htmlFor="quit">Years since quitting</label>
                <input id="quit" className={`${field} data`} inputMode="numeric" value={quit} onChange={e => setQuit(e.target.value)} /></div>)}
          </div>
          <label className="flex items-center gap-2 text-sm text-[var(--ink-2)] mt-5">
            <input type="checkbox" checked={early} onChange={e => setEarly(e.target.checked)} />
            I am on Medicare before 65 (for example because of a disability)
          </label>
        </form>

        {!out && (
          <p className="text-sm text-[var(--ink-2)] leading-relaxed">
            Enter the four-digit year you were born, for example 1966, and the answers appear here straight away. Nothing you type leaves this page.
          </p>
        )}

        {out && (
          <div className="space-y-8">
            <section className={`border p-5 ${out.law.covered ? "border-[var(--stamp-line)] bg-[var(--stamp-bg)]" : "border-[var(--flag-line)] bg-[var(--flag-bg)]"}`}>
              <h2 className="display text-xl mb-2">{out.law.head}</h2>
              <p className="text-sm text-[var(--ink-2)] leading-relaxed">{out.law.text}</p>
            </section>

            {!out.risk ? (
              <p className="text-sm text-[var(--ink-2)]">The risk estimates cover ages 35 to 84, so there is nothing more to show for age {out.age}.</p>
            ) : (
              <>
                <section>
                  <h2 className="display text-lg mb-2 rule-out">Your risk</h2>
                  <p className="text-sm text-[var(--ink-2)] leading-relaxed">
                    Of 1,000 people your age and sex with your smoking history, about <span className="data text-[var(--ink)]">{Math.round(out.risk.p * 1000)}</span> die
                    of some cancer within five years. That is higher than <span className="data text-[var(--ink)]">{out.risk.rank >= 100 ? ">99" : out.risk.rank}%</span> of
                    the people the law covers in 2029, who are mostly Medicare members aged 50 to 64.
                  </p>
                </section>

                <section>
                  <h2 className="display text-lg mb-3 rule-out">If 1,000 people like you took the test once</h2>
                  <dl className="grid grid-cols-3 border-y border-[var(--rule)] divide-x divide-[var(--rule)] mb-3">
                    <div className="py-3 px-3"><dt className="field-label">Cancers found</dt><dd className="data text-2xl mt-1">{(out.risk.det * 1000).toFixed(1)}</dd></div>
                    <div className="py-3 px-3"><dt className="field-label">False alarms</dt><dd className="data text-2xl mt-1">{(out.risk.fp * 1000).toFixed(1)}</dd></div>
                    <div className="py-3 px-3"><dt className="field-label">Positives that are real</dt><dd className="data text-2xl mt-1">{Math.round(out.risk.ppv * 100)}%</dd></div>
                  </dl>
                  <p className="text-sm text-[var(--ink-2)] leading-relaxed">
                    At the price the law sets (<span className="data">${M.price.toFixed(2)}</span> a test), Medicare would spend about{" "}
                    <span className="data text-[var(--ink)]">${Math.round(out.risk.cost).toLocaleString()}</span> on tests for every cancer found in people like you,
                    before any follow-up scans. The test also misses most cancers: it found about 4 in 10 of them in its largest published study, so a
                    negative result does not rule cancer out.
                  </p>
                </section>

                <section>
                  <h2 className="display text-lg mb-2 rule-out">What matters more than this test</h2>
                  <ul className="text-sm text-[var(--ink-2)] space-y-2 leading-relaxed">
                    <li>The screenings that are proven to save lives come first. <Link className="underline text-[var(--stamp)]" href="/screening">See which ones you are due for</Link>.</li>
                    {smoke === "current" && <li>Stopping smoking lowers your cancer risk more than any test can. Free help in the US: 1-800-QUIT-NOW.</li>}
                    {smoke !== "never" && <li>If you smoked heavily, ask about a yearly low-dose CT scan of the lungs. It is the one cancer screen built for smokers, and it saves lives.</li>}
                    <li>Weight loss you cannot explain, bleeding where there should be none, or a new lump: see a doctor now rather than waiting for any test.</li>
                  </ul>
                </section>
              </>
            )}

            <footer className="border-t border-[var(--rule)] pt-5 text-xs text-[var(--ink-3)] space-y-2 leading-relaxed">
              <p>Risk comes from a model fitted on 110,000 US adults in the National Health Interview Survey (1997-2002) and checked on 73,000 more
                (2005-2009), with deaths followed through 2019. Test performance comes from the PATHFINDER 2 study (specificity 99.64%; cancer found
                in 0.54% of adults over 50), assuming the chance the test finds a cancer rises in step with the chance of dying of one. That assumption
                is uncertain, and survey data from those years overstate today&apos;s smoking.</p>
              <p>Sources: <a className="underline" href={LAW_URL} target="_blank" rel="noreferrer">the law as passed</a>,{" "}
                <a className="underline" href="https://pubmed.ncbi.nlm.nih.gov/42773207/" target="_blank" rel="noreferrer">PATHFINDER 2</a>, and the
                pre-registered analysis in the project repository.</p>
            </footer>
          </div>
        )}
      </main>
    </div>
  );
}
