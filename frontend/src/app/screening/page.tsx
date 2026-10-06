"use client";

import { useState } from "react";
import { AlertTriangle, ArrowLeft, ExternalLink, ShieldCheck } from "lucide-react";

// Same resolution as the main page: the Render service is "oncovisonai" (one i short of
// the project name), and an older service on a similar host must never be trusted.
const DEFAULT_API = "https://oncovisonai.onrender.com";
const LEGACY_API = "oncovision-backend.onrender.com";
function apiBase(): string {
  const configured = (process.env.NEXT_PUBLIC_API_URL || "").trim().replace(/\/+$/, "");
  if (!configured || configured.includes(LEGACY_API)) return DEFAULT_API;
  return configured;
}
const API_BASE = apiBase();

type Item = {
  id: string; site: string; test: string; status: string; headline: string; detail: string;
  source: string; url: string; needs: string[]; options: string[];
};
type Result = {
  headline: string; items: Item[]; needs: string[]; family_note: string; disclaimer: string;
  meta: { review_status: string; checked_against: string; scope: string; not_covered: string };
};

const STATUS: Record<string, { label: string; box: string; text: string }> = {
  due: { label: "Recommended", box: "border-[var(--stamp-line)] bg-[var(--stamp-bg)]", text: "text-[var(--stamp)]" },
  individual: { label: "Your choice", box: "border-[var(--warn-line)] bg-[var(--warn-bg)]", text: "text-[var(--warn)]" },
  need_info: { label: "Needs more facts", box: "border-[var(--rule-strong)] bg-[var(--paper-2)]", text: "text-[var(--ink-2)]" },
  no_recommendation: { label: "No recommendation", box: "border-[var(--rule)] bg-[var(--surface)]", text: "text-[var(--ink-3)]" },
  not_yet: { label: "Not yet", box: "border-[var(--rule)] bg-[var(--surface)]", text: "text-[var(--ink-3)]" },
  not_needed: { label: "Not recommended for you", box: "border-[var(--rule)] bg-[var(--surface)]", text: "text-[var(--ink-3)]" },
  not_recommended: { label: "Not recommended", box: "border-[var(--rule)] bg-[var(--surface)]", text: "text-[var(--ink-3)]" },
};

const field = "w-full bg-[var(--paper)] border border-[var(--rule-strong)] px-3 py-2 text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--stamp-line)]";
const lbl = "block text-[10px] font-bold uppercase tracking-widest text-[var(--ink-3)] mb-1";

export default function ScreeningPage() {
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [cervix, setCervix] = useState("");
  const [smoked, setSmoked] = useState("");
  const [now, setNow] = useState("");
  const [quit, setQuit] = useState("");
  const [packs, setPacks] = useState("");
  const [years, setYears] = useState("");
  const [packYears, setPackYears] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    const body: Record<string, unknown> = {};
    if (age.trim() !== "") body.age = Number(age);
    if (sex) body.sex_at_birth = sex;
    if (cervix) body.has_cervix = cervix === "yes";
    if (smoked) body.ever_smoked = smoked === "yes";
    if (now) body.smokes_now = now === "yes";
    if (quit.trim() !== "") body.years_since_quit = Number(quit);
    const py = packYears.trim() !== "" ? Number(packYears) : packs.trim() !== "" && years.trim() !== "" ? Number(packs) * Number(years) : null;
    if (py !== null) body.pack_years = py;
    try {
      const res = await fetch(`${API_BASE}/screening`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) setError(data.message || "Something was wrong with what was entered.");
      else setResult(data);
    } catch {
      setError("The service may be starting up, which can take up to a minute on the free host. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  }

  const smokerFollowUps = smoked === "yes";

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] font-sans">
      <div className="max-w-3xl mx-auto px-5 py-8">
        <a href="/" className="inline-flex items-center gap-2 text-xs text-[var(--stamp)] mb-6"><ArrowLeft className="w-4 h-4" /> Back to Oncovision</a>
        <h1 className="display text-3xl mb-2">Which cancer screenings are you due for?</h1>
        <p className="text-sm text-[var(--ink-2)] mb-5 leading-relaxed">
          Enter your age and a few facts. This lists the routine screenings that the US Preventive Services Task Force recommends
          for people at average risk, and explains each one.
        </p>

        <div className="border border-[var(--warn-line)] bg-[var(--warn-bg)] p-4 mb-6 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-[var(--warn)] flex-shrink-0 mt-0.5" />
          <div className="text-sm text-[var(--ink-2)] leading-relaxed">
            <p className="font-bold text-[var(--warn)] uppercase tracking-wide text-xs mb-1">Research prototype. Not medical advice.</p>
            <p>
              It covers average-risk adults only, and only breast, cervical, bowel, lung and prostate cancer. It has not been reviewed by a
              clinician. If you have symptoms, use the <a className="underline" href="/navigator">symptom navigator</a> or see a doctor.
            </p>
          </div>
        </div>

        <section className="border border-[var(--rule)] bg-[var(--surface)] p-5 mb-6">
          <h2 className="display text-lg mb-4">About you</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div><label className={lbl} htmlFor="age">Age</label>
              <input id="age" className={field} inputMode="decimal" value={age} onChange={e => setAge(e.target.value)} placeholder="years" /></div>
            <div><label className={lbl} htmlFor="sex">Sex at birth</label>
              <select id="sex" className={field} value={sex} onChange={e => setSex(e.target.value)}>
                <option value="">Not given</option><option value="female">Female</option><option value="male">Male</option></select></div>
            {sex === "female" && (
              <div><label className={lbl} htmlFor="cervix">Still have a cervix</label>
                <select id="cervix" className={field} value={cervix} onChange={e => setCervix(e.target.value)}>
                  <option value="">Not sure</option><option value="yes">Yes</option><option value="no">No (hysterectomy)</option></select></div>
            )}
            <div><label className={lbl} htmlFor="smoked">Ever smoked regularly</label>
              <select id="smoked" className={field} value={smoked} onChange={e => setSmoked(e.target.value)}>
                <option value="">Not given</option><option value="yes">Yes</option><option value="no">No</option></select></div>
          </div>
          {smokerFollowUps && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
              <div><label className={lbl} htmlFor="now">Smoke now</label>
                <select id="now" className={field} value={now} onChange={e => setNow(e.target.value)}>
                  <option value="">Not given</option><option value="yes">Yes</option><option value="no">No, I quit</option></select></div>
              {now === "no" && (
                <div><label className={lbl} htmlFor="quit">Years since quitting</label>
                  <input id="quit" className={field} inputMode="decimal" value={quit} onChange={e => setQuit(e.target.value)} placeholder="years" /></div>
              )}
              <div><label className={lbl} htmlFor="packs">Packs a day</label>
                <input id="packs" className={field} inputMode="decimal" value={packs} onChange={e => setPacks(e.target.value)} placeholder="e.g. 1" /></div>
              <div><label className={lbl} htmlFor="years">Years smoked</label>
                <input id="years" className={field} inputMode="decimal" value={years} onChange={e => setYears(e.target.value)} placeholder="years" /></div>
              <div className="col-span-2"><label className={lbl} htmlFor="py">Or pack-years if you know it</label>
                <input id="py" className={field} inputMode="decimal" value={packYears} onChange={e => setPackYears(e.target.value)} placeholder="packs a day x years" /></div>
            </div>
          )}
        </section>

        <button onClick={run} disabled={busy}
          className="w-full bg-[var(--stamp-solid)] hover:bg-[var(--stamp-solid-hover)] disabled:opacity-60 px-6 py-3 text-sm font-bold uppercase tracking-wider mb-6">
          {busy ? "Checking…" : "Check my screenings"}
        </button>
        {error && <p className="text-sm text-[var(--flag)] mb-4" role="alert">{error}</p>}

        {result && (
          <div className="space-y-4">
            <div className="border border-[var(--stamp-line)] bg-[var(--stamp-bg)] p-5">
              <p className="text-lg leading-relaxed">{result.headline}</p>
            </div>
            {result.items.map(i => {
              const s = STATUS[i.status] || STATUS.not_yet;
              return (
                <div key={i.id} className={`border p-5 ${s.box}`}>
                  <div className="flex flex-wrap items-center gap-3 mb-2">
                    <h3 className="font-bold text-base">{i.headline}</h3>
                    <span className={`text-[10px] font-bold uppercase tracking-widest border px-2 py-0.5 ${s.text}`}>{s.label}</span>
                  </div>
                  <p className="text-sm text-[var(--ink-2)] leading-relaxed">{i.detail}</p>
                  {i.needs.length > 0 && (
                    <p className="text-sm mt-2"><span className="font-bold">Needed to say:</span> {i.needs.join(", ")}.</p>
                  )}
                  {i.options.length > 0 && (
                    <div className="mt-3 text-sm">
                      <p className={lbl}>Choose one</p>
                      <ul className="list-disc pl-5 text-[var(--ink-2)] space-y-0.5">{i.options.map(o => <li key={o}>{o}</li>)}</ul>
                    </div>
                  )}
                  <p className="text-xs text-[var(--ink-3)] mt-3">
                    Source: {i.source}{" "}
                    <a className="underline inline-flex items-center gap-1" href={i.url} target="_blank" rel="noreferrer">read it <ExternalLink className="w-3 h-3" /></a>
                  </p>
                </div>
              );
            })}
            <div className="border border-[var(--rule)] bg-[var(--surface)] p-5 text-sm space-y-2">
              <p className="text-[var(--ink-2)] leading-relaxed flex gap-2"><ShieldCheck className="w-4 h-4 text-[var(--ok)] flex-shrink-0 mt-0.5" /> {result.family_note}</p>
              <p className="text-[var(--ink-3)] text-xs leading-relaxed">{result.disclaimer}</p>
              <p className="text-[var(--ink-3)] text-xs leading-relaxed">{result.meta.not_covered}</p>
              <p className="text-[var(--ink-3)] text-xs">Checked against: {result.meta.checked_against}. {result.meta.review_status}.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
