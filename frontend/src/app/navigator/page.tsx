"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowLeft, ClipboardList, Copy, ShieldCheck } from "lucide-react";

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

type Sym = { key: string; label: string; hint: string; asks_frequency: boolean };
type Group = { group: string; symptoms: Sym[] };
type Finding = { key: string; label: string };
type Feature = { key: string; points: number; label: string };
type LabDef = { key: string; label: string; unit: string };
type Vocab = {
  symptom_groups: Group[];
  findings: Finding[];
  melanoma_features: Feature[];
  labs: LabDef[];
  meta: { guideline_version: string; review_status: string; known_limitations: string[] };
};
type Match = {
  id: string; site: string; tier: string; timeframe: string; strength: string;
  action: string; source: string; because: string[];
};
type Could = { id: string; site: string; tier: string; needs: string[]; if_so: string };
type Result = {
  state: string; headline: string; matches: Match[]; could_apply_if: Could[];
  melanoma_score: number | null; lab_notes: string[]; lab_findings: Record<string, boolean>;
  ignored_symptoms: string[]; safety: string[]; disclaimer: string;
  rules: { guideline_version: string; review_status: string; n_rules: number };
};

const STATE_STYLE: Record<string, { box: string; label: string; text: string }> = {
  talk_soon: { box: "border-[var(--flag-line)] bg-[var(--flag-bg)]", text: "text-[var(--flag)]", label: "Worth seeing a doctor soon" },
  worth_raising: { box: "border-[var(--warn-line)] bg-[var(--warn-bg)]", text: "text-[var(--warn)]", label: "Worth raising with a doctor" },
  mention: { box: "border-[var(--rule-strong)] bg-[var(--paper-2)]", text: "text-[var(--ink-2)]", label: "Mention at a routine visit" },
  need_info: { box: "border-[var(--stamp-line)] bg-[var(--stamp-bg)]", text: "text-[var(--stamp)]", label: "A few more facts could change this" },
  nothing_meets: { box: "border-[var(--ok-line)] bg-[var(--ok-bg)]", text: "text-[var(--ok)]", label: "No guideline threshold met" },
};
const TIER_LABEL: Record<string, string> = {
  talk_soon: "See a doctor soon", worth_raising: "Worth raising", mention: "Routine visit",
};
const LAB_FINDING_TEXT: Record<string, string> = {
  anaemia: "low haemoglobin (anaemia)", iron_deficiency: "low iron stores",
  iron_deficiency_anaemia: "iron-deficiency anaemia", thrombocytosis: "a high platelet count",
  raised_wbc: "a raised white blood cell count", ca125_high: "CA-125 of 35 or above",
};

const field = "w-full bg-[var(--paper)] border border-[var(--rule-strong)] px-3 py-2 text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--stamp-line)]";
const lbl = "block text-[10px] font-bold uppercase tracking-widest text-[var(--ink-3)] mb-1";

export default function NavigatorPage() {
  const [vocab, setVocab] = useState<Vocab | null>(null);
  const [loadError, setLoadError] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [smoked, setSmoked] = useState("");
  const [asbestos, setAsbestos] = useState("");
  const [picked, setPicked] = useState<Record<string, { explained: boolean; freq: string }>>({});
  const [labs, setLabs] = useState<Record<string, string>>({});
  const [findings, setFindings] = useState<Record<string, boolean>>({});
  const [lesion, setLesion] = useState(false);
  const [lesionFeatures, setLesionFeatures] = useState<Record<string, boolean>>({});
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  async function loadVocab() {
    setLoadError("");
    try {
      const res = await fetch(`${API_BASE}/navigator/vocabulary`);
      if (!res.ok) throw new Error(String(res.status));
      setVocab(await res.json());
    } catch {
      setLoadError(
        "The service may be starting up, which can take up to a minute on the free host. Try again in a moment.",
      );
    }
  }
  useEffect(() => { loadVocab(); }, []);

  const symLabel = useMemo(() => {
    const m: Record<string, string> = {};
    vocab?.symptom_groups.forEach(g => g.symptoms.forEach(s => { m[s.key] = s.label; }));
    return m;
  }, [vocab]);

  function toggle(key: string) {
    setPicked(p => {
      const n = { ...p };
      if (n[key]) delete n[key]; else n[key] = { explained: false, freq: "" };
      return n;
    });
  }

  function buildRequest() {
    const body: Record<string, unknown> = {};
    if (age.trim() !== "") body.age = Number(age);
    if (sex) body.sex = sex;
    if (smoked) body.ever_smoked = smoked === "yes";
    if (asbestos) body.asbestos = asbestos === "yes";
    body.symptoms = Object.entries(picked).map(([key, v]) => ({
      key, explained: v.explained,
      times_per_month: v.freq.trim() === "" ? null : Number(v.freq),
    }));
    const l: Record<string, number> = {};
    Object.entries(labs).forEach(([k, v]) => { if (v.trim() !== "") l[k] = Number(v); });
    if (Object.keys(l).length) body.labs = l;
    const f = Object.entries(findings).filter(([, v]) => v).map(([k]) => k);
    if (f.length) body.findings = f;
    if (lesion) body.skin_lesion = lesionFeatures;
    return body;
  }

  async function run() {
    setBusy(true); setError(""); setResult(null); setCopied(false);
    try {
      const res = await fetch(`${API_BASE}/navigator`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildRequest()),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.message || data.detail?.[0]?.msg || "That could not be read."); return; }
      setResult(data);
    } catch {
      setError("Could not reach the service. It may be starting up; try again in a moment.");
    } finally { setBusy(false); }
  }

  function summaryText(r: Result): string {
    const lines: string[] = [];
    lines.push("Symptom and lab pattern summary (research prototype, not medical advice)");
    lines.push(`Guideline basis: NICE NG12 ${r.rules.guideline_version} edition, not yet reviewed by a clinician.`);
    lines.push("");
    lines.push(`Age: ${age || "not given"}   Sex: ${sex || "not given"}`);
    const symptoms = Object.entries(picked).filter(([, v]) => !v.explained).map(([k, v]) =>
      `- ${symLabel[k] || k}${v.freq ? ` (${v.freq} days a month)` : ""}`);
    lines.push("Symptoms I have noticed:");
    lines.push(symptoms.length ? symptoms.join("\n") : "- none entered");
    const labLines = Object.entries(labs).filter(([, v]) => v.trim() !== "").map(([k, v]) => {
      const d = vocab?.labs.find(x => x.key === k);
      return `- ${d?.label || k}: ${v} ${d?.unit || ""}`;
    });
    if (labLines.length) { lines.push("Lab results I entered:"); lines.push(labLines.join("\n")); }
    lines.push("");
    lines.push(`Result: ${r.headline}`);
    r.matches.forEach(m => {
      lines.push(`- ${m.site} (${TIER_LABEL[m.tier]}): ${m.action}`);
      lines.push(`  Flagged because: ${m.because.join("; ")}.  Source: ${m.source}`);
    });
    r.could_apply_if.forEach(c => lines.push(`- Could apply to ${c.site} if known: ${c.needs.join("; ")}`));
    lines.push("");
    lines.push("This tool cannot examine me. Please use your judgement over its output.");
    return lines.join("\n");
  }

  async function copy() {
    if (!result) return;
    try { await navigator.clipboard.writeText(summaryText(result)); setCopied(true); } catch { setCopied(false); }
  }

  const st = result ? STATE_STYLE[result.state] : null;

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] font-sans">
      <div className="max-w-3xl mx-auto px-5 py-8">
        <a href="/" className="inline-flex items-center gap-2 text-xs font-bold text-[var(--stamp)] mb-6">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Oncovision
        </a>

        <h1 className="display text-3xl mb-2">Symptom and lab pattern navigator</h1>
        <p className="text-[var(--ink-2)] text-sm mb-5">
          Tell it what you have noticed and, if you have them, your lab results. It checks whether the
          combination meets a published guideline threshold for a prompt check, and says what to ask for.
        </p>

        <div className="border border-[var(--warn-line)] bg-[var(--warn-bg)] p-4 mb-6 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-[var(--warn)] flex-shrink-0 mt-0.5" />
          <div className="text-xs text-[var(--ink-2)] leading-relaxed">
            <p className="font-bold text-[var(--warn)] uppercase tracking-wider mb-1">Research prototype. Not medical advice.</p>
            It applies UK guideline thresholds (the 2015 edition) that a clinician has not yet reviewed for
            this tool, and it cannot examine you. It never says you have cancer, and meeting no threshold does
            not mean nothing is wrong. If something worries you, or is getting worse, see a doctor.
            <span className="block mt-2 font-bold">
              Coughing or vomiting a lot of blood, severe chest pain, trouble breathing, or black bloody stools
              with faintness need urgent care now, not this page.
            </span>
          </div>
        </div>

        {!vocab && !loadError && <p className="text-sm text-[var(--ink-3)]">Loading…</p>}
        {loadError && (
          <div className="border border-[var(--rule-strong)] bg-[var(--paper-2)] p-4 text-sm">
            <p className="text-[var(--ink-2)] mb-3">{loadError}</p>
            <button onClick={loadVocab} className="bg-[var(--stamp-solid)] hover:bg-[var(--stamp-solid-hover)] px-4 py-2 text-xs font-bold uppercase tracking-wider">
              Try again
            </button>
          </div>
        )}

        {vocab && (
          <div className="space-y-6">
            <section className="border border-[var(--rule)] bg-[var(--surface)] p-5">
              <h2 className="display text-lg mb-4">About you</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div><label className={lbl} htmlFor="age">Age</label>
                  <input id="age" className={field} inputMode="decimal" value={age} onChange={e => setAge(e.target.value)} placeholder="years" /></div>
                <div><label className={lbl} htmlFor="sex">Sex</label>
                  <select id="sex" className={field} value={sex} onChange={e => setSex(e.target.value)}>
                    <option value="">Not given</option><option value="female">Female</option><option value="male">Male</option></select></div>
                <div><label className={lbl} htmlFor="smoked">Ever smoked</label>
                  <select id="smoked" className={field} value={smoked} onChange={e => setSmoked(e.target.value)}>
                    <option value="">Not given</option><option value="yes">Yes</option><option value="no">No</option></select></div>
                <div><label className={lbl} htmlFor="asb">Asbestos exposure</label>
                  <select id="asb" className={field} value={asbestos} onChange={e => setAsbestos(e.target.value)}>
                    <option value="">Not given</option><option value="yes">Yes</option><option value="no">No</option></select></div>
              </div>
            </section>

            <section className="border border-[var(--rule)] bg-[var(--surface)] p-5">
              <h2 className="display text-lg mb-1">What you have noticed</h2>
              <p className="text-xs text-[var(--ink-3)] mb-4">
                Tick only things with no known cause. If something has an explanation (an injury, a diagnosed
                condition, a medicine), do not tick it.
              </p>
              <div className="space-y-5">
                {vocab.symptom_groups.map(g => (
                  <div key={g.group}>
                    <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--stamp)] mb-2">{g.group}</p>
                    <div className="space-y-1.5">
                      {g.symptoms.map(s => {
                        const on = !!picked[s.key];
                        return (
                          <div key={s.key} className={`flex flex-wrap items-center gap-3 px-3 py-2 border ${on ? "border-[var(--stamp-line)] bg-[var(--stamp-bg)]" : "border-[var(--rule)]"}`}>
                            <label className="flex items-start gap-3 flex-1 min-w-[220px] cursor-pointer text-sm">
                              <input type="checkbox" checked={on} onChange={() => toggle(s.key)} className="mt-1" />
                              <span>{s.label}{s.hint && <span className="block text-[11px] text-[var(--ink-3)]">{s.hint}</span>}</span>
                            </label>
                            {on && s.asks_frequency && (
                              <span className="flex items-center gap-2 text-xs text-[var(--ink-2)]">
                                <input aria-label={`Days a month: ${s.label}`} className={`${field} !w-16 !py-1`} inputMode="numeric" placeholder="0–31"
                                  value={picked[s.key].freq}
                                  onChange={e => setPicked(p => ({ ...p, [s.key]: { ...p[s.key], freq: e.target.value } }))} />
                                days a month
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="border border-[var(--rule)] bg-[var(--surface)] p-5">
              <h2 className="display text-lg mb-1">Your lab results, if you have them</h2>
              <p className="text-xs text-[var(--ink-3)] mb-4">All optional. A result that is missing is asked about, never assumed.</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                {vocab.labs.map(d => (
                  <div key={d.key}><label className={lbl} htmlFor={`lab-${d.key}`}>{d.label} ({d.unit})</label>
                    <input id={`lab-${d.key}`} className={`${field} font-mono`} inputMode="decimal" value={labs[d.key] || ""}
                      onChange={e => setLabs(l => ({ ...l, [d.key]: e.target.value }))} /></div>
                ))}
              </div>
              <div className="mt-5 space-y-1.5">
                <p className={lbl}>Reports you were given</p>
                {vocab.findings.map(f => (
                  <label key={f.key} className="flex items-center gap-3 text-sm cursor-pointer">
                    <input type="checkbox" checked={!!findings[f.key]} onChange={e => setFindings(x => ({ ...x, [f.key]: e.target.checked }))} />
                    {f.label}
                  </label>
                ))}
              </div>
            </section>

            <section className="border border-[var(--rule)] bg-[var(--surface)] p-5">
              <label className="flex items-center gap-3 text-sm font-bold cursor-pointer">
                <input type="checkbox" checked={lesion} onChange={e => setLesion(e.target.checked)} />
                I have a mole or skin spot I am worried about
              </label>
              {lesion && (
                <div className="mt-4 space-y-1.5">
                  <p className="text-xs text-[var(--ink-3)] mb-2">
                    This is the checklist doctors score. Scoring your own skin is not reliable, so use it only
                    as a prompt to have the spot looked at.
                  </p>
                  {vocab.melanoma_features.map(f => (
                    <label key={f.key} className="flex items-center gap-3 text-sm cursor-pointer">
                      <input type="checkbox" checked={!!lesionFeatures[f.key]}
                        onChange={e => setLesionFeatures(x => ({ ...x, [f.key]: e.target.checked }))} />
                      {f.label}
                    </label>
                  ))}
                </div>
              )}
            </section>

            <button onClick={run} disabled={busy}
              className="w-full bg-[var(--stamp-solid)] hover:bg-[var(--stamp-solid-hover)] disabled:opacity-60 px-6 py-3 text-sm font-bold uppercase tracking-wider">
              {busy ? "Checking…" : "Check this pattern"}
            </button>
            {error && <p className="text-sm text-[var(--flag)]">{error}</p>}
          </div>
        )}

        {result && st && (
          <div className="mt-8 space-y-5">
            <div className={`border p-5 ${st.box}`}>
              <p className={`text-[10px] font-bold uppercase tracking-widest mb-2 ${st.text}`}>{st.label}</p>
              <p className="text-base leading-relaxed">{result.headline}</p>
            </div>

            {result.matches.map(m => (
              <div key={m.id} className="border border-[var(--rule)] bg-[var(--surface)] p-5">
                <div className="flex flex-wrap items-center gap-3 mb-2">
                  <h3 className="display text-lg capitalize">{m.site}</h3>
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 border ${STATE_STYLE[m.tier].box} ${STATE_STYLE[m.tier].text}`}>
                    {TIER_LABEL[m.tier]}
                  </span>
                </div>
                <p className="text-sm leading-relaxed mb-3">{m.action}</p>
                <p className={lbl}>Why this was flagged</p>
                <ul className="text-sm text-[var(--ink-2)] list-disc pl-5 space-y-0.5 mb-3">
                  {m.because.map((b, i) => <li key={i}>{b}</li>)}
                </ul>
                <p className="text-[11px] text-[var(--ink-4)]">Source: {m.source}</p>
              </div>
            ))}

            {result.could_apply_if.length > 0 && (
              <div className="border border-[var(--stamp-line)] bg-[var(--stamp-bg)] p-5">
                <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--stamp)] mb-3">Could apply if you add</p>
                <ul className="space-y-2 text-sm">
                  {result.could_apply_if.map(c => (
                    <li key={c.id}><span className="capitalize font-bold">{c.site}</span>: {c.needs.join(" and ")}.</li>
                  ))}
                </ul>
              </div>
            )}

            {(Object.keys(result.lab_findings).length > 0 || result.lab_notes.length > 0) && (
              <div className="border border-[var(--rule)] bg-[var(--surface)] p-5 text-sm">
                <p className={lbl}>What your lab results show</p>
                <ul className="list-disc pl-5 text-[var(--ink-2)] space-y-0.5">
                  {Object.entries(result.lab_findings).filter(([, v]) => v).map(([k]) => (
                    <li key={k}>{LAB_FINDING_TEXT[k] || k}</li>
                  ))}
                </ul>
                {result.lab_notes.map((n, i) => <p key={i} className="text-xs text-[var(--warn)] mt-2">{n}</p>)}
              </div>
            )}

            {result.ignored_symptoms.length > 0 && (
              <p className="text-xs text-[var(--ink-3)]">Not recognised and ignored: {result.ignored_symptoms.join(", ")}.</p>
            )}

            <div className="border border-[var(--rule)] bg-[var(--surface)] p-5">
              {result.safety.map((s, i) => (
                <p key={i} className="text-sm text-[var(--ink-2)] leading-relaxed mb-2 flex gap-2">
                  <ShieldCheck className="w-4 h-4 text-[var(--ok)] flex-shrink-0 mt-0.5" /> {s}
                </p>
              ))}
              <p className="text-[11px] text-[var(--ink-3)] mt-3 leading-relaxed">{result.disclaimer}</p>
            </div>

            <button onClick={copy}
              className="inline-flex items-center gap-2 border border-[var(--stamp-line)] text-[var(--stamp)] hover:bg-[var(--stamp-bg)] px-4 py-2 text-xs font-bold uppercase tracking-wider">
              {copied ? <ClipboardList className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? "Copied. Paste it into a note or message." : "Copy a summary for my doctor"}
            </button>
          </div>
        )}

        {vocab && (
          <details className="mt-10 text-xs text-[var(--ink-3)]">
            <summary className="cursor-pointer font-bold uppercase tracking-widest">What this tool cannot do</summary>
            <ul className="list-disc pl-5 mt-3 space-y-1.5 leading-relaxed">
              {vocab.meta.known_limitations.map((l, i) => <li key={i}>{l}</li>)}
              <li>Review status of the rules: {vocab.meta.review_status}.</li>
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}
