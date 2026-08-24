"use client";

import { useState, useRef, useEffect } from "react";
import {
  Activity, BrainCircuit, Scan, UploadCloud, FileText, ChevronDown, ChevronUp,
  LayoutDashboard, FileCheck, Info, Mail, Globe, ShieldCheck, Microscope,
  Linkedin, BookOpen, Code2, AlertTriangle, Beaker, Layers, GitBranch, Server,
  Cpu, Shuffle, ClipboardList, Target,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

import { LAB_GROUPS, LAB_KEYS, HISTORY_FIELDS, ALL_KEYS } from "./fields";
import { CASE_POOL, OPENING_CASE, caseValues, randomCase, type DemoCase } from "./cases";

// The Render service that deploys backend/ from this repo. The host is
// "oncovisonai" without the second i, which is the real service name.
const DEFAULT_API = "https://oncovisonai.onrender.com";

// An older, unrelated service that still answers on this host. It runs a
// different API with one model and a Gemini integration, so it 404s on /models
// and 422s on /predict no matter what this app sends. It is rejected here
// rather than trusted, because a stale NEXT_PUBLIC_API_URL pointing at it is
// otherwise silent and looks exactly like a backend outage.
const LEGACY_API = "oncovision-backend.onrender.com";

function resolveApiBase(): string {
  const configured = (process.env.NEXT_PUBLIC_API_URL || "").trim().replace(/\/+$/, "");
  if (!configured) return DEFAULT_API;
  if (configured.includes(LEGACY_API)) return DEFAULT_API;
  return configured;
}

const API_BASE = resolveApiBase();

// Held-out test results from evaluate.py. Measured on a 20% split cut before
// anything was fitted and never used for training, selection, or calibration.
// Used until the live registry responds.
const FALLBACK_METRICS: Record<string, any> = {
  general: {
    label: "General Cancer Risk", auc: 0.966, auc_ci: [0.937, 0.989],
    sensitivity: 0.91, sensitivity_ci: [0.856, 0.961],
    specificity: 0.984, specificity_ci: [0.964, 1.0],
    brier: 0.0435, calibration_slope: 1.176,
    ppv_at_population_prevalence: 0.20605, people_flagged_per_true_case: 4.9,
    population_prevalence: 0.004507, cohort_prevalence: 0.371,
    baseline_logistic_auc: 0.917, baseline_age_sex_auc: 0.66,
    n_samples: 1500, n_test: 300, n_features: 8,
  },
  breast: {
    label: "Breast Cancer Risk", auc: 0.972, auc_ci: [0.94, 0.994],
    sensitivity: 0.786, sensitivity_ci: [0.66, 0.902],
    specificity: 0.958, specificity_ci: [0.904, 1.0],
    brier: 0.0668, calibration_slope: 1.118,
    ppv_at_population_prevalence: 0.02441, people_flagged_per_true_case: 41.0,
    population_prevalence: 0.001325, cohort_prevalence: 0.373,
    baseline_logistic_auc: 0.964, baseline_age_sex_auc: null,
    n_samples: 569, n_test: 114, n_features: 4,
  },
  liver: {
    label: "Liver Cancer Risk", auc: 0.97, auc_ci: [0.958, 0.979],
    sensitivity: 0.798, sensitivity_ci: [0.745, 0.848],
    specificity: 0.986, specificity_ci: [0.977, 0.994],
    brier: 0.044, calibration_slope: 0.861,
    ppv_at_population_prevalence: 0.00536, people_flagged_per_true_case: 186.5,
    population_prevalence: 0.000095, cohort_prevalence: 0.218,
    baseline_logistic_auc: 0.942, baseline_age_sex_auc: 0.634,
    n_samples: 5000, n_test: 1000, n_features: 12,
  },
  pancreatic: {
    label: "Pancreatic Cancer Risk", auc: 0.969, auc_ci: [0.939, 0.991],
    sensitivity: 0.731, sensitivity_ci: [0.55, 0.885],
    specificity: 0.947, specificity_ci: [0.896, 0.989],
    brier: 0.0617, calibration_slope: 0.46,
    ppv_at_population_prevalence: 0.00191, people_flagged_per_true_case: 524.6,
    population_prevalence: 0.000139, cohort_prevalence: 0.217,
    baseline_logistic_auc: 0.968, baseline_age_sex_auc: 0.5,
    n_samples: 600, n_test: 120, n_features: 6,
  },
};

// Trained and measured, deliberately not served. Reported rather than deleted,
// because a withdrawn panel is evidence about the method.
const WITHDRAWN_PANELS = [
  {
    name: "Prostate",
    auc: 0.786, ci: [0.505, 0.99], logistic: 0.769,
    specificity: 0.571, spec_ci: [0.167, 1.0],
    n: 97, n_test: 20, features: 2,
    reason:
      "The lower bound of the AUC interval sits on chance, so the panel cannot be shown to work at all. " +
      "Specificity is 0.571 with an interval from 0.167 to 1.0, which carries no information, because the " +
      "test split is 20 records. The ensemble does not meaningfully beat plain logistic regression either. " +
      "97 records and two usable screening features cannot support a clinical claim.",
  },
];

const OncovisionLogo = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 50C10 50 25 20 50 20C75 20 90 50 90 50C90 50 75 80 50 80C25 80 10 50 10 50Z" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="50" cy="50" r="10" stroke="currentColor" strokeWidth="4" fill="#020617" />
    <circle cx="50" cy="50" r="3" fill="currentColor" />
    <path d="M25 40L50 50M25 60L50 50M75 40L50 50M75 60L50 50" stroke="currentColor" strokeWidth="2" strokeDasharray="3 3" />
  </svg>
);

const asStrings = (v: Record<string, number>) =>
  Object.fromEntries(Object.entries(v).map(([k, val]) => [k, String(val)]));

const EMPTY = Object.fromEntries(ALL_KEYS.map(k => [k, ""])) as Record<string, string>;

export default function OncovisionDashboard() {
  const [currentPage, setCurrentPage] = useState<"dashboard" | "guide" | "about" | "developer">("dashboard");

  const [loading, setLoading] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [ignored, setIgnored] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [registry, setRegistry] = useState<Record<string, any> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Opens on a fixed breast record so the first render is stable, then every
  // press of Generate case draws a different record at random.
  const [activeCase, setActiveCase] = useState<DemoCase | null>(OPENING_CASE);
  const [formData, setFormData] = useState<Record<string, string>>(
    { ...EMPTY, ...asStrings(caseValues(OPENING_CASE)) }
  );

  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(d => { if (d.status === "success") setRegistry(d.models); })
      .catch(() => { /* fallback values already in place */ });
  }, []);

  const setField = (key: string, value: string) => {
    setActiveCase(null);
    setFormData(prev => ({ ...prev, [key]: value }));
  };

  // Draws a different record from the pool every press, never the same one
  // twice in a row.
  const nextCase = () => {
    const drawn = randomCase(activeCase?.id);
    setActiveCase(drawn);
    setFormData({ ...EMPTY, ...asStrings(caseValues(drawn)) });
    setResults(null);
    setIgnored({});
    setExpanded(null);
    setUploadedFiles([]);
    setNotice(null);
  };

  const clearData = () => {
    setFormData({ ...EMPTY });
    setActiveCase(null);
    setResults(null);
    setIgnored({});
    setExpanded(null);
    setUploadedFiles([]);
    setNotice(null);
  };

  const handleFileUpload = async (e: any) => {
    const files = Array.from(e.target.files).slice(0, 5) as File[];
    if (files.length === 0) return;

    setUploadedFiles(files.map(f => f.name));
    setParsing(true);
    setNotice(null);

    const body = new FormData();
    files.forEach(f => body.append("files", f));

    try {
      const res = await fetch(`${API_BASE}/parse-pdf`, { method: "POST", body });
      const r = await res.json();
      if (r.status === "success") {
        const found = Object.keys(r.data).length;
        setFormData(prev => ({ ...EMPTY, ...(activeCase ? {} : prev), ...asStrings(r.data) }));
        setActiveCase(null);
        setNotice(`Read ${found} value${found === 1 ? "" : "s"} from ${files.length} file${files.length === 1 ? "" : "s"}. Check them against your report before running.`);
      } else {
        setNotice(r.message || "Nothing could be read from those documents.");
      }
    } catch {
      setNotice("Could not reach the analysis server.");
    }
    setParsing(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const calculateRisk = async () => {
    const payload = Object.fromEntries(
      Object.entries(formData).filter(([, v]) => v !== "" && v !== null)
    );
    if (Object.keys(payload).length === 0) {
      setNotice("Enter some values, upload a report, or generate a case first.");
      return;
    }

    setLoading(true);
    setExpanded(null);
    setNotice(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.status === "error") {
        setNotice(data.message);
      } else {
        setResults(data.predictions);
        setIgnored(data.ignored || {});
      }
    } catch {
      setNotice("Analysis failed. The server may still be starting up, so try again in a moment.");
    }
    setLoading(false);
  };

  const bandStyle = (risk: number, isBenign: boolean) => {
    if (isBenign) {
      return risk >= 50
        ? { text: "text-emerald-400", bg: "bg-emerald-500", label: "Nothing flagged" }
        : { text: "text-slate-400", bg: "bg-slate-500", label: "Outranked by a cancer panel" };
    }
    if (risk < 20) return { text: "text-emerald-400", bg: "bg-emerald-500", label: "Low" };
    if (risk < 50) return { text: "text-amber-400", bg: "bg-amber-500", label: "Moderate" };
    return { text: "text-red-500", bg: "bg-red-500", label: "High" };
  };

  const sortedResults = results ? Object.entries(results) : [];
  const filled = Object.values(formData).filter(v => v !== "").length;
  const metrics = registry
    ? Object.fromEntries(Object.entries(registry).map(([k, v]: any) => [k, { label: v.label, ...v.metrics }]))
    : FALLBACK_METRICS;
  const metricRows = Object.entries(metrics).sort((a: any, b: any) => b[1].auc - a[1].auc);

  const navItems = [
    { id: "dashboard", label: "Assessment", icon: LayoutDashboard },
    { id: "guide", label: "How To Use", icon: BookOpen },
    { id: "about", label: "About & Methodology", icon: Info },
    { id: "developer", label: "Developer", icon: Code2 },
  ] as const;

  return (
    <div className="flex h-screen bg-[#020617] text-slate-50 font-sans overflow-hidden">

      {/* SIDEBAR */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex-col hidden md:flex">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center gap-3 mb-1">
            <OncovisionLogo className="w-8 h-8 text-cyan-500" />
            <h1 className="text-2xl font-black text-white">Oncovision <span className="text-cyan-500">AI</span></h1>
          </div>
          <p className="text-slate-400 text-[10px] tracking-widest uppercase font-bold">Computational Oncology</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setCurrentPage(id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-bold transition-all duration-300 ${currentPage === id ? "bg-cyan-500/10 text-cyan-400 shadow-[inset_0_0_0_1px_rgba(6,182,212,0.3)]" : "text-slate-400 hover:bg-slate-800/50 hover:text-white"}`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" /> {label}
            </button>
          ))}
        </nav>

        <button
          onClick={() => setCurrentPage("developer")}
          className="p-4 m-4 bg-slate-950 border border-slate-800 rounded-xl hover:border-cyan-500/30 transition-colors duration-300 text-left"
        >
          <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Lead Developer</p>
          <p className="text-white font-black text-sm">Palash Rakshit</p>
          <p className="text-[10px] text-cyan-400 font-bold mt-2">View developer info</p>
        </button>
      </aside>

      {/* MOBILE NAV */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-slate-900 border-t border-slate-800 flex">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setCurrentPage(id)}
            className={`flex-1 flex flex-col items-center gap-1 py-3 text-[9px] font-bold uppercase tracking-wide ${currentPage === id ? "text-cyan-400" : "text-slate-500"}`}
          >
            <Icon className="w-4 h-4" />
            {label.split(" ")[0]}
          </button>
        ))}
      </div>

      <main className="flex-1 overflow-y-auto relative custom-scrollbar flex flex-col pb-16 md:pb-0">

        {/* ================= ASSESSMENT ================= */}
        {currentPage === "dashboard" && (
          <div className="p-6 lg:p-10 max-w-[1600px] mx-auto animate-in fade-in duration-500 flex-1 w-full">
            <header className="mb-8">
              <h2 className="text-3xl font-bold text-white">Patient Assessment</h2>
              <p className="text-slate-400 text-sm mt-1">
                Your lab reports and information about you, read together. Upload a report or fill the panel in
                by hand, answer the history questions, then run the analysis.
              </p>
            </header>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">

              {/* INPUT */}
              <div className="xl:col-span-5 flex flex-col gap-4">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <div className="flex justify-between items-start gap-4 mb-4">
                    <div>
                      <h3 className="text-base font-bold text-slate-100">Patient Data</h3>
                      <p className="text-[11px] text-slate-500 mt-1">
                        Drop in a PDF, type values, or mix both. Every field is optional.
                      </p>
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <button
                        onClick={nextCase}
                        title="Load the next sample case"
                        className="flex items-center gap-1.5 text-[10px] font-bold text-cyan-400 hover:text-cyan-300 bg-cyan-500/10 border border-cyan-500/30 px-2.5 py-1.5 rounded transition-colors"
                      >
                        <Shuffle className="w-3 h-3" /> Generate case
                      </button>
                      <button
                        onClick={clearData}
                        className="text-[10px] font-bold text-slate-400 hover:text-red-400 bg-slate-950 border border-slate-800 px-2.5 py-1.5 rounded transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  {activeCase && (
                    <div className="mb-5 p-4 rounded-lg bg-cyan-500/5 border border-cyan-500/20">
                      <div className="flex items-center justify-between gap-3 mb-2">
                        <p className="text-xs font-black text-cyan-300 uppercase tracking-wider">
                          Sample case: {activeCase.domain} panel
                        </p>
                        <span className={`text-[9px] font-bold uppercase tracking-wider flex-shrink-0 px-2 py-1 rounded ${activeCase.positive ? "bg-red-500/10 text-red-300" : "bg-emerald-500/10 text-emerald-300"}`}>
                          Expect: {activeCase.expect}
                        </span>
                      </div>
                      <p className="text-[11px] text-cyan-100/70 leading-relaxed">{activeCase.note}</p>
                      <p className="text-[10px] text-slate-500 mt-2 italic">
                        Drawn at random from {CASE_POOL.length} real records. Source: {activeCase.source}
                      </p>
                    </div>
                  )}

                  {notice && (
                    <div className="mb-5 p-3 rounded-lg bg-amber-500/5 border border-amber-500/30 flex items-start gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                      <p className="text-[11px] text-amber-200/80 leading-relaxed">{notice}</p>
                    </div>
                  )}

                  {/* UPLOAD */}
                  <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-2 flex items-center gap-2">
                    <UploadCloud className="w-3.5 h-3.5 text-cyan-500" /> Import from a lab report
                  </p>
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-700 hover:border-cyan-500 bg-slate-950 rounded-lg p-5 text-center cursor-pointer transition-colors group"
                  >
                    <input type="file" multiple ref={fileInputRef} className="hidden" accept=".pdf" onChange={handleFileUpload} />
                    {parsing
                      ? <Activity className="animate-spin mx-auto w-7 h-7 text-cyan-500" />
                      : <FileText className="mx-auto w-7 h-7 text-slate-500 group-hover:text-cyan-400 transition-colors" />}
                    <p className="mt-2 text-sm text-cyan-400 font-bold">
                      {parsing ? "Reading documents" : "Select PDF documents"}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-1">Up to 5 files, text based PDFs only</p>
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="mt-3 p-3 bg-slate-950 rounded border border-slate-800">
                      <p className="text-[10px] text-slate-500 uppercase font-bold mb-2">Files read</p>
                      <div className="space-y-2">
                        {uploadedFiles.map((f, i) => (
                          <div key={i} className="flex items-center gap-2 text-xs text-slate-300">
                            <FileCheck className="w-3 h-3 text-emerald-500" />
                            <span className="truncate">{f}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* LAB VALUES */}
                  <div className="mt-6 pt-5 border-t border-slate-800">
                    <div className="flex justify-between items-center mb-3">
                      <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest flex items-center gap-2">
                        <Beaker className="w-3.5 h-3.5 text-cyan-500" /> Lab values
                      </p>
                      <span className="text-[10px] font-bold text-slate-500">{filled} of {ALL_KEYS.length} filled</span>
                    </div>

                    <div className="grid grid-cols-2 gap-x-3 gap-y-2 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar">
                      {LAB_KEYS.map(key => (
                        <div key={key} className={`p-2 rounded bg-slate-950 border transition-colors focus-within:border-cyan-500 ${formData[key] !== "" ? "border-cyan-500/50" : "border-slate-800"}`}>
                          <label htmlFor={`f-${key}`} className="text-[10px] text-slate-400 block mb-1 uppercase tracking-wider truncate">
                            {key.replace(/_/g, " ")}
                          </label>
                          <input
                            id={`f-${key}`}
                            type="number"
                            value={formData[key]}
                            onChange={e => setField(key, e.target.value)}
                            className="w-full bg-transparent text-white font-bold outline-none text-sm placeholder-slate-700"
                            placeholder="-"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* HISTORY */}
                  <div className="mt-6 pt-5 border-t border-slate-800">
                    <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1 flex items-center gap-2">
                      <ClipboardList className="w-3.5 h-3.5 text-cyan-500" /> Patient history
                    </p>
                    <p className="text-[10px] text-slate-600 mb-3">
                      Information about you that is not printed on a lab report.
                    </p>

                    <div className="grid grid-cols-2 gap-x-3 gap-y-2 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar">
                      {HISTORY_FIELDS.map(f => (
                        <div key={f.key} className={`p-2 rounded bg-slate-950 border transition-colors focus-within:border-cyan-500 ${formData[f.key] !== "" ? "border-cyan-500/50" : "border-slate-800"}`} title={f.meaning}>
                          <label htmlFor={`f-${f.key}`} className="text-[10px] text-slate-400 block mb-1 uppercase tracking-wider truncate">
                            {f.label}
                          </label>
                          {f.type === "select" ? (
                            <select
                              id={`f-${f.key}`}
                              value={formData[f.key]}
                              onChange={e => setField(f.key, e.target.value)}
                              className="w-full bg-transparent text-white font-bold outline-none text-sm [&>option]:bg-slate-900"
                            >
                              <option value="">-</option>
                              {f.options!.map(o => (
                                <option key={o.value} value={o.value}>{o.label}</option>
                              ))}
                            </select>
                          ) : (
                            <div className="flex items-baseline gap-1">
                              <input
                                id={`f-${f.key}`}
                                type="number"
                                min={f.min} max={f.max} step={f.step}
                                value={formData[f.key]}
                                onChange={e => setField(f.key, e.target.value)}
                                className="w-full bg-transparent text-white font-bold outline-none text-sm placeholder-slate-700"
                                placeholder="-"
                              />
                              <span className="text-[9px] text-slate-600 flex-shrink-0">{f.suffix}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <button
                  onClick={calculateRisk}
                  disabled={loading}
                  className="w-full py-5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-60 text-white font-black text-lg rounded-xl transition-all duration-300 transform hover:-translate-y-1 flex justify-center items-center gap-3 shadow-[0_0_15px_rgba(8,145,178,0.4)]"
                >
                  {loading ? <Activity className="animate-spin w-6 h-6" /> : <Scan className="w-6 h-6" />}
                  {loading ? "ANALYZING" : "RUN ANALYSIS"}
                </button>
              </div>

              {/* RESULTS */}
              <div className="xl:col-span-7">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 min-h-[600px] h-full overflow-y-auto custom-scrollbar">
                  <h3 className="text-lg font-bold text-slate-200 mb-6">Assessment Report</h3>

                  {!results ? (
                    <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 text-sm py-20 gap-3">
                      <Scan className="w-10 h-10 text-slate-700" />
                      <p>Run the analysis to see the report.</p>
                      <button onClick={() => setCurrentPage("guide")} className="text-cyan-500 hover:text-cyan-400 text-xs font-bold underline underline-offset-4">
                        First time here? Read the guide
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {Object.keys(ignored).length > 0 && (
                        <div className="flex items-start gap-3 p-4 rounded-lg bg-amber-500/5 border border-amber-500/30">
                          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="text-xs font-bold text-amber-300 mb-1">Some values were excluded</p>
                            <p className="text-[11px] text-amber-200/70 leading-relaxed">
                              These sat outside the range a living patient can have, so they were dropped before scoring:{" "}
                              {Object.entries(ignored).map(([k, why]) => (
                                <span key={k} className="font-mono">{k.replace(/_/g, " ")} ({why}) </span>
                              ))}
                            </p>
                          </div>
                        </div>
                      )}

                      {sortedResults.map(([name, d]: any, index: number) => {
                        const isBenign = d.key === "benign";
                        const style = bandStyle(d.risk, isBenign);
                        const open = expanded === name;

                        let box = "border-slate-800 p-5";
                        let head = "text-lg";
                        let num = "text-4xl";
                        if (index === 0 && !isBenign && d.risk >= 50) {
                          box = "border-red-500 border-2 shadow-[0_0_30px_rgba(239,68,68,0.35)] p-8";
                          head = "text-2xl text-red-400";
                          num = "text-6xl";
                        } else if (index === 0 && isBenign && d.risk >= 50) {
                          box = "border-emerald-500 border-2 shadow-[0_0_30px_rgba(16,185,129,0.3)] p-8";
                          head = "text-2xl text-emerald-400";
                          num = "text-6xl";
                        } else if (index === 1 && !isBenign && d.risk >= 50) {
                          box = "border-orange-500 border-2 shadow-[0_0_20px_rgba(249,115,22,0.25)] p-6";
                          head = "text-xl text-orange-400";
                          num = "text-5xl";
                        }

                        return (
                          <div key={name} className={`bg-slate-950 rounded-xl transition-all duration-500 ${box}`}>
                            <div className="flex justify-between items-start gap-4 mb-3">
                              <div>
                                <span className={`text-white font-black block ${head}`}>{name}</span>
                                <span className={`text-xs font-bold mt-1 inline-block px-2 py-1 rounded bg-slate-900 ${style.text}`}>
                                  {style.label}
                                </span>
                              </div>
                              <span className={`font-black flex-shrink-0 ${num} ${style.text}`}>{d.risk}%</span>
                            </div>

                            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden mb-4">
                              <div className={`h-full ${style.bg} transition-all duration-1000`} style={{ width: `${d.risk}%` }} />
                            </div>

                            {d.flags?.length > 0 && (
                              <div className="flex flex-wrap gap-2 mb-3">
                                {d.flags.map((f: any, i: number) => (
                                  <span key={i} className="text-[10px] font-bold px-2 py-1 rounded bg-red-500/10 text-red-300 border border-red-500/30">
                                    {f.label}
                                  </span>
                                ))}
                              </div>
                            )}

                            {!isBenign && (
                              <>
                                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                                  <span>
                                    AUC {d.auc}
                                    {d.auc_ci && <span className="text-slate-600 normal-case"> (95% CI {d.auc_ci[0]} to {d.auc_ci[1]})</span>}
                                  </span>
                                  <span>Sens {d.sensitivity}</span>
                                  <span>Spec {d.specificity}</span>
                                  <span>{d.inputs_used} of {d.inputs_total} inputs supplied</span>
                                </div>

                                {/* The number that decides whether this is usable. */}
                                {d.ppv_at_population_prevalence != null && (
                                  <div className="mt-3 p-3 rounded-lg bg-amber-500/5 border border-amber-500/25">
                                    <p className="text-[10px] font-black text-amber-300 uppercase tracking-wider mb-1">
                                      What this score is worth in the real world
                                    </p>
                                    <p className="text-[11px] text-amber-100/70 leading-relaxed">
                                      This panel was measured on a cohort that was{" "}
                                      <strong className="text-amber-200">{(d.cohort_prevalence * 100).toFixed(0)}% positive</strong>.
                                      Actual incidence is{" "}
                                      <strong className="text-amber-200">{(d.population_prevalence * 100).toFixed(4)}%</strong>.
                                      At that rate a positive here is correct about{" "}
                                      <strong className="text-amber-200">{(d.ppv_at_population_prevalence * 100).toFixed(2)}%</strong>{" "}
                                      of the time, meaning roughly{" "}
                                      <strong className="text-amber-200">{d.people_flagged_per_true_case} people flagged for every real case</strong>.
                                    </p>
                                    {d.cohort_design && (
                                      <p className="text-[10px] text-slate-500 mt-2 italic">{d.cohort_design}</p>
                                    )}
                                  </div>
                                )}
                              </>
                            )}
                            {isBenign && d.note && (
                              <p className="text-[10px] text-slate-500 leading-relaxed">{d.note}</p>
                            )}

                            {(d.drivers?.length > 0 || d.contributors?.length > 0) && (
                              <button
                                onClick={() => setExpanded(open ? null : name)}
                                className="w-full flex items-center justify-center gap-2 py-2 mt-4 text-xs font-bold text-slate-400 bg-slate-900 rounded hover:bg-slate-800 transition-colors"
                              >
                                {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                {open ? "Hide breakdown" : "Show what drove this score"}
                              </button>
                            )}

                            {open && (
                              <div className="mt-6 pt-6 border-t border-slate-800 space-y-6">
                                {/* Per-patient SHAP. Unlike the global importance
                                    list below, this explains THIS score. */}
                                {d.shap?.length > 0 && (
                                  <div>
                                    <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-1">
                                      What moved your score
                                    </p>
                                    <p className="text-[10px] text-slate-600 mb-3">
                                      SHAP values for this specific prediction, not an average across patients.
                                    </p>
                                    <div className="space-y-1.5">
                                      {d.shap.map((s: any, i: number) => {
                                        const up = s.direction === "raises";
                                        return (
                                          <div key={i} className="flex items-center gap-3 text-xs">
                                            <span className="w-32 flex-shrink-0 text-slate-300 font-bold truncate">{s.name}</span>
                                            <span className={`w-16 flex-shrink-0 font-bold ${up ? "text-red-400" : "text-emerald-400"}`}>
                                              {up ? "raises" : "lowers"}
                                            </span>
                                            <div className="flex-1 h-2 bg-slate-900 rounded-full overflow-hidden flex">
                                              <div
                                                className={`h-full ${up ? "bg-red-500/80" : "bg-emerald-500/80"}`}
                                                style={{ width: `${Math.min(s.share, 100)}%` }}
                                              />
                                            </div>
                                            <span className="w-12 text-right text-slate-400 font-mono flex-shrink-0">{s.share}%</span>
                                          </div>
                                        );
                                      })}
                                    </div>
                                    <p className="text-[10px] text-slate-600 mt-3 leading-relaxed">
                                      Share of this prediction attributable to each value you supplied. Both
                                      ensemble members are explained separately and normalised before averaging,
                                      because XGBoost reports in log odds and Extra Trees in probability.
                                      Values you left blank are excluded, since a contribution from an imputed
                                      median describes the training set rather than you.
                                    </p>
                                  </div>
                                )}

                                {d.drivers?.length > 0 && (
                                  <div>
                                    <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-3">
                                      Inputs this model relied on
                                    </p>
                                    <div className="space-y-1.5">
                                      {d.drivers.map((dr: any, i: number) => (
                                        <div key={i} className="flex items-center gap-3 text-xs">
                                          <span className="w-32 flex-shrink-0 text-slate-300 font-bold truncate">{dr.name}</span>
                                          <span className={`w-28 flex-shrink-0 font-mono ${dr.abnormal ? "text-red-400" : "text-slate-400"}`}>
                                            {dr.reading}
                                          </span>
                                          <div className="flex-1 h-1.5 bg-slate-900 rounded-full overflow-hidden">
                                            <div className="h-full bg-cyan-500/70" style={{ width: `${Math.min(dr.weight * 2.5, 100)}%` }} />
                                          </div>
                                          <span className="w-10 text-right text-slate-500 font-mono flex-shrink-0">{dr.weight}%</span>
                                        </div>
                                      ))}
                                    </div>
                                    <p className="text-[10px] text-slate-600 mt-3 leading-relaxed">
                                      Weights are the model&apos;s learned feature importance, not a breakdown of your score.
                                    </p>
                                  </div>
                                )}

                                {d.contributors?.length > 0 && (
                                  <div className={`grid grid-cols-1 ${d.contributors.length >= 3 ? "md:grid-cols-2" : ""} gap-6`}>
                                    {d.contributors.length >= 3 && (
                                      <div className="h-56">
                                        <p className="text-[10px] text-slate-400 uppercase tracking-widest text-center mb-2">Marker profile</p>
                                        <ResponsiveContainer width="100%" height="100%">
                                          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={d.contributors}>
                                            <PolarGrid stroke="#334155" />
                                            <PolarAngleAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 8 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, "auto"]} tick={false} axisLine={false} />
                                            <Radar dataKey="impact" stroke="#ef4444" fill="#ef4444" fillOpacity={0.35} />
                                            <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: "8px", fontSize: "12px" }} />
                                          </RadarChart>
                                        </ResponsiveContainer>
                                      </div>
                                    )}

                                    <div className="h-56">
                                      <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-2">Patient value against reference limit</p>
                                      <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={d.contributors} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                                          <XAxis type="number" hide />
                                          <YAxis dataKey="name" type="category" width={95} tick={{ fontSize: 9, fill: "#94a3b8" }} />
                                          <Tooltip
                                            cursor={{ fill: "rgba(255,255,255,0.05)" }}
                                            content={({ active, payload }: any) => {
                                              if (active && payload?.length) {
                                                const p = payload[0].payload;
                                                return (
                                                  <div className="bg-slate-800 border border-slate-700 p-3 rounded-lg shadow-xl">
                                                    <p className="text-white font-bold text-xs mb-1">{p.name}</p>
                                                    <p className={p.over ? "text-red-400 font-bold" : "text-cyan-400 font-bold"}>
                                                      Patient value {p.value}
                                                    </p>
                                                    <p className="text-slate-400 text-[10px] mt-1">Normal limit {p.limit}</p>
                                                  </div>
                                                );
                                              }
                                              return null;
                                            }}
                                          />
                                          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
                                            {d.contributors.map((c: any, i: number) => (
                                              <Cell key={i} fill={c.over ? "#ef4444" : "#0ea5e9"} />
                                            ))}
                                          </Bar>
                                        </BarChart>
                                      </ResponsiveContainer>
                                    </div>
                                  </div>
                                )}

                                {d.missing?.length > 0 && (
                                  <p className="text-[10px] text-slate-500 leading-relaxed">
                                    <span className="font-bold text-slate-400">Not supplied:</span>{" "}
                                    {d.missing.join(", ")}. The training median was used for each of these, which
                                    pulls the score toward the average patient.
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================= HOW TO USE ================= */}
        {currentPage === "guide" && (
          <div className="p-6 lg:p-16 max-w-[1000px] mx-auto animate-in fade-in duration-500 flex-1 w-full">
            <header className="mb-12">
              <p className="text-cyan-400 text-[11px] font-bold uppercase tracking-[0.25em] mb-3">User Guide</p>
              <h2 className="text-4xl font-black text-white">How to use Oncovision</h2>
              <p className="text-slate-400 mt-3 text-lg leading-relaxed">
                Written for someone with no medical background. If you read one section, make it
                <span className="text-cyan-400 font-bold"> Reading your results</span> at the bottom.
              </p>
            </header>

            <div className="space-y-10 text-slate-300">

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-4">What this tool does</h3>
                <p className="leading-relaxed">
                  Oncovision works from two things: <strong className="text-white">your lab reports</strong> and{" "}
                  <strong className="text-white">information about you</strong>.
                </p>
                <p className="leading-relaxed text-slate-400 mt-4">
                  The lab side is not only a blood test. It covers whatever panels you have had run, including
                  your blood count, metabolic and liver chemistry, tumour markers, and measurements taken from
                  biopsy imaging. The second side is everything a lab report does not contain: your age, sex,
                  smoking and drinking, exercise, family history, inherited risk, and conditions such as
                  hepatitis, cirrhosis or diabetes. Several of the models lean more on that second half than on
                  the chemistry, which is why the history questions are worth answering.
                </p>
                <p className="leading-relaxed text-slate-400 mt-4">
                  A full panel gives you twenty or thirty numbers, and most look fine on their own. What is hard
                  for a person to do, and straightforward for a trained model, is to read all of them together
                  alongside your history and ask whether that combination resembles patients who turned out to
                  have cancer. Oncovision runs that comparison against five models trained on anonymised patient
                  records and returns a probability for each cancer type alongside a healthy baseline. It is a
                  second read on data you already own. It is not a diagnosis and it does not replace your doctor.
                </p>
              </section>

              <section>
                <h3 className="text-2xl font-bold text-white mb-6">The flow, start to finish</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    { icon: UploadCloud, title: "Get your data in", body: "Upload the PDF of your lab report and let the parser fill the panel, or type the values yourself. Both routes feed the same models, and you can upload a PDF and then correct anything it misread. Nothing is required, so the models work with whatever you give them." },
                    { icon: ClipboardList, title: "Answer the history questions", body: "Sex, smoking, alcohol, exercise, family history, hepatitis status, cirrhosis and diabetes. This is the information about you that no lab report contains, and it carries real weight in the scoring, so filling it in is worth the minute it takes." },
                    { icon: Scan, title: "Run the analysis", body: "Values are checked against the range a living patient can have, then scored by each model that has enough to work with. Anything impossible, such as a typo with an extra zero, is dropped and reported back rather than quietly changing your score." },
                    { icon: Layers, title: "Read the report", body: "Six cards come back sorted highest first: five cancer panels and a healthy baseline. Each one expands to show which inputs the model leaned on, how your values compare to their reference limits, and how accurate that model is." },
                  ].map(({ icon: Icon, title, body }) => (
                    <div key={title} className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="p-2 rounded-lg bg-cyan-500/10"><Icon className="w-4 h-4 text-cyan-400" /></div>
                        <h4 className="font-bold text-white text-sm">{title}</h4>
                      </div>
                      <p className="text-sm text-slate-400 leading-relaxed">{body}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                  <FileText className="text-cyan-500 w-6 h-6" /> Uploading a lab report
                </h3>
                <p className="text-slate-400 text-sm mb-6">How to get a PDF in, and what to do when it does not work.</p>

                <div className="space-y-5">
                  {[
                    { n: "1", t: "Download the PDF from your patient portal", d: "MyChart, Quest, Labcorp, or whichever portal your clinic uses. Save the real PDF. A screenshot or a photo will not work, because the parser reads text rather than pixels." },
                    { n: "2", t: "Click the upload box on the assessment page", d: "You can pick up to five PDFs at once, which helps when your blood count, metabolic panel and tumour markers were drawn on different days and came back as separate documents." },
                    { n: "3", t: "Wait for the panel to fill", d: "The parser scans for each biomarker under its various printed names, WBC or white blood, AFP or alpha fetoprotein, CA 19-9 or CA19-9, and drops the matching number into the right field. Filled fields get a cyan border." },
                    { n: "4", t: "Check the numbers before you run", d: "This step matters. Lab PDFs are laid out very differently from one another and the parser sometimes picks up a reference range instead of your result. Compare the panel against your report and type over anything that looks wrong." },
                    { n: "5", t: "Fill in the history questions", d: "The parser cannot get these from a lab report because they are not on it. Answer them yourself for a more accurate result on the general and liver panels." },
                  ].map(({ n, t, d }) => (
                    <div key={n} className="flex gap-4">
                      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-black text-xs">{n}</div>
                      <div>
                        <p className="font-bold text-white text-sm mb-1">{t}</p>
                        <p className="text-sm text-slate-400 leading-relaxed">{d}</p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-slate-800 flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-slate-400 leading-relaxed">
                    <strong className="text-amber-300">If nothing fills in:</strong> your PDF is probably a scanned
                    image rather than text. Open it and try to select a word with your cursor. If you cannot
                    highlight it, the parser cannot read it either, so type the values in by hand instead.
                  </p>
                </div>
              </section>

              <section>
                <h3 className="text-2xl font-bold text-white mb-2">What each value means</h3>
                <p className="text-slate-400 text-sm mb-6">
                  You will not have all of these, and that is fine. Leave the blanks blank. The normal ranges shown
                  are the limits Oncovision scores against, and your own lab may print slightly different ones.
                </p>

                <div className="space-y-6">
                  {LAB_GROUPS.map(({ group, blurb, items }) => (
                    <div key={group} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                      <div className="p-5 border-b border-slate-800 bg-slate-950/50">
                        <h4 className="font-bold text-white">{group}</h4>
                        <p className="text-xs text-slate-500 mt-1">{blurb}</p>
                      </div>
                      <div className="divide-y divide-slate-800">
                        {items.map(({ key, label, unit, normal, meaning }) => (
                          <div key={key} className="p-5">
                            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-2">
                              <span className="font-bold text-cyan-400 text-sm">{label}</span>
                              <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">{key.replace(/_/g, " ")}</span>
                              <span className="ml-auto text-[10px] font-bold text-slate-400 bg-slate-950 border border-slate-800 px-2 py-1 rounded">
                                Normal {normal} {unit}
                              </span>
                            </div>
                            <p className="text-sm text-slate-400 leading-relaxed">{meaning}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}

                  <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                    <div className="p-5 border-b border-slate-800 bg-slate-950/50">
                      <h4 className="font-bold text-white">Patient history</h4>
                      <p className="text-xs text-slate-500 mt-1">
                        Information about you, answered by you rather than read off a lab report.
                      </p>
                    </div>
                    <div className="divide-y divide-slate-800">
                      {HISTORY_FIELDS.map(f => (
                        <div key={f.key} className="p-5">
                          <div className="flex flex-wrap items-baseline gap-x-3 mb-2">
                            <span className="font-bold text-cyan-400 text-sm">{f.label}</span>
                            <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">{f.key.replace(/_/g, " ")}</span>
                          </div>
                          <p className="text-sm text-slate-400 leading-relaxed">{f.meaning}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>

              <section className="bg-slate-900 border border-cyan-500/20 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-2">Reading your results</h3>
                <p className="text-slate-400 text-sm mb-6">Six cards come back, sorted from highest score to lowest.</p>

                <div className="space-y-4">
                  <div className="bg-slate-950 border border-emerald-500/30 rounded-lg p-5">
                    <p className="font-bold text-emerald-400 text-sm mb-2">No cancer detected</p>
                    <p className="text-sm text-slate-400 leading-relaxed">
                      Your healthy baseline, calculated as the complement of the highest cancer score. When it sits
                      at the top of the list, no model found a pattern it recognises in what you entered.
                    </p>
                  </div>

                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                    <p className="font-bold text-white text-sm mb-3">The five cancer panels</p>
                    <p className="text-sm text-slate-400 leading-relaxed mb-4">
                      General, breast, liver and pancreatic. A prostate panel was built and then withdrawn
                      because it could not be shown to beat chance, which is explained on the methodology
                      page. The percentage is the model&apos;s own output,
                      and the honest way to read it is as how closely your profile matches patients in the training
                      data who had that cancer. It is not your literal odds of having it.
                    </p>
                    <div className="space-y-2">
                      {[
                        { r: "0 to 19", l: "Low", c: "text-emerald-400", b: "border-emerald-500/30", m: "Nothing in what you entered resembles that pattern." },
                        { r: "20 to 49", l: "Moderate", c: "text-amber-400", b: "border-amber-500/30", m: "Some markers are drifting. Worth raising at your next appointment." },
                        { r: "50 to 100", l: "High", c: "text-red-400", b: "border-red-500/30", m: "Your profile matches the disease pattern closely. Take the breakdown to a physician. Do not wait for symptoms, and do not panic either, because raised markers have many harmless causes." },
                      ].map(({ r, l, c, b, m }) => (
                        <div key={r} className={`flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 p-3 rounded bg-slate-900 border ${b}`}>
                          <span className={`font-black text-sm w-20 flex-shrink-0 ${c}`}>{r}%</span>
                          <span className={`text-xs font-bold w-20 flex-shrink-0 ${c}`}>{l}</span>
                          <span className="text-xs text-slate-400 leading-relaxed">{m}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                    <p className="font-bold text-white text-sm mb-2">Inside each card</p>
                    <ul className="space-y-3 text-sm text-slate-400 leading-relaxed">
                      <li><strong className="text-cyan-400">Red flag chips</strong> sit under the score when one of your markers crossed an established clinical threshold, such as PSA above 4 or CA 19-9 above 37. These are reported separately and never change the number.</li>
                      <li><strong className="text-cyan-400">AUC, sensitivity and specificity</strong> tell you how much to trust that particular panel. See the accuracy table on the methodology page.</li>
                      <li><strong className="text-cyan-400">Inputs this model relied on</strong> lists the values you supplied ranked by how much weight the model gives them, with your reading beside each. Red means outside the normal range.</li>
                      <li><strong className="text-cyan-400">The two charts</strong> show your marker profile and each value against its reference limit. Bars turn red above the limit and stay blue inside it.</li>
                      <li><strong className="text-cyan-400">Not supplied</strong> lists what the model wanted but did not get. Each missing value was filled with the training median, which pulls the score toward the average patient.</li>
                    </ul>
                  </div>

                  <div className="bg-slate-950 border border-amber-500/30 rounded-lg p-5">
                    <p className="font-bold text-amber-300 text-sm mb-2">The excluded values banner</p>
                    <p className="text-sm text-slate-400 leading-relaxed">
                      If this appears, something you entered was outside the range a living patient can have, almost
                      always a typo or a misread PDF line. That value was thrown out before scoring. Fix it and run again.
                    </p>
                  </div>
                </div>
              </section>

              <section className="bg-slate-900/50 border border-red-500/30 rounded-xl p-8">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                  <AlertTriangle className="text-red-400 w-5 h-5" /> What this tool cannot do
                </h3>
                <ul className="space-y-3 text-sm text-slate-400 leading-relaxed list-disc pl-5">
                  <li>It cannot diagnose cancer. Only a biopsy can do that.</li>
                  <li>It cannot rule cancer out. Plenty of cancers produce completely normal lab results early on, so low scores across the board are reassuring without being proof.</li>
                  <li>It cannot see past what you give it. Your lab reports and the history you enter, and nothing else. No imaging, no genetic sequencing, no symptoms, no physical exam.</li>
                  <li>It was trained on public research data that does not represent every population equally, so accuracy varies between groups.</li>
                  <li>It is a student research prototype. Read every output as a reason to ask your doctor a sharper question, never as an answer.</li>
                </ul>
              </section>
            </div>
          </div>
        )}

        {/* ================= ABOUT & METHODOLOGY ================= */}
        {currentPage === "about" && (
          <div className="p-6 lg:p-16 max-w-[1000px] mx-auto animate-in fade-in duration-500 flex-1 w-full">
            <header className="mb-12 text-center">
              <OncovisionLogo className="w-24 h-24 text-cyan-500 mx-auto mb-6 drop-shadow-[0_0_15px_rgba(6,182,212,0.4)]" />
              <h2 className="text-4xl font-black text-white">Project Oncovision</h2>
              <p className="text-cyan-400 text-lg mt-2 font-bold uppercase tracking-widest">Computational Oncology for the Public</p>
            </header>

            <div className="space-y-12 text-slate-300">

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <Target className="text-cyan-500 w-6 h-6" /> The goal
                </h3>
                <p className="leading-relaxed text-lg">
                  Make multi-cancer screening cheap enough to be routine, and make the results readable to someone
                  without a medical background.
                </p>
                <p className="leading-relaxed text-slate-400 mt-4">
                  Most patients get lab work every year and never learn what is in it beyond a flag or two. This
                  closes that gap.
                </p>
                <p className="leading-relaxed text-slate-400 mt-4">
                  Oncovision reads two things together: the lab reports you already have, and information about
                  you that no report contains. Neither half is enough on its own.
                </p>
              </section>

              {/* THE PROBLEM */}
              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                  <AlertTriangle className="text-cyan-500 w-6 h-6" /> The problem this addresses
                </h3>
                <p className="text-sm text-slate-400 mb-6 leading-relaxed">
                  Cancer screening in the United States has two gaps. It covers very few cancers, and the tests
                  that do exist cost enough that plenty of people never take them. Every figure below is sourced.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {[
                    {
                      stat: "4",
                      unit: "cancer types",
                      title: "Almost nothing is screened for",
                      body: "Only breast, cervical, colorectal and lung cancer have a screening test recommended by the US Preventive Services Task Force. Those four account for 29 percent of cancer cases. 57 percent of diagnosed cancers have no recommended screening test at all, and that group causes 70 percent of cancer deaths.",
                    },
                    {
                      stat: "14%",
                      unit: "of cancers",
                      title: "Screening catches a small share",
                      body: "Only 14 percent of cancers in the United States are diagnosed after the patient had a recommended screening test. A 2025 analysis estimated that current screening, as it is actually used, leaves as much as 87 percent of cancer deaths unaddressed.",
                    },
                    {
                      stat: "92% vs 15%",
                      unit: "five year survival",
                      title: "Timing decides the outcome",
                      body: "Across all cancers combined, five year relative survival is roughly 92 percent when the cancer is still localised and roughly 15 percent once it has spread to distant sites. The difference between those two numbers is mostly a question of when it was found.",
                    },
                    {
                      stat: "$949",
                      unit: "per test",
                      title: "Cost keeps people out",
                      body: "Multi-cancer blood tests already exist. Galleri, the best known, lists at $949 and is not covered by Medicare or most insurance. A screening colonoscopy averages about $2,750 without insurance and still runs several hundred dollars out of pocket for many who have it.",
                    },
                  ].map(({ stat, unit, title, body }) => (
                    <div key={title} className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                      <p className="text-2xl font-black text-cyan-400 leading-none">{stat}</p>
                      <p className="text-[9px] uppercase font-bold text-slate-500 tracking-widest mt-1 mb-3">{unit}</p>
                      <h4 className="font-bold text-white text-sm mb-2">{title}</h4>
                      <p className="text-sm text-slate-400 leading-relaxed">{body}</p>
                    </div>
                  ))}
                </div>

                <div className="bg-slate-950 border border-cyan-500/30 rounded-lg p-6">
                  <h4 className="font-bold text-white mb-3">What Oncovision does about it</h4>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    It adds no new test and no new cost. It works from a lab report you have already paid for,
                    plus information about yourself that you can answer in a minute, and it checks that
                    combination against five cancer models at once rather than the one or two your age and sex
                    happen to qualify you for. The marginal cost of running it is nothing, which is the entire
                    point. A test that costs $949 will not become routine. A test that reads a PDF you already
                    have in your patient portal can.
                  </p>
                  <p className="text-sm text-slate-400 leading-relaxed mt-4">
                    What it cannot do is stand in for the screening above. A colonoscopy looks at a colon and a
                    mammogram looks at breast tissue. Oncovision looks at numbers, and numbers can be normal in
                    someone who has cancer. It is built to raise a question early enough to be worth asking, not
                    to answer one.
                  </p>
                </div>

                <p className="text-[10px] text-slate-600 leading-relaxed mt-5">
                  Sources: NORC at the University of Chicago, analysis of cancers detected by screening.
                  Ofman et al., <em>Cancer Biomarkers</em>, 2025, on cancer deaths not addressed by current
                  screening. SEER five year relative survival by stage at diagnosis. Published list price for
                  Galleri and reported average colonoscopy cost, both as of 2025.
                </p>
              </section>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <Microscope className="text-cyan-500 w-6 h-6" /> How this was built
                </h3>
                <ul className="space-y-4">
                  {[
                    "Built as a formal mentored research project under the direct guidance of a clinical oncologist at UCI CHOC, ensuring the clinical data and testing parameters met professional medical standards.",
                    "Developed a full stack platform that reads standard patient lab reports and returns multi-cancer risk assessments in real time.",
                    "Engineered an automated PDF parsing system that pulls specific medical variables out of lab reports across widely varying formats.",
                    "Trained an ensemble machine learning backend on XGBoost and Extra Trees to analyze correlations between biomarkers and complex blood variables.",
                    "Deployed the full system on a FastAPI backend and Next.js frontend.",
                  ].map(line => (
                    <li key={line} className="flex gap-3 text-sm leading-relaxed text-slate-300">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mt-2 flex-shrink-0" />
                      {line}
                    </li>
                  ))}
                </ul>
              </section>

              {/* MODEL PERFORMANCE */}
              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                  <Target className="text-cyan-500 w-6 h-6" /> Measured performance
                </h3>
                <p className="text-sm text-slate-400 mb-6 leading-relaxed">
                  Every figure here comes from a 20 percent test split that was cut before any model was
                  fitted and was never used for training, model selection, or calibration. Confidence
                  intervals are bootstrap percentile intervals over 2,000 resamples. Reproduce all of it
                  with <span className="font-mono text-cyan-400">python evaluate.py</span>.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pr-4 font-bold">Panel</th>
                        <th className="pb-3 pr-4 font-bold">Test AUC</th>
                        <th className="pb-3 pr-4 font-bold">95% CI</th>
                        <th className="pb-3 pr-4 font-bold">Sens.</th>
                        <th className="pb-3 pr-4 font-bold">Spec.</th>
                        <th className="pb-3 font-bold">Test n</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {metricRows.map(([key, m]: any) => (
                        <tr key={key} className="text-slate-400">
                          <td className="py-3 pr-4 font-bold text-cyan-400 whitespace-nowrap capitalize">{key}</td>
                          <td className="py-3 pr-4 font-mono font-bold text-emerald-400">{m.auc}</td>
                          <td className="py-3 pr-4 font-mono text-slate-500">
                            {m.auc_ci ? `${m.auc_ci[0]} to ${m.auc_ci[1]}` : "n/a"}
                          </td>
                          <td className="py-3 pr-4 font-mono">{m.sensitivity}</td>
                          <td className="py-3 pr-4 font-mono">{m.specificity}</td>
                          <td className="py-3 font-mono">{m.n_test ?? "n/a"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* THE NUMBER THAT MATTERS */}
                <h4 className="text-lg font-bold text-white mt-10 mb-2">Precision once the disease is rare</h4>
                <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                  AUC above 0.95 sounds decisive, and on its own it is close to meaningless for screening.
                  Every cohort here is enriched for disease, between 21 and 37 percent positive, while real
                  incidence is a fraction of a percent. Projecting the measured sensitivity and specificity
                  onto SEER incidence gives the number that actually decides whether a tool is usable.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pr-4 font-bold">Panel</th>
                        <th className="pb-3 pr-4 font-bold">Cohort</th>
                        <th className="pb-3 pr-4 font-bold">Real incidence</th>
                        <th className="pb-3 pr-4 font-bold">PPV there</th>
                        <th className="pb-3 font-bold">Flagged per true case</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {metricRows.map(([key, m]: any) => (
                        <tr key={key} className="text-slate-400">
                          <td className="py-3 pr-4 font-bold text-cyan-400 whitespace-nowrap capitalize">{key}</td>
                          <td className="py-3 pr-4 font-mono">{(m.cohort_prevalence * 100).toFixed(0)}%</td>
                          <td className="py-3 pr-4 font-mono">{(m.population_prevalence * 100).toFixed(4)}%</td>
                          <td className={`py-3 pr-4 font-mono font-bold ${m.ppv_at_population_prevalence > 0.1 ? "text-amber-400" : "text-red-400"}`}>
                            {(m.ppv_at_population_prevalence * 100).toFixed(2)}%
                          </td>
                          <td className="py-3 font-mono font-bold text-red-400">{m.people_flagged_per_true_case}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-5 p-4 rounded-lg bg-red-500/5 border border-red-500/25">
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Read the right-hand column plainly. Used as a population screen today, the pancreatic
                    panel would flag roughly 525 people for every one who has the disease. That is not a
                    usable screening test, and no AUC figure changes it. What these models can reasonably do
                    is rank and explain values for someone who already has a reason to be asking, which is
                    why the interface reports this alongside every score rather than burying it here.
                  </p>
                </div>

                {/* BASELINES */}
                <h4 className="text-lg font-bold text-white mt-10 mb-2">Does the ensemble earn its complexity?</h4>
                <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                  A gradient boosted forest paired with an Extra Trees classifier is only worth the cost if
                  it beats something simple. Both baselines were trained and tested on the identical splits.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pr-4 font-bold">Panel</th>
                        <th className="pb-3 pr-4 font-bold">Ensemble</th>
                        <th className="pb-3 pr-4 font-bold">Logistic regression</th>
                        <th className="pb-3 pr-4 font-bold">Age and sex only</th>
                        <th className="pb-3 font-bold">Verdict</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {metricRows.map(([key, m]: any) => {
                        const lr = m.baseline_logistic_auc;
                        const gain = lr ? m.auc - lr : null;
                        const verdict =
                          gain == null ? "n/a" : gain > 0.02 ? "worth it" : "marginal";
                        return (
                          <tr key={key} className="text-slate-400">
                            <td className="py-3 pr-4 font-bold text-cyan-400 whitespace-nowrap capitalize">{key}</td>
                            <td className="py-3 pr-4 font-mono font-bold text-white">{m.auc}</td>
                            <td className="py-3 pr-4 font-mono">{lr ?? "n/a"}</td>
                            <td className="py-3 pr-4 font-mono">{m.baseline_age_sex_auc ?? "n/a"}</td>
                            <td className={`py-3 font-bold ${verdict === "worth it" ? "text-emerald-400" : "text-amber-400"}`}>
                              {verdict}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <p className="text-xs text-slate-500 mt-4 leading-relaxed">
                  Honest reading: the ensemble is clearly worth it on general and liver. On breast (0.972 against
                  0.964) and pancreatic (0.969 against 0.968) it is within noise of logistic regression, and a
                  reviewer would be right to say the simpler model should ship for those two.
                </p>

                {/* CALIBRATION */}
                <h4 className="text-lg font-bold text-white mt-10 mb-2">Calibration</h4>
                <p className="text-sm text-slate-400 mb-5 leading-relaxed">
                  The interface shows people a percentage, so the percentages have to correspond to observed
                  frequencies. Every shipped model is wrapped in isotonic calibration fitted by internal cross
                  validation. A calibration slope of 1.0 is perfect and below 1.0 means over-confident.
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pr-4 font-bold">Panel</th>
                        <th className="pb-3 pr-4 font-bold">Brier score</th>
                        <th className="pb-3 font-bold">Calibration slope</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {metricRows.map(([key, m]: any) => (
                        <tr key={key} className="text-slate-400">
                          <td className="py-3 pr-4 font-bold text-cyan-400 whitespace-nowrap capitalize">{key}</td>
                          <td className="py-3 pr-4 font-mono">{m.brier}</td>
                          <td className={`py-3 font-mono ${m.calibration_slope < 0.7 ? "text-amber-400" : ""}`}>
                            {m.calibration_slope}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="text-xs text-slate-500 mt-4 leading-relaxed">
                  The pancreatic slope of 0.46 is the weak one. Its probabilities are still over-confident
                  after calibration, so treat that panel&apos;s percentage as a ranking rather than a literal
                  likelihood.
                </p>

                {/* WITHDRAWN */}
                <h4 className="text-lg font-bold text-white mt-10 mb-2">Withdrawn panels</h4>
                <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                  Reported rather than deleted, because a panel that failed its evaluation is evidence about
                  the method.
                </p>
                {WITHDRAWN_PANELS.map(w => (
                  <div key={w.name} className="p-5 rounded-lg bg-slate-950 border border-red-500/30">
                    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-3">
                      <span className="font-black text-white">{w.name}</span>
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded bg-red-500/10 text-red-300">
                        Not served
                      </span>
                      <span className="text-[11px] font-mono text-slate-500">
                        AUC {w.auc}, 95% CI {w.ci[0]} to {w.ci[1]} · logistic {w.logistic} · spec {w.specificity} (CI {w.spec_ci[0]} to {w.spec_ci[1]}) · n={w.n}, test n={w.n_test}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 leading-relaxed">{w.reason}</p>
                  </div>
                ))}
              </section>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                  <BrainCircuit className="text-cyan-500 w-6 h-6" /> Architecture
                </h3>
                <div className="space-y-4">
                  {[
                    {
                      icon: Server,
                      title: "Ingestion and sanitisation",
                      body: "Values arrive either from the form or from the PDF endpoint, which uses pdfplumber to pull raw text and a regex synonym map to locate 23 biomarkers under the different names labs print them with. Each value is then checked against a table of biologically possible ranges. Anything impossible is dropped and returned to the interface rather than clamped, so a typo cannot quietly move a score.",
                    },
                    {
                      icon: Cpu,
                      title: "Ensemble inference",
                      body: "Five soft voting ensembles, one per domain, each combining an XGBoost gradient boosted forest with an Extra Trees classifier. XGBoost fits sequentially against its own errors while Extra Trees randomises split thresholds across independent trees, so the two make different mistakes and averaging their probabilities is steadier than either alone. Class imbalance is handled with positive class weighting inside XGBoost and balanced subsampling inside Extra Trees.",
                    },
                    {
                      icon: Layers,
                      title: "Schema alignment",
                      body: "A model may only train on features the application can collect. Columns the interface never asks for are dropped before fitting rather than left in, because a model trained on inputs it will never receive reports an accuracy that does not describe how it performs in use. Each bundle stores its own feature list and the training median of every feature, and values a patient does not supply are filled with that median instead of a zero.",
                    },
                    {
                      icon: ShieldCheck,
                      title: "Clinical thresholds",
                      body: "Established decision limits, PSA at 4.0, CA 19-9 at 37, AFP at 10, bilirubin at 1.2 and the rest, are evaluated separately and surfaced as flags on the relevant panel. They are reported next to the model output and never overwrite it, so what you see is the model's own probability rather than a rule wearing a probability's clothes.",
                    },
                    {
                      icon: Target,
                      title: "Attribution",
                      body: "Feature importance is averaged across both members of each ensemble and returned with every prediction, ranked and paired with the patient's own reading and the clinical limit for that marker. This is a ranking of what the model leans on, not an additive decomposition of the score.",
                    },
                  ].map(({ icon: Icon, title, body }) => (
                    <div key={title} className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                      <div className="flex items-center gap-3 mb-3">
                        <Icon className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                        <h4 className="font-bold text-white text-sm">{title}</h4>
                      </div>
                      <p className="text-sm text-slate-400 leading-relaxed">{body}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <Microscope className="text-cyan-500 w-6 h-6" /> Training data
                </h3>
                <p className="text-sm leading-relaxed text-slate-400 mb-6">
                  Every model was trained on de-identified, publicly released clinical research data. Categorical
                  columns were translated into a single shared encoding so that a smoking answer means the same
                  thing to every model, and missing numerics were filled with the column median before fitting.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pr-4 font-bold">Panel</th>
                        <th className="pb-3 pr-4 font-bold">Source</th>
                        <th className="pb-3 font-bold">What counts as positive</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {[
                        ["Liver", "Hepatocellular cohort, 5,000 records", "A liver cancer diagnosis"],
                        ["General", "Cancer risk cohort, 1,500 records", "A recorded cancer diagnosis"],
                        ["Pancreatic", "Pancreatic biomarker cohort, 600 records", "Confirmed adenocarcinoma, separated from both healthy controls and benign hepatobiliary disease"],
                        ["Breast", "Wisconsin Diagnostic Breast Cancer, 569 records", "A malignant fine needle aspirate"],
                        ["Prostate", "Stanford prostate cohort, 97 records", "Gleason score of 7 or above"],
                      ].map(([a, b, c]) => (
                        <tr key={a} className="text-slate-400">
                          <td className="py-3 pr-4 font-bold text-cyan-400 whitespace-nowrap">{a}</td>
                          <td className="py-3 pr-4">{b}</td>
                          <td className="py-3">{c}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="bg-slate-900 border border-amber-500/20 rounded-xl p-8">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                  <AlertTriangle className="text-amber-400 w-5 h-5" /> Known limitations
                </h3>
                <ul className="space-y-3 text-sm text-slate-400 leading-relaxed list-disc pl-5">
                  <li><strong className="text-white">Every cohort is case-control, not a screening series.</strong> These records come from people who already had a reason to be tested, so the cohorts run 21 to 37 percent positive against a real incidence measured in hundredths of a percent. That gap is why the precision table above matters more than the AUC table.</li>
                  <li><strong className="text-white">The breast panel contradicts the schema rule.</strong> Its four inputs are nuclear morphology from a fine needle aspirate, which requires a biopsy that has already happened. It interprets a biopsy rather than screening for one, and calling it a screening panel would be wrong.</li>
                  <li><strong className="text-white">The liver cohort is synthetic.</strong> It is the largest dataset here at 5,000 records and posts the joint-highest AUC, and the records are generated rather than observed. That AUC describes a generator, not a patient population.</li>
                  <li><strong className="text-white">No external validation.</strong> Every number comes from a held-out split of the same cohort the model trained on. Nothing here has been tested against a dataset collected somewhere else, which is the single largest gap.</li>
                  <li><strong className="text-white">No prospective test and no IRB.</strong> No real patient report has been run through this and followed to an outcome. There is no ethics approval, no registration, and no clinical validation of any kind.</li>
                  <li><strong className="text-white">The ensemble is within noise of logistic regression on two panels.</strong> Breast at 0.972 against 0.964, pancreatic at 0.969 against 0.968. The added complexity is not clearly earning its place there.</li>
                  <li><strong className="text-white">Subgroup coverage is thin.</strong> AUC is broken out by sex and age band where the test split allows, but the cohorts carry no race or ethnicity, so accuracy across those groups is unmeasured rather than acceptable.</li>
                  <li>The source datasets do not share a schema, so each panel sees a different slice of what you enter. A model scores only when it receives at least one real value, and every card reports how many of its inputs you supplied.</li>
                </ul>
              </section>

              <section className="bg-slate-900/50 border border-emerald-500/30 rounded-xl p-8">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
                  <ShieldCheck className="text-emerald-400 w-6 h-6" /> Data handling
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Uploaded PDFs are read into memory, parsed, and discarded inside the request. Nothing is written
                  to disk and no database is attached to the service. Your values live in browser state for the
                  length of your session and are gone when you close the tab.
                </p>
              </section>
            </div>
          </div>
        )}

        {/* ================= DEVELOPER ================= */}
        {currentPage === "developer" && (
          <div className="p-6 lg:p-16 max-w-[1000px] mx-auto animate-in fade-in duration-500 flex-1 w-full">
            <header className="mb-12">
              <p className="text-cyan-400 text-[11px] font-bold uppercase tracking-[0.25em] mb-3">Developer</p>
              <h2 className="text-4xl font-black text-white">Behind the project</h2>
            </header>

            <div className="space-y-8 text-slate-300">
              <div className="bg-gradient-to-r from-slate-900 to-slate-950 border border-slate-800 rounded-xl p-8 flex flex-col items-center shadow-xl">
                <p className="text-[10px] text-slate-500 uppercase font-bold mb-1 tracking-widest">Founder and Lead Developer</p>
                <p className="text-white font-black text-3xl mb-4">Palash Rakshit</p>
                <p className="text-center text-slate-400 text-sm max-w-2xl mb-8 leading-relaxed">
                  Palash is a developer and researcher working at the intersection of biomedical engineering and
                  machine learning. He founded and leads the <strong className="text-white">MedTech Club</strong> at
                  RHS, which supports 100 students across the district. He is also a{" "}
                  <strong className="text-white">National Debater</strong>. He built Oncovision to close the distance
                  between complex medical data and the people it describes.
                </p>

                <div className="flex flex-col md:flex-row flex-wrap gap-4 w-full justify-center">
                  <a href="mailto:palash.raks@gmail.com" className="flex items-center gap-3 bg-slate-950 p-4 rounded-lg border border-slate-800 hover:border-cyan-500/50 transition-colors">
                    <Mail className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                    <div className="flex flex-col">
                      <span className="text-[9px] uppercase font-bold text-slate-500">Developer contact</span>
                      <span className="font-bold text-xs">palash.raks@gmail.com</span>
                    </div>
                  </a>
                  <a href="mailto:medtechcentral@gmail.com" className="flex items-center gap-3 bg-slate-950 p-4 rounded-lg border border-slate-800 hover:border-cyan-400/50 transition-colors">
                    <Globe className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                    <div className="flex flex-col">
                      <span className="text-[9px] uppercase font-bold text-slate-500">Project inquiries</span>
                      <span className="font-bold text-xs">medtechcentral@gmail.com</span>
                    </div>
                  </a>
                  <a href="https://www.linkedin.com/in/Palash-Rakshit10" target="_blank" rel="noopener noreferrer" className="flex items-center gap-3 bg-slate-950 p-4 rounded-lg border border-slate-800 hover:border-blue-500/50 transition-colors">
                    <Linkedin className="w-5 h-5 text-blue-400 flex-shrink-0" />
                    <div className="flex flex-col">
                      <span className="text-[9px] uppercase font-bold text-slate-500">Professional profile</span>
                      <span className="font-bold text-xs">Palash-Rakshit10</span>
                    </div>
                  </a>
                </div>
              </div>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                  <GitBranch className="text-cyan-500 w-5 h-5" /> Stack
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { title: "Frontend", items: ["Next.js 16, App Router", "React 19", "TypeScript", "Tailwind CSS v4", "Recharts", "Lucide icons", "Vercel"] },
                    { title: "Backend", items: ["FastAPI", "Uvicorn", "Pydantic", "pdfplumber", "pandas", "Render"] },
                    { title: "Machine learning", items: ["XGBoost", "scikit-learn Extra Trees", "Soft voting ensemble x5", "Stratified 5 fold CV", "joblib serialisation", "NumPy"] },
                  ].map(({ title, items }) => (
                    <div key={title} className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                      <p className="text-[10px] uppercase font-bold text-cyan-400 tracking-widest mb-3">{title}</p>
                      <ul className="space-y-2">
                        {items.map(i => (
                          <li key={i} className="text-xs text-slate-400 flex items-start gap-2">
                            <span className="w-1 h-1 rounded-full bg-slate-700 mt-1.5 flex-shrink-0" />{i}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                  <Code2 className="text-cyan-500 w-5 h-5" /> Repository
                </h3>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-5 font-mono text-xs text-slate-400 overflow-x-auto">
                  <div className="whitespace-pre">{`oncovision/
├── backend/
│   ├── api.py              FastAPI service
│   ├── models/             5 serialised ensemble bundles
│   ├── model_metrics.json  measured performance
│   └── requirements.txt
├── data/                   source training datasets
├── frontend/
│   └── src/app/
│       ├── page.tsx        all four views
│       ├── fields.ts       input schema and glossary
│       ├── cases.ts        the six sample cases
│       └── layout.tsx
└── train_models.py         training pipeline`}</div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                    <p className="text-[10px] uppercase font-bold text-cyan-400 tracking-widest mb-3">Endpoints</p>
                    <div className="space-y-3 text-xs">
                      <div>
                        <p className="font-mono text-emerald-400 font-bold">POST /predict</p>
                        <p className="text-slate-500 mt-1">A flat object of lab values and history, every field optional. Returns a ranked per panel assessment with attribution and model metrics.</p>
                      </div>
                      <div>
                        <p className="font-mono text-emerald-400 font-bold">POST /parse-pdf</p>
                        <p className="text-slate-500 mt-1">Up to 5 PDFs in, extracted biomarker values out.</p>
                      </div>
                      <div>
                        <p className="font-mono text-emerald-400 font-bold">GET /models</p>
                        <p className="text-slate-500 mt-1">The model registry with measured AUC, sensitivity and specificity.</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-5">
                    <p className="text-[10px] uppercase font-bold text-cyan-400 tracking-widest mb-3">Running locally</p>
                    <div className="font-mono text-[11px] text-slate-400 space-y-1 leading-relaxed">
                      <p className="text-slate-600"># train</p>
                      <p>pip install -r requirements.txt</p>
                      <p>python train_models.py</p>
                      <p className="text-slate-600 pt-2"># serve</p>
                      <p>cd backend</p>
                      <p>uvicorn api:app --reload --port 8000</p>
                      <p className="text-slate-600 pt-2"># frontend</p>
                      <p>cd frontend</p>
                      <p>npm install</p>
                      <p>npm run dev</p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        )}

        <footer className="p-8 border-t border-slate-800 bg-slate-950/80 backdrop-blur-sm mt-auto">
          <div className="max-w-4xl mx-auto text-center space-y-3">
            <p className="text-slate-500 text-[11px] leading-relaxed italic">
              <strong>Disclaimer:</strong> Oncovision is a diagnostic support prototype built for educational and
              research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.
              Always seek the advice of your physician or another qualified health provider with any question you
              have about a medical condition.
            </p>
            <div className="flex justify-center items-center gap-4 text-slate-600 text-[9px] uppercase tracking-[0.2em]">
              <span>2026 Project Oncovision</span>
              <span className="w-1 h-1 bg-slate-800 rounded-full" />
              <span>Redlands, CA</span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
