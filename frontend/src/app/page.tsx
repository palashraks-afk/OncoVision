"use client";

import { useState, useRef } from "react";
import { Activity, BrainCircuit, Scan, UploadCloud, ShieldAlert, FileText, ChevronDown, ChevronUp, LayoutDashboard, FileCheck, Info, Mail, Globe, ShieldCheck, HeartPulse, Eye } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

// --- CUSTOM SVG LOGO ---
const OncovisionLogo = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 100 100" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 50C10 50 25 20 50 20C75 20 90 50 90 50C90 50 75 80 50 80C25 80 10 50 10 50Z" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="50" cy="50" r="10" stroke="currentColor" strokeWidth="4" fill="#020617"/>
    <circle cx="50" cy="50" r="3" fill="currentColor"/>
    <path d="M25 40L50 50M25 60L50 50M75 40L50 50M75 60L50 50" stroke="currentColor" strokeWidth="2" strokeDasharray="3 3"/>
  </svg>
);

export default function OncovisionDashboard() {
  // Navigation State
  const [currentPage, setCurrentPage] = useState<"dashboard" | "about">("dashboard");
  
  const [loading, setLoading] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [ignored, setIgnored] = useState<Record<string, string>>({});
  const [expandedGraph, setExpandedGraph] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [formData, setFormData] = useState<Record<string, string | number>>({
    age: "", bmi: "", wbc: "", rbc: "", hemoglobin: "", platelets: "", glucose: "", calcium: "", bun: "", creatinine: "", protein_total: "", albumin: "",
    ast: "", alt: "", bilirubin: "", alkaline_phosphatase: "", alpha_fetoprotein_level: "", psa: "", plasma_CA19_9: "", 
    radius_mean: "", texture_mean: "", perimeter_mean: "", area_mean: ""
  });

  const handleInputChange = (e: any) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleFileUpload = async (e: any) => {
    const files = Array.from(e.target.files).slice(0, 5) as File[];
    if (files.length === 0) return;
    
    setUploadedFiles(files.map(f => f.name));
    setParsing(true);
    
    const body = new FormData();
    files.forEach(file => body.append("files", file));

    try {
      const res = await fetch("http://localhost:8000/parse-pdf", { method: "POST", body });
      const r = await res.json();
      if (r.status === "success") {
        setFormData((prev: any) => ({ ...prev, ...r.data }));
      } else {
        alert(r.message);
      }
    } catch (err) { alert("Server connection failed."); }
    setParsing(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const calculateRisk = async () => {
    const hasData = Object.values(formData).some(val => val !== "" && Number(val) > 0);
    if (!hasData) {
      alert("Please enter patient data or upload a lab report.");
      return;
    }

    setLoading(true);
    setExpandedGraph(null);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lab_values: formData }),
      });
      const data = await res.json();
      
      if (data.status === "error") {
         alert(data.message);
      } else {
         setResults(data.predictions);
         setIgnored(data.ignored);
      }
    } catch (err) { alert("Analysis failed."); }
    setLoading(false);
  };

  const getThreatColor = (risk: number, isBenign: boolean) => {
    if (isBenign) {
        if (risk >= 80) return { text: "text-emerald-400", bg: "bg-emerald-500", label: "Healthy Patient Baseline" };
        return { text: "text-slate-400", bg: "bg-slate-500", label: "Baseline Overridden by Risk" };
    }
    if (risk < 20) return { text: "text-emerald-400", bg: "bg-emerald-500", label: "Low Risk Profile" };
    if (risk < 50) return { text: "text-amber-400", bg: "bg-amber-500", label: "Elevated Risk Profile" };
    return { text: "text-red-500", bg: "bg-red-500", label: "Critical Risk Profile" };
  };

  const toggleGraph = (cancerType: string) => {
      setExpandedGraph(expandedGraph === cancerType ? null : cancerType);
  };

  const sortedResults = results ? Object.entries(results) : [];

  return (
    <div className="flex h-screen bg-[#020617] text-slate-50 font-sans overflow-hidden">
      
      {/* SIDEBAR */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col hidden md:flex">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center gap-3 mb-1">
            <OncovisionLogo className="w-8 h-8 text-cyan-500" />
            <h1 className="text-2xl font-black text-white">Oncovision <span className="text-cyan-500">AI</span></h1>
          </div>
          <p className="text-slate-400 text-[10px] tracking-widest uppercase">Accessible Cancer Detection</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          <button 
            onClick={() => setCurrentPage("dashboard")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-bold transition-all duration-300 ${currentPage === "dashboard" ? "bg-cyan-500/10 text-cyan-400 shadow-[inset_0_0_0_1px_rgba(6,182,212,0.3)]" : "text-slate-400 hover:bg-slate-800/50 hover:text-white"}`}
          >
            <LayoutDashboard className="w-4 h-4" /> Assessment Center
          </button>
          <button 
            onClick={() => setCurrentPage("about")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-bold transition-all duration-300 ${currentPage === "about" ? "bg-cyan-500/10 text-cyan-400 shadow-[inset_0_0_0_1px_rgba(6,182,212,0.3)]" : "text-slate-400 hover:bg-slate-800/50 hover:text-white"}`}
          >
            <Info className="w-4 h-4" /> About & Mission
          </button>
        </nav>

        {/* FOUNDER CARD */}
        <div className="p-4 m-4 bg-slate-950 border border-slate-800 rounded-xl hover:border-cyan-500/30 transition-colors duration-300">
          <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Lead Developer</p>
          <p className="text-white font-black text-sm">Palash Rakshit</p>
          <div className="flex flex-col gap-2 mt-3">
            <div className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors cursor-pointer">
              <Mail className="w-3 h-3 flex-shrink-0" />
              <a href="mailto:palash.raks@gmail.com" className="text-[10px] font-bold truncate">palash.raks@gmail.com</a>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN VIEWPORT */}
      <main className="flex-1 overflow-y-auto relative custom-scrollbar">
        
        {/* DASHBOARD VIEW */}
        {currentPage === "dashboard" && (
          <div className="p-6 lg:p-10 max-w-[1600px] mx-auto animate-in fade-in duration-500">
            <header className="mb-10">
              <h2 className="text-3xl font-bold text-white">Patient Diagnostics</h2>
              <p className="text-slate-400 text-sm mt-1">Upload up to 5 lab reports simultaneously or verify manual values to generate a calibrated risk report.</p>
            </header>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
              
              {/* INPUT SECTION */}
              <div className="xl:col-span-5 flex flex-col gap-4">
                
                {/* UPLOAD BOX */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 transition-all duration-300 hover:shadow-lg">
                  <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center gap-2">
                    <UploadCloud className="w-4 h-4 text-cyan-500" /> Step 1: Upload Lab Reports
                  </h3>
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-700 hover:border-cyan-500 bg-slate-950 rounded-lg p-6 text-center cursor-pointer transition-colors duration-300 group"
                  >
                    <input type="file" multiple ref={fileInputRef} className="hidden" accept=".pdf" onChange={handleFileUpload} />
                    {parsing ? <Activity className="animate-spin mx-auto w-8 h-8 text-cyan-500" /> : <FileText className="mx-auto w-8 h-8 text-slate-500 group-hover:text-cyan-400 transition-colors" />}
                    <p className="mt-2 text-sm text-cyan-400 font-bold">Select PDF Documents (Max 5)</p>
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="mt-4 p-3 bg-slate-950 rounded border border-slate-800 animate-in slide-in-from-top-2">
                      <p className="text-[10px] text-slate-500 uppercase font-bold mb-2">Ingested Files:</p>
                      <div className="space-y-2">
                        {uploadedFiles.map((fname, idx) => (
                          <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                            <FileCheck className="w-3 h-3 text-emerald-500" />
                            <span className="truncate">{fname}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* RUN BUTTON */}
                <button 
                  onClick={calculateRisk}
                  disabled={loading}
                  className="w-full py-5 bg-cyan-600 hover:bg-cyan-500 text-white font-black text-lg rounded-xl transition-all duration-300 transform hover:-translate-y-1 flex justify-center items-center gap-3 shadow-[0_0_15px_rgba(8,145,178,0.4)]"
                >
                  {loading ? <Activity className="animate-spin w-6 h-6" /> : <Scan className="w-6 h-6" />}
                  {loading ? "ANALYZING..." : "RUN AI ANALYSIS"}
                </button>

                {/* VERIFICATION MATRIX */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex-1 transition-all duration-300 hover:shadow-lg">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-bold text-slate-200">Step 2: Verify Patient Data</h3>
                    <button 
                      onClick={() => {
                          setFormData(Object.keys(formData).reduce((acc, key) => ({ ...acc, [key]: "" }), {}));
                          setUploadedFiles([]);
                      }}
                      className="text-xs text-slate-400 hover:text-red-400 transition-colors bg-slate-950 px-2 py-1 rounded"
                    >
                      Clear Data
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-x-3 gap-y-2 overflow-y-auto max-h-[400px] pr-2 custom-scrollbar">
                    {Object.keys(formData).map((key) => (
                      <div key={key} className={`p-2 rounded bg-slate-950 border transition-colors duration-300 focus-within:border-cyan-500 ${formData[key] !== "" ? 'border-cyan-500/50' : 'border-slate-800'}`}>
                        <label className="text-[10px] text-slate-400 block mb-1 uppercase tracking-wider truncate" title={key.replace(/_/g, ' ')}>
                          {key.replace(/_/g, ' ')}
                        </label>
                        <input 
                          type="number"
                          name={key}
                          value={formData[key]}
                          onChange={handleInputChange}
                          className="w-full bg-transparent text-white font-bold outline-none text-sm placeholder-slate-700"
                          placeholder="-"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* RESULTS SECTION */}
              <div className="xl:col-span-7">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 min-h-[600px] h-full overflow-y-auto custom-scrollbar transition-all duration-300 hover:shadow-lg">
                  <h3 className="text-lg font-bold text-slate-200 mb-6">AI Assessment Report</h3>

                  {!results ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500 text-sm py-20">
                      Enter data and run analysis to view patient predictions.
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {sortedResults.map(([cancerType, data]: any, index: number) => {
                        const isBenign = cancerType === "No Cancer Detected (Benign)";
                        const style = getThreatColor(data.risk, isBenign);
                        const isExpanded = expandedGraph === cancerType;
                        
                        // DYNAMIC GLOWING BORDERS
                        let glowClass = "border-slate-800 p-5";
                        let headerSize = "text-lg";
                        let numSize = "text-4xl";
                        
                        if (!isBenign && data.risk >= 50) {
                            if (index === 0) {
                                glowClass = "border-red-500 border-2 shadow-[0_0_30px_rgba(239,68,68,0.4)] scale-[1.02] p-8";
                                headerSize = "text-2xl text-red-400";
                                numSize = "text-6xl";
                            } else if (index === 1) {
                                glowClass = "border-orange-500 border-2 shadow-[0_0_20px_rgba(249,115,22,0.3)] p-6";
                                headerSize = "text-xl text-orange-400";
                                numSize = "text-5xl";
                            } else if (index === 2) {
                                glowClass = "border-yellow-500 border-2 shadow-[0_0_15px_rgba(234,179,8,0.2)] p-6";
                                headerSize = "text-xl text-yellow-400";
                                numSize = "text-5xl";
                            }
                        } else if (isBenign && data.risk > 80 && index === 0) {
                            glowClass = "border-emerald-500 border-2 shadow-[0_0_30px_rgba(16,185,129,0.3)] scale-[1.02] p-8";
                            headerSize = "text-2xl text-emerald-400";
                            numSize = "text-6xl";
                        }

                        return (
                          <div key={cancerType} className={`bg-slate-950 rounded-xl transition-all duration-500 ${glowClass}`}>
                            <div className="flex justify-between items-center mb-3">
                              <div>
                                <span className={`text-white font-black block ${headerSize}`}>
                                  {cancerType}
                                </span>
                                <span className={`text-xs font-bold mt-1 inline-block px-2 py-1 rounded bg-slate-900 ${style.text}`}>
                                  {style.label}
                                </span>
                              </div>
                              <span className={`font-black ${numSize} ${style.text}`}>
                                {data.risk}%
                              </span>
                            </div>
                            
                            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden mb-4">
                              <div className={`h-full ${style.bg} transition-all duration-1000`} style={{ width: `${data.risk}%` }}></div>
                            </div>

                            {data.contributors && data.contributors.length > 0 && (
                                <button 
                                    onClick={() => toggleGraph(cancerType)}
                                    className="w-full flex items-center justify-center gap-2 py-2 mt-4 text-xs font-bold text-slate-400 bg-slate-900 rounded hover:bg-slate-800 transition-colors"
                                >
                                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    {isExpanded ? "Hide Factor Breakdown" : "View Contributing Medical Factors"}
                                </button>
                            )}
                            
                            {isExpanded && data.contributors && data.contributors.length > 0 && (
                              <div className="mt-6 pt-6 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-6 animate-in slide-in-from-top-2 duration-300">
                                
                                <div className="h-56">
                                  <p className="text-[10px] text-slate-400 uppercase tracking-widest text-center mb-2">Biological Risk Vector</p>
                                  <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data.contributors}>
                                      <PolarGrid stroke="#334155" />
                                      <PolarAngleAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 8 }} />
                                      <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} axisLine={false} />
                                      <Radar name="Risk Impact" dataKey="impact" stroke={isBenign ? '#10b981' : '#ef4444'} fill={isBenign ? '#10b981' : '#ef4444'} fillOpacity={0.4} />
                                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }} />
                                    </RadarChart>
                                  </ResponsiveContainer>
                                </div>

                                <div className="h-56">
                                  <p className="text-[10px] text-slate-400 uppercase tracking-widest mb-2">Patient vs Reference Limits</p>
                                  <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={data.contributors} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                                      <XAxis type="number" hide />
                                      <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                                      <Tooltip 
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        content={({ active, payload }: any) => {
                                          if (active && payload && payload.length) {
                                            const dataPt = payload[0].payload;
                                            const isDanger = dataPt.value > dataPt.limit;
                                            return (
                                              <div className="bg-slate-800 border border-slate-700 p-3 rounded-lg shadow-xl">
                                                <p className="text-white font-bold text-xs mb-1">{dataPt.name}</p>
                                                <p className={isDanger ? "text-red-400 font-bold" : "text-cyan-400 font-bold"}>
                                                  Patient Value: {dataPt.value}
                                                </p>
                                                <p className="text-slate-400 text-[10px] mt-1">Normal Limit: {dataPt.limit}</p>
                                              </div>
                                            );
                                          }
                                          return null;
                                        }}
                                      />
                                      <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
                                        {data.contributors.map((entry: any, index: number) => (
                                          <Cell key={`cell-${index}`} fill={entry.value > entry.limit ? '#ef4444' : (isBenign ? '#10b981' : '#0ea5e9')} />
                                        ))}
                                      </Bar>
                                    </BarChart>
                                  </ResponsiveContainer>
                                </div>

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

        {/* --- PLAIN ENGLISH ABOUT & MISSION PAGE --- */}
        {currentPage === "about" && (
          <div className="p-6 lg:p-16 max-w-[1000px] mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="mb-12 text-center">
              <OncovisionLogo className="w-24 h-24 text-cyan-500 mx-auto mb-6 drop-shadow-[0_0_15px_rgba(6,182,212,0.4)]" />
              <h2 className="text-4xl font-black text-white">Project Oncovision</h2>
              <p className="text-cyan-400 text-lg mt-2 font-bold uppercase tracking-widest">Accessible Cancer Detection</p>
            </header>
            
            <div className="space-y-8 text-slate-300">
              
              {/* The Core Mission */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-2xl">
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <HeartPulse className="text-cyan-500 w-6 h-6" /> Why We Built This
                </h3>
                <p className="leading-relaxed text-lg">
                  Understanding your own health shouldn't require a medical degree or expensive specialist visits. Millions of people get routine blood tests and lab reports every year, but the results are full of confusing numbers and medical jargon. 
                  <br/><br/>
                  <strong>Oncovision changes that.</strong> We built a free, easy-to-use tool that lets you upload your standard lab report and instantly translates those numbers into a clear, visual picture of your health. By looking at all your variables at once, we can spot early warning signs for multiple types of cancer, giving you the power to have an informed, proactive conversation with your doctor.
                </p>
              </div>

              {/* How it Works / The Variables */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center hover:border-cyan-500/50 transition-colors duration-300">
                  <Eye className="w-8 h-8 text-cyan-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">Clear & Transparent</h4>
                  <p className="text-sm text-slate-400">
                    No mystery math. We show you exactly which numbers from your blood test are perfectly healthy, and which ones are raising a flag, so you are never left guessing.
                  </p>
                </div>
                
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center hover:border-cyan-500/50 transition-colors duration-300">
                  <Globe className="w-8 h-8 text-cyan-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">For Everyone, Everywhere</h4>
                  <p className="text-sm text-slate-400">
                    Just upload the PDF file your doctor gives you. Our system automatically reads the text, so anyone with an internet connection can use it instantly.
                  </p>
                </div>
                
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center hover:border-cyan-500/50 transition-colors duration-300">
                  <ShieldCheck className="w-8 h-8 text-cyan-400 mx-auto mb-3" />
                  <h4 className="text-lg font-bold text-white mb-2">Built for the Real World</h4>
                  <p className="text-sm text-slate-400">
                    We programmed the app with real, healthy boundaries. It knows the difference between a harmless bump in your numbers and a real warning sign, so you don't panic over nothing.
                  </p>
                </div>

              </div>

              {/* Privacy Notice */}
              <div className="bg-slate-900/50 border border-emerald-500/30 rounded-xl p-6 text-center">
                 <p className="text-emerald-400 font-bold mb-2 flex justify-center items-center gap-2">
                   <ShieldAlert className="w-5 h-5" /> 100% Private & Secure
                 </p>
                 <p className="text-sm text-slate-400">
                   Your medical data is yours alone. Oncovision never saves, stores, or shares your uploaded lab reports. As soon as you close the page, your information is permanently erased.
                 </p>
              </div>

              {/* Dual Contact Founder Profile */}
              <div className="bg-gradient-to-r from-slate-900 to-slate-950 border border-slate-800 rounded-xl p-8 flex flex-col items-center shadow-xl mt-8">
                 <p className="text-[10px] text-slate-500 uppercase font-bold mb-1 tracking-widest">Founder & Lead Developer</p>
                 <p className="text-white font-black text-3xl mb-4">Palash Rakshit</p>
                 <p className="text-center text-slate-400 text-sm max-w-xl mb-8">
                   Palash built Oncovision to bridge the gap between complex medical data and everyday people. By combining his background in software development and bioinformatics, he is dedicated to creating digital solutions that make advanced healthcare accessible to everyone.
                 </p>
                 
                 <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
                    {/* Developer Contact */}
                    <div className="flex items-center gap-3 text-slate-400 hover:text-cyan-400 transition-colors bg-slate-950 p-4 rounded-lg border border-slate-800 cursor-pointer">
                       <Mail className="w-6 h-6 flex-shrink-0" />
                       <div className="flex flex-col">
                         <span className="text-[9px] uppercase tracking-widest font-bold text-slate-500">Developer Contact</span>
                         <a href="mailto:palash.raks@gmail.com" className="font-bold text-sm">palash.raks@gmail.com</a>
                       </div>
                    </div>

                    {/* Project Inquiries */}
                    <div className="flex items-center gap-3 text-slate-400 hover:text-cyan-400 transition-colors bg-slate-950 p-4 rounded-lg border border-slate-800 cursor-pointer">
                       <Globe className="w-6 h-6 flex-shrink-0" />
                       <div className="flex flex-col">
                         <span className="text-[9px] uppercase tracking-widest font-bold text-slate-500">Project Inquiries</span>
                         <a href="mailto:medtechcentral@gmail.com" className="font-bold text-sm">medtechcentral@gmail.com</a>
                       </div>
                    </div>
                 </div>
              </div>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}