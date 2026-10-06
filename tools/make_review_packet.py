"""
Builds the clinician review packet: every rule the navigator applies, in plain words with
tick boxes, so a doctor can review the rule file in an hour without reading code.

Run:  python tools/make_review_packet.py     writes docs/clinician_review/Rule_Review_Packet.pdf
"""

import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import navigator as nav  # noqa: E402
import screening as sc  # noqa: E402

OUT = os.path.join(ROOT, "docs", "clinician_review")
os.makedirs(OUT, exist_ok=True)
kb = nav.load()
skb = sc.load()
e = html.escape

TIER = {"talk_soon": "See a doctor soon", "worth_raising": "Worth raising", "mention": "Mention at a routine visit"}
TF = {"immediate": "immediately", "48h": "within 48 hours", "2w": "within 2 weeks", "primary_care": "primary-care test", "routine": "routine / non-urgent"}


def path_text(path):
    return " AND ".join(nav.describe(c) for c in path)


rows = []
for r in kb["rules"] + [kb["melanoma_checklist"]]:
    if "paths" in r:
        crit = "<br>".join(f"<b>{'OR ' if i else ''}</b>{e(path_text(p))}" for i, p in enumerate(r["paths"]))
    else:
        crit = "Self-scored checklist: " + e("; ".join(f"{f['label']} ({f['points']})" for f in r["features"])) + f". Threshold {r['threshold']} or more."
    rows.append(f"""<div class="rule"><div class="rh"><span class="id">{e(r['id'])}</span> <span class="site">{e(r['site'])}</span>
<span class="tier">{TIER.get(nav.tier_of(r), '')} · {TF.get(r['timeframe'], r['timeframe'])} · guideline strength: {e(r['strength'])}</span></div>
<div class="crit"><b>Criteria in the file:</b><br>{crit}</div>
<div class="act"><b>What the tool tells the person:</b> {e(r['action'])}</div>
<div class="src"><b>Source page cited:</b> {e(r['source'])}</div>
<div class="tick">☐ matches the current guideline &nbsp;&nbsp; ☐ needs a change &nbsp;&nbsp; ☐ remove &nbsp;&nbsp; Comment: ______________________________________________</div></div>""")

concerns = [
    ("Version", "The rules are transcribed from the 2015 edition of NICE NG12. NICE has since updated it (secondary sources report August and October 2023, including a quantitative faecal immunochemical test, FIT, as an early step in the bowel pathway). Please compare each rule with the current version."),
    ("US relevance", "The audience is likely to include US users. Should the symptom rules be shown alongside, replaced by, or reconciled with US guidance?"),
    ("Platelet gating (lung)", "NG12 lists a high platelet count as a trigger for considering a chest X-ray. Applied literally to symptom-free adults over 40 it alerted 3.7% of a US sample. The rule file requires at least one symptom as well. Do you agree?"),
    ("Lab-only alerts", "With no symptoms, the anaemia and iron-deficiency rules alone alert 16 to 27% of people aged 80+ and about 14% of women aged 40 to 49 (likely menstrual iron loss). Should lab-only alerts be shown to symptom-free people at all, or only with a symptom?"),
    ("Wording", "Is every action sentence safe, accurate and appropriately cautious? The tool never says a person has cancer."),
    ("Tiers", "Are the three levels (see a doctor soon, worth raising, mention) a fair translation of the guideline's timeframes and its 'recommend' versus 'consider'?"),
    ("Missing rules", "Are any commonly important presentations missing, for example for children, for thyroid, or for head and neck?"),
    ("Safety text", "Is the emergency wording adequate? \"Seek urgent care now if you cough up or vomit a lot of blood. Do the same for severe chest pain, trouble breathing, or black or bloody stools with faintness.\""),
]

screen_rows = "".join(
    f"<tr><td>{e(i['site'])}</td><td>{e(i['source'])}</td><td>{e(i['url'])}</td><td>☐ ok &nbsp; ☐ change</td></tr>" for i in skb["items"])

css = """
@page { size: Letter; margin: 18mm; @bottom-center { content: counter(page); font: 9pt Georgia; color:#666; } }
body { font-family: Georgia, serif; font-size: 10pt; line-height: 1.4; color: #1b1f24; }
h1 { font-size: 18pt; margin: 0 0 4pt; } h2 { font-size: 12.5pt; border-bottom: 1.5px solid #1b1f24; padding-bottom: 2pt; margin: 16pt 0 6pt; }
.rule { border: 1px solid #c9ced4; padding: 6pt 8pt; margin: 6pt 0; break-inside: avoid; }
.rh { margin-bottom: 3pt; } .id { font-family: Consolas, monospace; font-weight: bold; } .site { text-transform: capitalize; font-weight: bold; margin-left: 6pt; }
.tier { color: #5b6672; font-size: 9pt; margin-left: 8pt; } .crit, .act, .src { margin: 2pt 0; font-size: 9.4pt; } .tick { margin-top: 5pt; font-size: 9.4pt; }
.box { border: 1.5px solid #b3261e; background: #fdf1f0; padding: 7pt 10pt; margin: 8pt 0; font-size: 9.6pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; } td, th { border: 1px solid #c9ced4; padding: 3pt 5pt; vertical-align: top; text-align: left; }
ol li { margin-bottom: 4pt; }
"""
concern_html = "".join(f"<li><b>{e(a)}.</b> {e(b)}<br>Your view: ____________________________________________________________</li>" for a, b in concerns)
doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Clinician review packet</title><style>{css}</style></head><body>
<h1>Clinician review packet: cancer symptom and lab navigator</h1>
<p>Prepared by Palash Rakshit, Oncovision project (github.com/palashraks-afk/OncoVision). Live prototype: oncovisionai.vercel.app/navigator. Estimated time: about one hour.</p>
<div class="box"><b>What this is.</b> A patient-facing prototype that applies published referral criteria to a person's symptoms and lab results and explains why. It has no clinical content of its own: every rule below was transcribed from a guideline by a student with the help of an AI assistant, and <b>no clinician has reviewed any of it</b>. The tool says so to every user. Your review is what would let it be called reviewed.</div>
<h2>What is being asked</h2>
<ol><li>Tick one box per rule: matches the current guideline, needs a change, or remove.</li>
<li>Give a view on the eight design questions below.</li>
<li>Check the five screening items against the current US Preventive Services Task Force text.</li>
<li>Say whether you are willing to be named as reviewer, and whether you wish to be an author. Neither is assumed.</li></ol>
<h2>Eight questions for you</h2><ol>{concern_html}</ol>
<h2>The {len(kb['rules']) + 1} navigator rules</h2>{''.join(rows)}
<h2>The five US screening items</h2>
<p>Each applies only to average-risk adults. Wording was checked against the Task Force pages on 5 October 2026.</p>
<table><tr><th>Cancer</th><th>Source cited</th><th>Page</th><th>Your view</th></tr>{screen_rows}</table>
<h2>Sign-off</h2><p>Reviewer name and role: ______________________ &nbsp; Date: __________ &nbsp; Signature: ______________________</p>
<p>Overall: ☐ rules acceptable as a research prototype &nbsp; ☐ acceptable after the changes marked &nbsp; ☐ not acceptable</p>
</body></html>"""
html_path = os.path.join(OUT, "Rule_Review_Packet.html")
open(html_path, "w", encoding="utf-8").write(doc)
pdf = os.path.join(OUT, "Rule_Review_Packet.pdf")
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + html_path.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf)
