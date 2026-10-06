# A message to send Dr. Chavan

Edit anything that does not sound like you. Attach: `docs/paper/Navigator_Research_Paper.pdf` and
`docs/clinician_review/Rule_Review_Packet.pdf`. Send it yourself; nothing here has been sent.

---

**Subject:** A student cancer-screening project, and one request for your clinical review

Dr. Chavan,

I'm Palash Rakshit. I've been building a project called Oncovision, a patient-facing tool that
reads symptoms and lab results against published cancer referral guidelines and explains the
result in plain language. There is also a second page that lists which routine screenings
(mammogram, bowel, lung, cervical, PSA) US guidelines recommend for a person's age.

I tested it on about 38,000 US adults, with the pass or fail bars written down before I ran
anything. Most of the bars were missed. The lab alerts do not tell cancer deaths from other
deaths, and a model using everything the survey recorded did no better than age, sex,
smoking and BMI. I wrote the paper around those results because I think an honest negative
result is more useful than an overstated one. It is attached, with the live prototype at
oncovisionai.vercel.app/navigator.

I don't want to ask for much. What I can't do myself is the one thing the project needs: **no
clinician has reviewed the rules.** I typed them in from the 2015 UK guideline with an AI
assistant, and the tool says that on every page. I've made a packet (attached) where each of the
34 rules is in plain words with tick boxes, plus eight design questions. I think it takes about
an hour.

Would you be willing to either (1) go through it yourself, or (2) point me to a colleague in
adult primary care, oncology or geriatrics who would, since the tool is aimed at adults? I'm
also happy to hear that it isn't a good use of your time.

I haven't assumed anything about your name on the work. If you did review it and wanted to be
named, or to be an author, I'd want to settle that with you openly first.

Thank you for reading this.

Palash Rakshit

---

## What to expect and how to handle it

- **If he says yes to reviewing:** have the packet printed or open on screen when you meet,
  take his notes literally, and change the rule file only the way he says. Record what he
  decided in `backend/navigator_rules.json` (`review_status`, `reviewed_by`).
- **If he suggests someone else:** that's fine. The packet is written for any clinician.
- **If he wants to be an author:** he must make a real contribution to the paper and approve
  it (the usual ICMJE standard). Reviewing the rules plus revising the draft usually meets it.
- **If he asks about AI use:** tell him the truth. An AI assistant helped write the code, run
  the analyses, and draft text; you reviewed it. The paper's Declarations section says this.
- **Do not** say it has been clinically validated, that the paper is published, or that there
  is ethics approval. None of those is true yet.
