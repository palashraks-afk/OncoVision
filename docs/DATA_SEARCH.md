# Data search: where the cancer-specificity question could be answered (27 November 2026)

**Need:** haemoglobin and ferritin on the same people, cancer diagnoses by site, ideally endoscopy records, and far more
events than the 14 cancer deaths in the NHANES analysis.

## Best fit: NIH All of Us (records plus volunteers)

Counts read from the public Data Browser (databrowser.researchallofus.org), which needs no account. Counts are rounded to 20,
are of all ages, and each row counts participants with at least one such record.

| Record | Participants |
|---|---|
| Haemoglobin in blood | 358,320 |
| Ferritin, serum or plasma (main code) | 122,820 |
| Ferritin measurement (second code; overlap with the first unknown) | 41,700 |
| Colonoscopy with biopsy | 67,340 |
| Colonoscopy, diagnostic | 59,420 |
| Upper endoscopy (EGD) codes | listed; counts not recorded here |
| Iron deficiency anaemia (diagnosis) | 58,000 |
| Malignant tumour of colon | 4,600 |

**Not yet known (needs access):** how many aged 50+ have both tests close in time, how many meet the guideline definition,
and how many cancers follow. So it is **not yet shown** that the study would have enough events.

**Access.** The registered-institutions page lists **University of California, Irvine (Registered and Controlled tiers)**,
listed contact Hoda Anton-Culver; Chapman University is also listed. The UCI library guide (guides.lib.uci.edu/AllofUs/access)
says UCI affiliates have access: researcher profile, identity check through Login.gov or ID.me (may ask for an SSN and
state ID), Responsible Conduct of Research training, Data User Code of Conduct. Only researchers whose institution has an
agreement may register. **No rule found on students, minors or outside collaborators** (the All of Us FAQ page refused
automated reading). Do not share anyone's login. Ask the program (aoudurasupport@vumc.org is named on the register page) and
UCI before assuming a student can take part.

**A design warning.** In All of Us, ferritin was ordered by clinicians, usually for a reason. That is the indication problem
the NHANES design avoided. It suits the **follow-up question** (after a guideline red flag, who gets scoped, how soon, and
what is found), which fills the largest gap in this project. For cancer specificity, results must be read as "among people a
doctor chose to test".

## Other routes checked

- **Korea, KNHANES 2008-2012:** haemoglobin and ferritin; one study merged it with national insurance claims for colorectal
  cancer follow-up to 2018, with administrative approval (PMC9666249). Korean population; linkage not openly downloadable.
- **UK Biobank:** a study reports ferritin from linked primary-care records for 82,678 participants, ordered tests only; no
  source found showing it in the baseline panel. Application needed.
- **MIMIC-IV:** open to credentialed users (training and, for students, a supervisor reference). Built from hospital records, so
  unsuited to long follow-up of well people (general knowledge, not verified here).
- **Open repositories (Zenodo, Dryad, figshare, PhysioNet):** one web search found no suitable open dataset. Not exhaustive.

## The study to propose (fix before looking at data)

1. Adults 50+, no earlier GI cancer, first haemoglobin and ferritin within 90 days meeting AGA 2020 IDA.
2. Primary: share with colonoscopy and upper endoscopy within 6 and 12 months (the gap this project cannot currently measure).
3. Secondary: GI cancer diagnosed within 1 and 3 years, in those scoped and not scoped; time to diagnosis.
4. Negative control: non-GI cancer and other outcomes, to test specificity.
5. Report by insurance and by demographic group (stratifiers only).

## Draft message (send yourself; edit freely; check every number first)

> Subject: Student project on untreated iron-deficiency anaemia: advice on All of Us access
>
> Dear Dr. [name],
>
> I am a student working on a project using public CDC survey data. I found that about 2.1 million US adults over 50 have
> iron-deficiency anaemia as defined by the AGA 2020 guideline, and about 80% report no anaemia treatment in the past three
> months. The survey cannot show whether they had endoscopy.
>
> All of Us appears to hold the records to answer that, and UC Irvine is listed as a registered institution. Would you, or
> someone you know, be willing to advise on this or consider it as a study? I used an AI assistant for much of the code and
> analysis, and I would like a person to check it. I can send the paper, the pre-registration and the code.
>
> Thank you,
> Palash Rakshit
