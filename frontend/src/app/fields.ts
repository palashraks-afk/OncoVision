// The application's input schema.
//
// Every key here is a feature one or more of the trained models consumes, or a
// value the PDF parser knows how to extract. Keys must stay in sync with
// LAB_FIELDS and HISTORY_FIELDS in train_models.py.

export type LabField = {
  key: string;
  label: string;
  unit: string;
  normal: string;
  meaning: string;
};

export type LabGroup = {
  group: string;
  blurb: string;
  items: LabField[];
};

export const LAB_GROUPS: LabGroup[] = [
  {
    group: "Body metrics",
    blurb: "Baseline context. These do not say anything on their own, but they scale most of the other numbers.",
    items: [
      { key: "age", label: "Age", unit: "years", normal: "any", meaning: "Cancer incidence climbs steeply with age, so every model uses it as a baseline." },
      { key: "bmi", label: "BMI", unit: "kg/m2", normal: "18.5 to 24.9", meaning: "Body mass index. A raised BMI is an established risk factor for liver, pancreatic, colorectal and post menopausal breast cancer." },
    ],
  },
  {
    group: "Complete blood count",
    blurb: "The most common blood test there is. It sits on the first page of almost every lab report.",
    items: [
      { key: "wbc", label: "WBC, white blood cells", unit: "K/uL", normal: "4.0 to 11.0", meaning: "Your immune cell count. A count that stays high points to chronic inflammation, infection, or a blood cancer." },
      { key: "rbc", label: "RBC, red blood cells", unit: "M/uL", normal: "4.2 to 5.5", meaning: "Oxygen carrying cell count. A low count often goes with slow, hidden bleeding from a tumour in the gut." },
      { key: "hemoglobin", label: "Hemoglobin", unit: "g/dL", normal: "12.0 to 15.0", meaning: "The protein inside red cells that carries oxygen. An unexplained drop is one of the earliest signs of chronic disease." },
      { key: "platelets", label: "Platelets", unit: "K/uL", normal: "150 to 400", meaning: "Clotting cells. A count that runs high is a known finding in ovarian, lung and gastrointestinal cancer." },
      { key: "hematocrit", label: "Hematocrit", unit: "%", normal: "36 to 46", meaning: "The share of blood made up of red cells. It falls alongside haemoglobin in chronic disease and slow blood loss." },
      { key: "mcv", label: "MCV, mean cell volume", unit: "fL", normal: "80 to 100", meaning: "Average red cell size. Small cells point to iron loss, which is how a bleeding gut tumour often shows itself first." },
      { key: "mch", label: "MCH, mean cell haemoglobin", unit: "pg", normal: "27 to 33", meaning: "Average haemoglobin per red cell. Read next to MCV to separate the causes of anaemia." },
      { key: "rdw", label: "RDW, red cell distribution width", unit: "%", normal: "11.5 to 14.5", meaning: "How unevenly sized the red cells are. A raised RDW is one of the earliest and least specific signs of chronic illness." },
      { key: "mpv", label: "MPV, mean platelet volume", unit: "fL", normal: "7.5 to 12.0", meaning: "Average platelet size. It shifts with the raised platelet turnover seen in several cancers." },
      { key: "neutrophil_pct", label: "Neutrophils", unit: "%", normal: "40 to 70", meaning: "The share of white cells that are neutrophils. A high proportion alongside low lymphocytes is a recognised inflammatory pattern in cancer." },
    ],
  },
  {
    group: "Metabolic panel",
    blurb: "Kidney function, sugar handling and mineral balance.",
    items: [
      { key: "glucose", label: "Glucose", unit: "mg/dL", normal: "70 to 99 fasting", meaning: "Blood sugar. Diabetes appearing for the first time in an older adult with no family history can be an early sign of pancreatic cancer." },
      { key: "calcium", label: "Calcium", unit: "mg/dL", normal: "8.5 to 10.3", meaning: "Serum calcium. A raised level is a recognised finding in lung and breast cancer and in myeloma." },
      { key: "bun", label: "BUN, blood urea nitrogen", unit: "mg/dL", normal: "7 to 20", meaning: "A waste product the kidneys filter out. Read next to creatinine to judge kidney function." },
      { key: "creatinine", label: "Creatinine", unit: "mg/dL", normal: "0.6 to 1.2", meaning: "The main measure of kidney filtration. Impaired kidneys change how the rest of the panel should be read." },
      { key: "protein_total", label: "Total protein", unit: "g/dL", normal: "6.0 to 8.0", meaning: "All protein circulating in the blood. A shift can point to liver disease, kidney loss or abnormal antibody production." },
      { key: "albumin", label: "Albumin", unit: "g/dL", normal: "3.5 to 5.0", meaning: "The main protein the liver makes. A low level reflects poor nutrition or a liver that is struggling to synthesise." },
    ],
  },
  {
    group: "Liver panel",
    blurb: "Enzymes and pigments that show liver cell injury and bile duct blockage.",
    items: [
      { key: "ast", label: "AST", unit: "U/L", normal: "under 40", meaning: "An enzyme released when liver cells are damaged. It also comes from heart and muscle, so it is read next to ALT." },
      { key: "alt", label: "ALT", unit: "U/L", normal: "under 40", meaning: "The more liver specific of the two enzymes. AST and ALT rising together means active liver cell injury." },
      { key: "bilirubin", label: "Total bilirubin", unit: "mg/dL", normal: "under 1.2", meaning: "The yellow pigment left over from broken down red cells. Jaundice without pain, with bilirubin climbing, is a common way pancreatic head tumours present." },
      { key: "alkaline_phosphatase", label: "Alkaline phosphatase", unit: "U/L", normal: "44 to 120", meaning: "An enzyme from the bile ducts and bone. A raised level suggests bile duct blockage or cancer that has spread to bone." },
      { key: "ggt", label: "GGT", unit: "U/L", normal: "under 50", meaning: "A bile duct enzyme. It confirms that a raised alkaline phosphatase came from the liver rather than from bone." },
    ],
  },
  {
    group: "Tumour markers",
    blurb: "Proteins shed by specific tumours. These carry the most weight in the models that use them.",
    items: [
      { key: "alpha_fetoprotein_level", label: "AFP, alpha fetoprotein", unit: "ng/mL", normal: "under 10", meaning: "The main blood marker for liver cancer. It also rises in some testicular tumours and in pregnancy." },
      { key: "psa", label: "PSA, prostate specific antigen", unit: "ng/mL", normal: "under 4.0", meaning: "The main screening marker for prostate cancer. A benign enlarged prostate raises it too, so it is never read alone." },
      { key: "plasma_ca19_9", label: "CA 19-9", unit: "U/mL", normal: "under 37", meaning: "The main blood marker for pancreatic cancer. Bile duct blockage and pancreatitis also raise it." },
      { key: "ca125", label: "CA 125", unit: "U/mL", normal: "under 35", meaning: "The main blood marker for ovarian cancer. Endometriosis, fibroids and even menstruation raise it too, which is why it is read next to HE4 rather than alone." },
      { key: "he4", label: "HE4", unit: "pmol/L", normal: "under 140", meaning: "Human epididymis protein 4. It stays normal in most benign gynaecological conditions, so it is the more specific half of the ovarian pair." },
      { key: "cea", label: "CEA", unit: "ng/mL", normal: "under 5", meaning: "Carcinoembryonic antigen. Mainly a colorectal marker, and used alongside CA 125 to tell a bowel primary from an ovarian one." },
    ],
  },
  {
    group: "Breast mass morphology",
    blurb: "Measurements taken from a digitised image of a breast biopsy. You will only have these if a lump has been sampled and imaged.",
    items: [
      { key: "radius_mean", label: "Radius mean", unit: "", normal: "under 15", meaning: "Average distance from the centre of a cell nucleus to its edge. Malignant nuclei run consistently larger." },
      { key: "texture_mean", label: "Texture mean", unit: "", normal: "around 19", meaning: "How much the grey scale varies across the nucleus. Malignant nuclei look more mottled." },
      { key: "perimeter_mean", label: "Perimeter mean", unit: "", normal: "under 95", meaning: "Average length around the edge of the nucleus. It rises with radius as cells turn malignant." },
      { key: "area_mean", label: "Area mean", unit: "", normal: "under 600", meaning: "Average cross sectional area of the nucleus. One of the strongest single separators in the Wisconsin data." },
    ],
  },
];

export const LAB_KEYS = LAB_GROUPS.flatMap(g => g.items.map(i => i.key));

// Information about the patient. Not printed on a lab report, so it has to be
// answered directly. Leaving these blank costs real accuracy.
//
// Only fields a shipped panel actually consumes are listed. Inherited risk,
// prior cancer diagnosis, family history and cirrhosis were removed once the
// general and liver panels moved to NHANES, because no model reads them any
// more and asking for information nothing uses is only friction.
export type HistoryField = {
  key: string;
  label: string;
  meaning: string;
  // Which heading this sits under. The list runs to 22 fields once the
  // cervical panel's risk history is included, which is too many to read as
  // one flat block.
  group: "General" | "Reproductive history";
  type: "select" | "number";
  options?: { value: number; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
};

export const HISTORY_FIELDS: HistoryField[] = [
  {
    key: "gender", label: "Sex at birth", type: "select", group: "General",
    meaning: "Used by the general, liver and pancreatic models.",
    options: [{ value: 0, label: "Female" }, { value: 1, label: "Male" }],
  },
  {
    key: "smoking", label: "Smoking", type: "select", group: "General",
    meaning: "Tobacco exposure is a direct risk factor for liver and pancreatic cancer.",
    options: [
      { value: 0, label: "Never" },
      { value: 1, label: "Former" },
      { value: 2, label: "Current" },
    ],
  },
  {
    key: "alcohol_intake", label: "Alcohol intake", type: "number", group: "General",
    meaning: "On a scale of 0 for none to 5 for heavy. The main driver of cirrhosis, which precedes most liver cancer.",
    min: 0, max: 5, step: 0.5, suffix: "of 5",
  },
  {
    key: "physical_activity", label: "Exercise", type: "number", group: "General",
    meaning: "Hours of activity in a normal week.",
    min: 0, max: 10, step: 0.5, suffix: "hrs/week",
  },
  {
    key: "hepatitis_b", label: "Hepatitis B", type: "select", group: "General",
    meaning: "Chronic hepatitis B is one of the strongest liver cancer risk factors known.",
    options: [{ value: 0, label: "Negative" }, { value: 1, label: "Positive" }],
  },
  {
    key: "hepatitis_c", label: "Hepatitis C", type: "select", group: "General",
    meaning: "Chronic hepatitis C carries the same risk through the same route.",
    options: [{ value: 0, label: "Negative" }, { value: 1, label: "Positive" }],
  },
  {
    key: "diabetes", label: "Diabetes", type: "select", group: "General",
    meaning: "Long standing diabetes raises both liver and pancreatic risk.",
    options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }],
  },

  // Only menopausal status survives here. The cervical panel that read the
  // sexual and reproductive history was withdrawn once repeated splits showed
  // its published AUC was a lucky shuffle, and this project does not ask a
  // patient for information no shipped model reads.
  // Cervical cancer is caused by persistent HPV infection, so these fields are
  // the exposure proxies for it rather than moral bookkeeping. Cutting the list
  // was tried and measured: eight of them scored 0.665 with an interval that
  // contains chance, and six scored 0.552, against 0.725 for the full set.
  // Every one of them is here because removing it costs accuracy.
  //
  // Leaving them blank is fine. Anything missing is filled with the training
  // median, and the cervical panel reports reduced coverage when that happens.
  {
    key: "menopause", label: "Menopausal status", type: "select",
    group: "Reproductive history",
    meaning: "Read by the ovarian panel. CA 125 and HE4 are interpreted against different cut-offs before and after menopause.",
    options: [{ value: 0, label: "Pre-menopausal" }, { value: 1, label: "Post-menopausal" }],
  },
];

export const HISTORY_GROUPS: HistoryField["group"][] = [
  "General",
  "Reproductive history",
];

export const HISTORY_KEYS = HISTORY_FIELDS.map(f => f.key);
export const ALL_KEYS = [...LAB_KEYS, ...HISTORY_KEYS];
