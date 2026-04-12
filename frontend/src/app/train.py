import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load your Kaggle dataset (e.g., patient_data.csv)
df = pd.read_csv("patient_data.csv")

# 2. Separate your inputs (X) and your answer (y)
X = df.drop("Cancer_Diagnosis_Label", axis=1)
y = df["Cancer_Diagnosis_Label"]

# 3. FIX THE IMBALANCE: Use SMOTE to generate synthetic data for rare cases
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# 4. Train the model using balanced class weights
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_balanced, y_balanced)

# 5. Save the fixed model so your API can use it
model.feature_names = list(X.columns)
joblib.dump(model, "models/model_fixed.joblib")
print("Successfully trained and balanced the AI model.")