from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model = joblib.load("model/rf_model.pkl")

# Define all input features (in exact training order)
features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
            'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension',
            'Diabetes', 'Depression', 'Stroke', 'CholesterolTriglycerides', 'UPDRS',
            'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
            'PosturalInstability', 'SpeechProblems', 'SleepDisorders',
            'Constipation', 'BP_Diff', 'Cholesterol_Ratio', 'LDL_to_HDL']

# Columns that were label encoded during training
categorical_features = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
                        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']

# Define label encoders with training categories
label_encoders = {
    'Gender': LabelEncoder().fit(['Male', 'Female', 'Other']),
    'Ethnicity': LabelEncoder().fit(['White', 'Black', 'Asian', 'Hispanic', 'Other']),
    'EducationLevel': LabelEncoder().fit(['HighSchool', 'Bachelors', 'Masters', 'PhD', 'Other']),
    'Smoking': LabelEncoder().fit(['Yes', 'No']),
    'AlcoholConsumption': LabelEncoder().fit(['None', 'Low', 'Moderate', 'High']),
    'PhysicalActivity': LabelEncoder().fit(['Low', 'Moderate', 'High']),
    'DietQuality': LabelEncoder().fit(['Poor', 'Average', 'Good']),
    'SleepQuality': LabelEncoder().fit(['Poor', 'Average', 'Good']),
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Collect form data
        form_data = {f: request.form.get(f, 0) for f in features}
        input_df = pd.DataFrame([form_data])

        # Encode categorical columns
        for col in categorical_features:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except Exception:
                    input_df[col] = 0  # fallback if unknown value

        # Convert remaining columns to float
        for col in input_df.columns:
            if col not in categorical_features:
                try:
                    input_df[col] = input_df[col].astype(float)
                except:
                    input_df[col] = 0.0

        # Predict using trained model
        pred = model.predict(input_df)[0]

        # Decode output
        prediction = "Parkinson's Positive" if pred == 1 else "Parkinson's Negative"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
