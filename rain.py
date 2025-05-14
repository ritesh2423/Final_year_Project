import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess training data
data = pd.read_csv("Training data.csv").dropna(axis=1)
X = data.iloc[:, :-1]
y = data["prognosis"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(random_state=18)
model.fit(X, y_encoded)

# Streamlit UI setup
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("Disease Prediction from Symptoms using Random Forest")
st.markdown("Select the symptoms you are experiencing from the list below.")

# User input: multiselect for symptoms
symptoms = st.multiselect("Symptoms", options=list(X.columns))

# Prepare input vector
input_vector = np.zeros(len(X.columns))
for symptom in symptoms:
    if symptom in X.columns:
        input_vector[X.columns.get_loc(symptom)] = 1

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([input_vector], columns=X.columns)
    prediction = model.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted Disease: **{predicted_disease}**")
