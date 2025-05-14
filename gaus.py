import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load and preprocess training data
data = pd.read_csv("Training data.csv").dropna(axis=1)
X = data.iloc[:, :-1]
y = data["prognosis"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train GaussianNB model
model = GaussianNB()
model.fit(X, y_encoded)

# Streamlit UI setup
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("Disease Prediction from Symptoms using GaussianNB")
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
    
    # Get prediction probabilities (GaussianNB supports this)
    probabilities = model.predict_proba(input_df)[0]
    
    # Decode prediction
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted Disease: **{predicted_disease}**")
    
    # Display top 3 probable diseases
    st.subheader("Top 3 Probable Diseases")
    top_indices = probabilities.argsort()[-3:][::-1]  # Sort top 3 predictions
    for i in top_indices:
        st.write(f"{label_encoder.inverse_transform([i])[0]}: {probabilities[i]:.2%}")
