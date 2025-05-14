import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading and preprocessing of training data
data = pd.read_csv("Training data.csv").dropna(axis=1)
X = data.iloc[:, :-1]
y = data["prognosis"]

# Encodeing labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test spliting for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize individual models
rf_model = RandomForestClassifier(random_state=18)
svc_model = SVC(probability=True)  # SVC should have probability=True to use soft voting
gnb_model = GaussianNB()

# Combining models into a voting classifier with soft voting
voting_model = VotingClassifier(estimators=[
    ('rf', rf_model), 
    ('svc', svc_model), 
    ('gnb', gnb_model)
], voting='soft')  # Use 'soft' for probability-based voting

# Training the voting classifier
voting_model.fit(X_train, y_train)

# Model accuracy
voting_accuracy = accuracy_score(y_test, voting_model.predict(X_test))

# Streamlit UI setup
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("Disease Prediction from Symptoms Using Voting Classifier (Combination of SVC, RF, and GNB)")
st.markdown("Select the symptoms you are experiencing from the list below.")

# User input: multiselect for symptoms
symptoms = st.multiselect("Symptoms", options=list(X.columns))

# Display model accuracy
st.info(f"Voting Classifier Accuracy: {voting_accuracy:.2%}")

# Predict from manual input
input_vector = np.zeros(len(X.columns))
for symptom in symptoms:
    if symptom in X.columns:
        input_vector[X.columns.get_loc(symptom)] = 1

if st.button("Predict"):
    input_df = pd.DataFrame([input_vector], columns=X.columns)
    
    # Make prediction using the voting classifier
    prediction = voting_model.predict(input_df)[0]
    probabilities = voting_model.predict_proba(input_df)[0]
    
    # Decode prediction
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted Disease: **{predicted_disease}**")

    # Display top 3 probable diseases
    st.subheader("Top 3 Probable Diseases")
    top_indices = probabilities.argsort()[-3:][::-1]
    for i in top_indices:
        st.write(f"{label_encoder.inverse_transform([i])[0]}: {probabilities[i]:.2%}")
