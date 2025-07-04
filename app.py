import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('loan_approval_model.pkl')

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details to predict loan approval status")

# User Inputs
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in 1000s)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ['Yes (1.0)', 'No (0.0)'])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Preprocessing input
def preprocess_input():
    gender_val = 1 if gender == 'Male' else 0
    married_val = 1 if married == 'Yes' else 0
    dependents_val = 3 if dependents == '3+' else int(dependents)
    education_val = 0 if education == 'Graduate' else 1
    self_employed_val = 1 if self_employed == 'Yes' else 0
    credit_history_val = 1.0 if credit_history == 'Yes (1.0)' else 0.0
    property_area_val = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[property_area]

    features = np.array([[gender_val, married_val, dependents_val, education_val, self_employed_val,
                         applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                         credit_history_val, property_area_val]])
    return features

# Prediction
if st.button("Predict Approval Status"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan is likely to be **Approved**!")
    else:
        st.error("❌ Loan is likely to be **Rejected**.")
