import streamlit as st
import pandas as pd
import joblib

model_path = "models/best_model.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Credit Risk Scoring", page_icon=":bank:", layout="centered")
st.title("Credit Risk Scoring Application")
st.write("Mijoz malumotlarini kiriting va kredit riskini baholang.")

# User inputs
person_age = st.slider("Yoshi", 18, 100, 25)
person_income = st.number_input("Yillik daromad (USD)", min_value=0, max_value=50000, value=500)
person_home_ownership = st.selectbox("Uy egalik turi", [
    "RENT", "OWN", "MORTGAGE", "OTHER"
])
person_emp_length = st.slider("Ish tajribasi (yil)", 0, 50, 5)
loan_intent = st.selectbox("Kredit maqsadi", [
    "EDUCATION", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "PERSONAL"
])
loan_amnt = st.number_input("Kredit summasi (USD)", min_value=0, value=3000)
loan_int_rate = st.slider("Kredit foizi (%)", 0, 50, 14)
cb_person_cred_hist_length = st.slider("Avvalgi kredit tarihi", 2, 30, 3)

data = pd.DataFrame({
    "person_age" : [person_age],
    "person_income" : [person_income],
    "person_home_ownership" : [person_home_ownership],
    "person_emp_length" : [person_emp_length],
    "loan_intent" : [loan_intent],
    "loan_amnt" : [loan_amnt],
    "loan_int_rate" : [loan_int_rate],
    "cb_person_cred_hist_length" : [cb_person_cred_hist_length]
})


if st.button("Predict"):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]
    
    if prediction == 1:
        st.error(f"Default bo'lish ehtimoli yuqori! Ehtimollik: {probability:.2f}")
    else:
        st.success(f"Kreditni qaytarish ehtimoli yuqori. Ehtimollik: {probability:.2f}")

st.markdown("---")
st.caption("ML Engineer Project — Credit Scoring System")