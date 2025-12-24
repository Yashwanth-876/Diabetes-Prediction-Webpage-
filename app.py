import streamlit as st
import pandas as pd
import pickle
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "brf_model.pkl"), "rb") as f:
    brf = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb") as f:
    model_columns = pickle.load(f)


def recategorize_smoking(status):
    s = str(status).strip().lower()
    if s in ['never', 'no info', 'non-smoker']:
        return 'non-smoker'
    elif s == 'current':
        return 'current'
    elif s in ['ever', 'former', 'not current', 'smoker']:
        return 'past_smoker'
    else:
        return 'unknown'

def preprocess_input(data):
    data['gender'] = le.transform([data['gender']])[0]
    data['smoking_history'] = recategorize_smoking(data['smoking_history'])

    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    for col in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
        df_input[col] = scaler.transform(df_input[[col]].values)

    return df_input

def predict_diabetes(user_input):
    X_input = preprocess_input(user_input)
    prediction = brf.predict(X_input)[0]
    probability = brf.predict_proba(X_input)[0][1]
    return prediction, probability
from PIL import Image

header_img = Image.open("diabetes.jpg")
st.image(header_img, use_container_width=True)
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'> Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown("### Patient Information")
st.markdown("Fill the details below to predict diabetes risk.")

gender = st.radio("Gender", ["Female", "Male"])
age = st.number_input("Age", 0, 100)
st.caption("YES Means 1 and NO Means 0")
hypertension = st.radio("Hypertension", [0,1])
heart_disease = st.radio("Heart Disease", [0, 1])
smoking_history = st.radio(
    "Smoking History",
    ["non-smoker", "current", "past_smoker"]
)
bmi = st.number_input("BMI", 10.0, 100.0)
hba1c = st.number_input("HbA1c Level", 3.0, 10.0)
glucose = st.number_input("Blood Glucose Level", 50, 300)

if st.button("Predict"):
    user_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }

    pred, prob = predict_diabetes(user_data)

    if pred == 1:
        st.error(f"Diabetes Detected (Risk Probability: {prob:.2f})")
    else:
        st.success(f"No Diabetes Detected (Risk Probability: {prob:.2f})")
