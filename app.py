import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0.42, 80.0, 30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.2)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Encode categorical inputs
sex_encoded = 0 if sex == "Male" else 1
embarked_encoded = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]

# Prepare input array
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("🎉 The passenger **likely survived**.")
    else:
        st.error("💀 The passenger **did not survive**.")
