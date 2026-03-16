import streamlit as st
import joblib
import numpy as np
import bisect

model = joblib.load("diabetes_model.joblib")

st.title("Diabetes Prediction")

st.write("This is a website made to predict the chances of having diabetes from given inputs. These predictions were based on historical data from the BRFSS 2021 Survey.")

st.subheader("Please fill in the following:")

# Age, must be over 18, there are 13 age groups
age_ranges = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
age = st.number_input("Age", min_value = 18, value = 18)
age_group = bisect.bisect_right(age_ranges, age)

# BMI
bmi = st.number_input("BMI", min_value = 0, value = 29)

# HighBP
highbp = st.radio("Do you have high blood pressure?", ("No", "Yes"))
highbp_value = 1 if highbp == "Yes" else 0

# HighChol
highchol = st.radio("Do you have high cholesterol levels?", ("No", "Yes"))
highchol_value = 1 if highchol == "Yes" else 0

# Smoker
smoker = st.radio("Do you smoke regularly?", ("No", "Yes"))
smoker_value = 1 if smoker == "Yes" else 0

# Stroke
stroke = st.radio("Have you had a stroke in the past?", ("No", "Yes"))
stroke_value = 1 if stroke == "Yes" else 0

# Heart Disease or Attack
heart = st.radio("Do you have/had a history of a heart disease or attack?", ("No", "Yes"))
heart_value = 1 if heart == "Yes" else 0

# Physical Activity
physact = st.radio("Have you had some sort of physical activity in past 30 days? - not including job", ("No", "Yes"))
physact_value = 1 if physact == "Yes" else 0

# Heavy Alcohol Consumption
heavyalc = st.radio("Do you consider yourself a heavy drinker? - Typically more than 14 drinks per week", ("No", "Yes"))
heavyalc_value = 1 if heavyalc == "Yes" else 0

# General Health
genhealth = st.slider("How would you rate your general health? 1 = Excellent, 5 = Very Poor", 1, 5, 3)

# Mental Health
menthealth = st.slider("How many days in the past month have you struggled mentally? 0 = no days, 1 = 1 day etc", 0, 30, 0)

# Physical Health
physhealth = st.slider("How many days in the past month have you struggled physically? 0 = no days, 1 = 1 day etc", 0, 30, 0)

# Difficulty Walking
diffwalking = st.radio("Do you have difficulty walking or climbing stairs?", ("No", "Yes"))
diffwalking_value = 1 if diffwalking == "Yes" else 0


st.subheader("Make sure to have answered all the questions.")
st.write("Click the button below to calculate your diabetes risk. The risk levels are as follows:")

st.write("""
1. **Extremely High Risk** - Immediate attention required. Consult a healthcare professional right away.
2. **High Risk** - You are at high risk. Take action to manage your health and consult a healthcare provider.
3. **Moderate Risk** - Increased risk. Consider preventive measures and monitor your health closely.
4. **Unlikely** - Your risk is low, but it's still important to maintain a healthy lifestyle.
5. **Very Unlikely** - Your risk is very low. Keep up with a healthy lifestyle to maintain good health.
6. **Extremely Unlikely** - Your risk is extremely low. Stay healthy and continue your wellness practices.
""")

if st.button("Calculate"):
    # Features in order
    features = np.array([age_group, bmi, highbp_value, highchol_value, smoker_value, stroke_value,
                         heart_value, physact_value, heavyalc_value, genhealth, menthealth, physhealth,
                         diffwalking_value]).reshape(1, -1)
    
    # Make the prediction (probability)
    prob = model.predict_proba(features)[0][1]
    st.write(f"The probability of diabetes is: {prob:.2f}")

    # Categorize based on probability thresholds
    if prob >= 0.8:
        st.subheader("1. Extremely High Risk")
        st.write("You are at a extremely high risk of having diabetes. It is important to consult with a healthcare professional and take immediate action to manage your health.")

    elif prob >= 0.7:
        st.subheader("2. High Risk")
        st.write("You are at high risk of developing diabetes. It's important to take preventive measures and consult a healthcare provider.")

    elif prob >= 0.6:
        st.subheader("3. Moderate Risk")
        st.write("Your risk of diabetes is moderate. Consider making lifestyle changes and monitoring your health closely.")

    elif prob >= 0.4:
        st.subheader("4. Unlikely")
        st.write("Your risk of diabetes is relatively low, but it's still important to maintain a healthy lifestyle.")

    elif prob >= 0.2:
        st.subheader("5. Very Unlikely")
        st.write("You are very unlikely to have diabetes. Keep up with a healthy lifestyle to maintain good health.")
    else:
        st.subheader("6. Extremely Unlikely")
        st.write("Your risk of diabetes is extremely low. Continue to live a healthy lifestyle.")

