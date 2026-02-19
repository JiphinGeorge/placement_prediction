import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/placement_model.pkl")
encoders = joblib.load("model/label_encoders.pkl")

# Page config
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Placement Prediction System")
st.markdown("Predict whether a student will be placed based on academic and employability factors.")

# Sidebar inputs
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["M", "F"])

ssc_p = st.sidebar.slider("SSC Percentage", 0.0, 100.0, 70.0)
hsc_p = st.sidebar.slider("HSC Percentage", 0.0, 100.0, 70.0)
degree_p = st.sidebar.slider("Degree Percentage", 0.0, 100.0, 70.0)
mba_p = st.sidebar.slider("MBA Percentage", 0.0, 100.0, 70.0)

workex = st.sidebar.selectbox("Work Experience", ["Yes", "No"])

etest_p = st.sidebar.slider("Employability Test Percentage", 0.0, 100.0, 70.0)

specialisation = st.sidebar.selectbox(
    "MBA Specialisation",
    ["Mkt&HR", "Mkt&Fin"]
)

# Feature engineering
academic_avg = (ssc_p + hsc_p + degree_p + mba_p) / 4
employability_score = (etest_p + mba_p) / 2

# Create dataframe
input_data = pd.DataFrame({
    "gender": [gender],
    "ssc_p": [ssc_p],
    "hsc_p": [hsc_p],
    "degree_p": [degree_p],
    "workex": [workex],
    "etest_p": [etest_p],
    "specialisation": [specialisation],
    "mba_p": [mba_p],
    "academic_avg": [academic_avg],
    "employability_score": [employability_score]
})

# Encode categorical data
for column in input_data.columns:
    if column in encoders:
        input_data[column] = encoders[column].transform(input_data[column])

# Main prediction area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Profile")
    st.write(input_data)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Placement", use_container_width=True):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("✅ Student is likely to be PLACED")
            st.metric("Placement Probability", f"{probability[1]*100:.2f}%")
        else:
            st.error("❌ Student is likely NOT to be placed")
            st.metric("Placement Probability", f"{probability[0]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, XGBoost, and Machine Learning")
