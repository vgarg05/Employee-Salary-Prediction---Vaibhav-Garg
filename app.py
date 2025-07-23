import streamlit as st
import pandas as pd
import joblib

# Load your trained best model
model = joblib.load("best_model.pkl")

# Page config
st.set_page_config(
    page_title="💼 Employee Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Title & description
st.title("💼 Employee Salary Prediction App")
st.markdown("""
Predict whether an employee's annual income is **>50K or ≤50K** based on their personal & work details.

This app uses a **Machine Learning Algorithm - GradientBoosting** trained on the **Employee Salary Dataset**.
""")

# Sidebar inputs for single prediction
st.sidebar.header("📝 Enter Employee Details")

# Example inputs
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked", "Others"
])
fnlwgt = st.sidebar.number_input("fnlwgt (final weight)", 0, 2000000, 100000)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving",
    "Priv-house-serv", "Protective-serv", "Armed-Forces", "Others"
])
relationship = st.sidebar.selectbox("Relationship", [
"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.radio("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)
educational_num = st.sidebar.slider("Education Num", 1, 16, 10)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Canada", "Mexico", "Others"
])

# Manual mappings
workclass_map = {
    "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2,
    "Federal-gov": 3, "Local-gov": 4, "State-gov": 5,
    "Without-pay": 6, "Never-worked": 7, "Others": 8
}
marital_status_map = {
    "Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2,
    "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6
}
occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
    "Armed-Forces": 13, "Others": 14
}
relationship_map = {
    "Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5
}
race_map = {
    "White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4
}
gender_map = {"Male": 0, "Female": 1}
native_country_map = {
    "United-States": 0, "Canada": 1, "Mexico": 2, "Others": 3
}

# Build input DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "workclass": [workclass_map[workclass]],
    'fnlwgt': [fnlwgt],
    "educational-num": [educational_num],
    "marital-status": [marital_status_map[marital_status]],
    "occupation": [occupation_map[occupation]],
    "relationship": [relationship_map[relationship]],
    "race": [race_map[race]],
    "gender": [gender_map[gender]],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss],
    "hours-per-week": [hours_per_week],
    "native-country": [native_country_map[native_country]]
})

st.write("### 📊 Input Data for Prediction")
st.write(input_data)

if st.button("🚀 Predict Income Class"):
    prediction = model.predict(input_data)[0]
    st.success(f"💡 **Prediction:** {'>50K' if prediction == 1 else '<=50K'}")

# Batch prediction
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file with same columns for batch prediction", type="csv")

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview", batch_df.head())
    batch_preds = model.predict(batch_df)
    batch_df['PredictedIncome'] = batch_preds
    st.write("✅ Prediction Result", batch_df.head())
    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Result CSV", csv, "predicted_income.csv", "text/csv")
