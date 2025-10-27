import streamlit as st
import pandas as pd
import joblib

# 🎯 Page configuration
st.set_page_config(page_title="🌍 Life Expectancy Prediction App", layout="centered")

# 🧠 Load model and encoders
try:
    model = joblib.load("best_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    le_country = joblib.load("country_encoder.pkl")
    le_status = joblib.load("status_encoder.pkl")
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or columns: {e}")
    st.stop()

# 🧾 Title and description
st.title("🌍 Life Expectancy Prediction App")
st.markdown("""
This app predicts **Life Expectancy** based on various health and economic indicators.  
Please provide the details below 👇
""")

# 🧮 User Inputs
st.subheader("🔢 Enter Input Features:")

country = st.selectbox("Country", le_country.classes_)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1, value=2015)
status = st.selectbox("Status", le_status.classes_)
adult_mortality = st.number_input("Adult Mortality", 0, 1000, 150)
infant_deaths = st.number_input("Infant Deaths", 0, 500, 50)
alcohol = st.number_input("Alcohol", 0.0, 20.0, 4.0)
percentage_expenditure = st.number_input("Percentage Expenditure", 0.0, 10000.0, 1000.0)
hepatitis_b = st.number_input("Hepatitis B", 0.0, 100.0, 80.0)
bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
polio = st.number_input("Polio", 0.0, 100.0, 90.0)
total_expenditure = st.number_input("Total Expenditure", 0.0, 100.0, 5.0)
hiv_aids = st.number_input("HIV/AIDS", 0.0, 50.0, 1.0)
gdp = st.number_input("GDP", 0.0, 100000.0, 5000.0)
thinness_1_19 = st.number_input("Thinness 10-19 years", 0.0, 50.0, 5.0)
thinness_5_9 = st.number_input("Thinness 5-9 years", 0.0, 50.0, 4.0)
income_comp = st.number_input("Income composition of resources", 0.0, 1.0, 0.7)
schooling = st.number_input("Schooling", 0.0, 20.0, 12.0)

# 🧩 Create input DataFrame
input_data = {
    'Country': [country],
    'Year': [year],
    'Status': [status],
    'Adult Mortality': [adult_mortality],
    'infant deaths': [infant_deaths],
    'Alcohol': [alcohol],
    'percentage expenditure': [percentage_expenditure],
    'Hepatitis B': [hepatitis_b],
    'BMI': [bmi],
    'Polio': [polio],
    'Total expenditure': [total_expenditure],
    'HIV/AIDS': [hiv_aids],
    'GDP': [gdp],
    'Thinness 1-19 years': [thinness_1_19],
    'Thinness 5-9 years': [thinness_5_9],
    'Income composition of resources': [income_comp],
    'Schooling': [schooling]
}
input_df = pd.DataFrame(input_data)

# 🧠 Encode categorical columns
input_df['Country'] = le_country.transform(input_df['Country'])
input_df['Status'] = le_status.transform(input_df['Status'])

# ✅ Reorder columns to match training
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# 🧾 Show the input
st.subheader("📊 Input Data Used for Prediction:")
st.dataframe(input_df)

# 🔮 Predict button
if st.button("🔮 Predict Life Expectancy"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Life Expectancy: **{prediction:.2f} years**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
