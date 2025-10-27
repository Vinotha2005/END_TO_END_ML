# 🌍 Life Expectancy Prediction App

This project predicts **Life Expectancy** based on various **health, economic, and demographic indicators** using Machine Learning.  
The app is built using **Streamlit** and trained on the **WHO Life Expectancy dataset**.

## App:
https://endtoendml-1.streamlit.app/

## 🚀 Project Overview

The goal of this project is to predict life expectancy of a country given health and economic factors such as:
- Adult mortality
- Infant deaths
- Alcohol consumption
- Expenditure on health
- Measles cases
- BMI
- GDP
- Population
- Schooling, etc.

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical columns (`Country`, `Status`)
   - Scale numeric columns

2. **Model Training**
   - Trained multiple regression models:
     - Linear Regression  
     - Random Forest Regressor  
     - XGBoost Regressor  
     - Gradient Boosting Regressor  

   - Compared performance using **R² score** and **Mean Absolute Error (MAE)**  
   - Saved the **best model** as `best_model.pkl` using `joblib`

3. **Streamlit App**
   - Users can input new data via form fields  
   - The model predicts **expected life expectancy**  
   - The app displays:
     - Input data table
     - Prediction result
     - Model metrics (optional)

## Visualization:

### Distribution of Life Expectancy
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/50c20206-8109-4861-a8f6-c4625677f6fe" />

### Skewness:
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/dfe6f074-26bd-4341-863b-9910be4de799" />

### Correlation Heatmap:
<img width="1123" height="899" alt="image" src="https://github.com/user-attachments/assets/506e2d6b-ba83-4453-8b22-181110038a65" />

### Feature Importance:
<img width="1053" height="547" alt="image" src="https://github.com/user-attachments/assets/df7d7cfe-d3a0-46da-9db8-4bbfa33ff3dd" />


## 📊 Technologies Used

Python 3.10+

Pandas, NumPy, Scikit-learn

XGBoost, RandomForest

Joblib

Streamlit

## 💡 Future Improvements

Add data visualization dashboard

Deploy on Streamlit Cloud / HuggingFace / Render

Add feature importance graph

Auto model selection toggle

