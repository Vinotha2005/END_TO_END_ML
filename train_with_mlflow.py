# train_with_mlflow.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn

# ========== Load & Clean Data ==========
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()

# Fill numeric missing values with median; categorical with mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ========== Encode categorical variables ==========
le_country = LabelEncoder()
le_status = LabelEncoder()

if 'Country' in df.columns:
    df['Country'] = le_country.fit_transform(df['Country'].astype(str))
if 'Status' in df.columns:
    df['Status'] = le_status.fit_transform(df['Status'].astype(str))

# Save encoders for app.py
joblib.dump(le_country, "country_encoder.pkl")
joblib.dump(le_status, "status_encoder.pkl")

# ========== Prepare train/test ==========
target_col = 'Life expectancy'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training columns for app alignment
joblib.dump(list(X_train.columns), "model_columns.pkl")

# ========== Define models ==========
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# ========== MLflow experiment ==========
mlflow.set_experiment("Life Expectancy Prediction")

best_r2 = -np.inf
best_model = None
best_name = None
results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # Use np.sqrt for RMSE — compatible with all scikit-learn versions
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log params & metrics to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("rmse", float(rmse))

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path=name)

        print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        results.append((name, r2, mae, rmse))

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

# ========== Save best model locally ==========
if best_model is not None:
    joblib.dump(best_model, "best_model.pkl")
    print(f"✅ Best model: {best_name} saved as best_model.pkl (R2={best_r2:.4f})")

# ========== Show summary ==========
results_df = pd.DataFrame(results, columns=["Model", "R2", "MAE", "RMSE"]).sort_values("R2", ascending=False)
print("\nModel comparison:\n", results_df)
