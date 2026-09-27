# Fitness Level Prediction System 

End-to-end Machine Learning pipeline for predicting fitness level from health data.

## Overview
Analyzed 18,593 health records with 22 features to predict continuous `fitness_level` (0.02 to 20.84) using Regression models. The project focuses on real-world data challenges like data corruption, missing values, and automated preprocessing.

## Dataset Info
- **Shape:** 18,593 rows x 22 columns
- **Target:** `fitness_level` (Continuous - Regression)
- **Key Features:** age, gender, height, weight, activity_type, duration, intensity, calories_burned, heart_rate, stress_level, daily_steps, hydration, bmi, blood_pressure, health_condition, smoking_status
- **Data Issues Fixed:**
    - Last row corrupted with all NaN values - Dropped
    - health_condition 68% missing - Filled with 'Unknown'
    - blood_pressure_systolic 1 missing - Median imputation

## Tech Stack
- **Languages:** Python, SQL
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Tools:** Jupyter Notebook, Git, GitHub

## Workflow
1.  **Data Loading & Cleaning:** Removed corrupted rows, handled missing values
2.  **EDA:** Analyzed distributions, correlations, outliers
3.  **Preprocessing:** Label Encoding for categoricals, Scaling for numericals using ColumnTransformer & Pipeline
4.  **Modeling:** Trained RandomForestRegressor, XGBoostRegressor
5.  **Evaluation:** R2 Score and MAE

## Results
- **Best Model:** RandomForestRegressor
- **R2 Score:** ~0.95
- **MAE:** ~0.80
- Achieved high accuracy after fixing classification vs regression issue

## How to Run
```python
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

# Run pipeline
df = pd.read_csv("health_fitness_dataset.csv")
df = df.dropna(subset=['fitness_level'])

