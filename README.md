# Fitness Level Prediction System 
> End-to-end ML pipeline built by troubleshooting real-world health data (18,593 records) - Developed during ML Internship at OTP Technologies, Eluru.

## Overview
Analyzed 18,593 health records with 22 features to predict continuous `fitness_level` (0.02 to 20.84) using Regression. Focused on **debugging data corruption, handling missing values, and automating preprocessing in Unix/Linux environment** - core skills for Amazon Support Engineer Intern.

##  Tech Stack (Amazon Job Description Match)
- **Languages:** Python, Shell Scripting, SQL
- **OS:** Unix, Linux
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Tools:** Git, GitHub, Jupyter, ColumnTransformer, Pipeline
- **CS Fundamentals:** Data Structures, Algorithms, OOP, Debugging, Troubleshooting

## Dataset Info
- **Shape:** 18,593 rows x 22 columns
- **Features:** age, gender, height, weight, activity_type, duration, intensity, calories, heart_rate, stress_level, daily_steps, hydration, bmi, resting_heart_rate, blood_pressure, health_condition, smoking_status
- **Target:** fitness_level (continuous - Regression problem)
- **Issues Found & Fixed:**
    - Last row corrupted with all NaN values
    - health_condition 68% missing (5895/18593 present)
    - blood_pressure_systolic 1 missing
    - Fixed via median imputation & 'Unknown' filling

##  Troubleshooting & Debugging Done
1. **Data Corruption:** Identified and dropped last row where fitness_level=NaN causing pipeline failure
2. **Missing Values:** Implemented robust imputation - median for numeric, 'Unknown' for categorical
3. **Unix Automation:** Created preprocessing pipeline using ColumnTransformer & Pipeline for large-scale data handling
4. **Performance Issue:** Initial accuracy low due to wrong problem type (Classification vs Regression) - fixed to Regressor

##  Workflow
```python
# 1. Load & Debug
df = pd.read_csv("health_fitness_dataset.csv")
df = df.dropna(subset=['fitness_level']) # Fix corrupted row

# 2. Encode categoricals for Unix pipeline
cat_cols = ['gender','activity_type','intensity']
# LabelEncoder used

# 3. Train-Test Split & Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

# 4. Evaluation
R2 Score: ~0.95 | MAE: ~0.8


