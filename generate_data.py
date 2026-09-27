"""
generate_data.py
Generates a synthetic but realistic HR dataset for 'ABC Ltd'.
Two prediction targets are embedded:
  - MonthlyIncome_INR   -> continuous target for LINEAR REGRESSION
  - Attrition (Yes/No)  -> binary target for LOGISTIC REGRESSION
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

departments = ["Sales", "Operations", "IT", "HR", "Finance", "Customer Support"]
dept_probs = [0.22, 0.25, 0.18, 0.08, 0.12, 0.15]

job_levels = [1, 2, 3, 4, 5]  # 1 = junior ... 5 = senior
job_level_probs = [0.30, 0.28, 0.22, 0.13, 0.07]

education = ["High School", "Bachelors", "Masters", "PhD"]
education_probs = [0.15, 0.50, 0.28, 0.07]

df = pd.DataFrame({
    "EmployeeID": [f"ABC{str(i).zfill(5)}" for i in range(1, N + 1)],
    "Age": np.random.randint(21, 60, N),
    "Department": np.random.choice(departments, N, p=dept_probs),
    "JobLevel": np.random.choice(job_levels, N, p=job_level_probs),
    "Education": np.random.choice(education, N, p=education_probs),
    "YearsAtCompany": np.random.randint(0, 25, N),
    "DistanceFromHome_KM": np.random.randint(1, 45, N),
    "OverTime": np.random.choice(["Yes", "No"], N, p=[0.30, 0.70]),
    "JobSatisfaction": np.random.randint(1, 5, N),       # 1-4 scale
    "WorkLifeBalance": np.random.randint(1, 5, N),       # 1-4 scale
    "PerformanceRating": np.random.randint(1, 5, N),     # 1-4 scale
    "TrainingHoursLastYear": np.random.randint(0, 80, N),
    "NumPromotionsLast5Yrs": np.random.randint(0, 4, N),
})

# Make YearsAtCompany consistent with Age (can't have more tenure than working years)
df["YearsAtCompany"] = df.apply(
    lambda r: min(r["YearsAtCompany"], max(r["Age"] - 21, 0)), axis=1
)

edu_bonus = {"High School": 0, "Bachelors": 8000, "Masters": 18000, "PhD": 30000}
dept_bonus = {"Sales": 4000, "Operations": 0, "IT": 12000, "HR": 2000,
              "Finance": 9000, "Customer Support": -2000}

# ---- MonthlyIncome (continuous target -> Linear Regression) ----
base_salary = 20000
df["MonthlyIncome_INR"] = (
    base_salary
    + df["JobLevel"] * 15000
    + df["YearsAtCompany"] * 1200
    + df["Age"] * 300
    + df["Education"].map(edu_bonus)
    + df["Department"].map(dept_bonus)
    + df["PerformanceRating"] * 2500
    + df["NumPromotionsLast5Yrs"] * 3000
    + np.random.normal(0, 6000, N)          # random noise
).round(-2).clip(lower=15000)

# ---- Attrition (binary target -> Logistic Regression) ----
# Build a "risk score" then convert to probability via logistic function
risk_score = (
    -0.05 * df["JobSatisfaction"]
    - 0.05 * df["WorkLifeBalance"]
    - 0.02 * df["YearsAtCompany"]
    + 0.35 * (df["OverTime"] == "Yes").astype(int)
    + 0.015 * df["DistanceFromHome_KM"]
    - 0.00002 * (df["MonthlyIncome_INR"] - df["MonthlyIncome_INR"].mean())
    - 0.10 * df["NumPromotionsLast5Yrs"]
    - 0.03 * df["PerformanceRating"]
    + np.random.normal(0, 0.6, N)
)
prob_attrition = 1 / (1 + np.exp(-risk_score))
# Calibrate threshold so overall attrition rate lands around a realistic ~20%
threshold = np.quantile(prob_attrition, 0.80)
df["Attrition"] = np.where(prob_attrition > threshold, "Yes", "No")

# Reorder columns nicely
cols = ["EmployeeID", "Age", "Department", "Education", "JobLevel", "YearsAtCompany",
        "DistanceFromHome_KM", "OverTime", "JobSatisfaction", "WorkLifeBalance",
        "PerformanceRating", "TrainingHoursLastYear", "NumPromotionsLast5Yrs",
        "MonthlyIncome_INR", "Attrition"]
df = df[cols]

df.to_csv("ABC_Ltd_HR_Dataset.csv", index=False)
print("Dataset created:", df.shape)
print(df["Attrition"].value_counts(normalize=True))
print(df.head())
