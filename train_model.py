"""
train_model.py
Trains two models on the ABC Ltd HR dataset:
  1. Linear Regression   -> predicts MonthlyIncome_INR (continuous)
  2. Logistic Regression -> predicts Attrition (Yes/No)

Saves both fitted pipelines (preprocessing + model) as .pkl files
so the Streamlit app can load them directly without retraining.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

df = pd.read_csv("ABC_Ltd_HR_Dataset.csv")

numeric_features = ["Age", "JobLevel", "YearsAtCompany", "DistanceFromHome_KM",
                     "JobSatisfaction", "WorkLifeBalance", "PerformanceRating",
                     "TrainingHoursLastYear", "NumPromotionsLast5Yrs"]
categorical_features = ["Department", "Education", "OverTime"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------------
# MODEL 1: LINEAR REGRESSION  -> predict MonthlyIncome_INR
# ---------------------------------------------------------------------
X_lin = df[numeric_features + categorical_features]
y_lin = df["MonthlyIncome_INR"]

X_train, X_test, y_train, y_test = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

linreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
linreg_pipeline.fit(X_train, y_train)
y_pred = linreg_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=" * 60)
print("LINEAR REGRESSION — Predicting Monthly Income")
print("=" * 60)
print(f"R2 Score : {r2:.3f}")
print(f"MAE      : Rs. {mae:,.0f}")
print(f"RMSE     : Rs. {rmse:,.0f}")

joblib.dump(linreg_pipeline, "linear_regression_income_model.pkl")

# ---------------------------------------------------------------------
# MODEL 2: LOGISTIC REGRESSION -> predict Attrition (Yes/No)
# ---------------------------------------------------------------------
numeric_features_log = numeric_features  # same numeric set
categorical_features_log = categorical_features

X_log = df[numeric_features_log + categorical_features_log]
y_log = (df["Attrition"] == "Yes").astype(int)

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
)

preprocessor_log = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features_log),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features_log),
])

logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_log),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
logreg_pipeline.fit(Xl_train, yl_train)
yl_pred = logreg_pipeline.predict(Xl_test)
yl_proba = logreg_pipeline.predict_proba(Xl_test)[:, 1]

acc = accuracy_score(yl_test, yl_pred)
prec = precision_score(yl_test, yl_pred)
rec = recall_score(yl_test, yl_pred)
f1 = f1_score(yl_test, yl_pred)
auc = roc_auc_score(yl_test, yl_proba)
cm = confusion_matrix(yl_test, yl_pred)

print()
print("=" * 60)
print("LOGISTIC REGRESSION — Predicting Attrition")
print("=" * 60)
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 Score : {f1:.3f}")
print(f"ROC-AUC  : {auc:.3f}")
print("Confusion Matrix:")
print(cm)

joblib.dump(logreg_pipeline, "logistic_regression_attrition_model.pkl")

# Save feature lists + metrics for the app to reuse
metrics = {
    "linear": {"r2": r2, "mae": mae, "rmse": rmse},
    "logistic": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                 "confusion_matrix": cm.tolist()},
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "departments": sorted(df["Department"].unique().tolist()),
    "education_levels": sorted(df["Education"].unique().tolist()),
}
joblib.dump(metrics, "model_metadata.pkl")

print("\nSaved: linear_regression_income_model.pkl")
print("Saved: logistic_regression_attrition_model.pkl")
print("Saved: model_metadata.pkl")
