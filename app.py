"""
app.py
Streamlit deployment of ABC Ltd's predictive HR analytics tool.
Run with:  streamlit run app.py

Two models, one app, built for non-technical managers/HR staff:
  1. Salary Estimator      (Linear Regression)
  2. Attrition Risk Checker (Logistic Regression)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ABC Ltd | People Analytics", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# Load trained models + metadata (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    linreg = joblib.load("linear_regression_income_model.pkl")
    logreg = joblib.load("logistic_regression_attrition_model.pkl")
    meta = joblib.load("model_metadata.pkl")
    df = pd.read_csv("ABC_Ltd_HR_Dataset.csv")
    return linreg, logreg, meta, df

linreg_model, logreg_model, meta, df = load_artifacts()

st.title("📊 ABC Ltd — People Analytics Predictive Tool")
st.caption("Built with Linear Regression (Salary) and Logistic Regression (Attrition Risk)")

tab1, tab2, tab3 = st.tabs(["💰 Salary Estimator", "🚪 Attrition Risk Checker", "📈 Model Performance"])

# ------------------------------------------------------------------
# Shared input widgets (used by both prediction tabs)
# ------------------------------------------------------------------
def employee_input_form(key_prefix):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 21, 60, 30, key=f"{key_prefix}_age")
        department = st.selectbox("Department", meta["departments"], key=f"{key_prefix}_dept")
        education = st.selectbox("Education", meta["education_levels"], key=f"{key_prefix}_edu")
    with col2:
        job_level = st.slider("Job Level (1=Junior, 5=Senior)", 1, 5, 2, key=f"{key_prefix}_level")
        years_at_company = st.slider("Years at Company", 0, 25, 3, key=f"{key_prefix}_years")
        distance = st.slider("Distance From Home (KM)", 1, 45, 10, key=f"{key_prefix}_dist")
    with col3:
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"], key=f"{key_prefix}_ot")
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3, key=f"{key_prefix}_sat")
        wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3, key=f"{key_prefix}_wlb")

    col4, col5 = st.columns(2)
    with col4:
        performance = st.slider("Performance Rating (1-4)", 1, 4, 3, key=f"{key_prefix}_perf")
        training_hours = st.slider("Training Hours Last Year", 0, 80, 20, key=f"{key_prefix}_train")
    with col5:
        promotions = st.slider("Promotions in Last 5 Years", 0, 4, 1, key=f"{key_prefix}_promo")

    return pd.DataFrame([{
        "Age": age, "JobLevel": job_level, "YearsAtCompany": years_at_company,
        "DistanceFromHome_KM": distance, "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": wlb, "PerformanceRating": performance,
        "TrainingHoursLastYear": training_hours, "NumPromotionsLast5Yrs": promotions,
        "Department": department, "Education": education, "OverTime": overtime,
    }])

# ------------------------------------------------------------------
# TAB 1: Salary Estimator (Linear Regression)
# ------------------------------------------------------------------
with tab1:
    st.subheader("Estimate a fair Monthly Income (INR)")
    st.write("Enter an employee's profile to get a predicted salary based on company-wide patterns.")
    input_df = employee_input_form("lin")
    if st.button("Predict Salary", type="primary"):
        pred_salary = linreg_model.predict(input_df)[0]
        st.success(f"### Predicted Monthly Income: ₹ {pred_salary:,.0f}")
        st.caption("This is a data-driven estimate, not a guaranteed salary offer. "
                   "Use it as one input among several for compensation decisions.")

# ------------------------------------------------------------------
# TAB 2: Attrition Risk Checker (Logistic Regression)
# ------------------------------------------------------------------
with tab2:
    st.subheader("Check an employee's Attrition Risk")
    st.write("Enter an employee's profile to estimate the probability they may leave the company.")
    input_df2 = employee_input_form("log")
    if st.button("Predict Attrition Risk", type="primary"):
        proba = logreg_model.predict_proba(input_df2)[0][1]
        pred_class = "Yes — At Risk" if proba >= 0.5 else "No — Likely to Stay"

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Attrition Probability", f"{proba*100:.1f}%")
            st.write(f"**Prediction:** {pred_class}")
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 0.6))
            ax.barh([0], [proba], color="#d62728" if proba >= 0.5 else "#2ca02c")
            ax.barh([0], [1], color="#eeeeee", zorder=0)
            ax.barh([0], [proba], color="#d62728" if proba >= 0.5 else "#2ca02c", zorder=1)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Risk of Attrition")
            st.pyplot(fig)

        if proba >= 0.5:
            st.warning("⚠️ This employee profile shows elevated attrition risk. "
                       "Consider a retention conversation, workload review, or career development plan.")
        else:
            st.info("✅ This employee profile shows low attrition risk based on current patterns.")

# ------------------------------------------------------------------
# TAB 3: Model Performance (transparency for managers)
# ------------------------------------------------------------------
with tab3:
    st.subheader("How good are these models?")

    st.markdown("#### 💰 Linear Regression — Salary Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{meta['linear']['r2']:.3f}")
    c2.metric("MAE", f"₹{meta['linear']['mae']:,.0f}")
    c3.metric("RMSE", f"₹{meta['linear']['rmse']:,.0f}")
    st.caption("R² close to 1.0 means the model explains most of the variation in salary. "
               "MAE/RMSE show the typical prediction error in rupees.")

    st.markdown("#### 🚪 Logistic Regression — Attrition Model")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{meta['logistic']['accuracy']:.2f}")
    c2.metric("Precision", f"{meta['logistic']['precision']:.2f}")
    c3.metric("Recall", f"{meta['logistic']['recall']:.2f}")
    c4.metric("F1 Score", f"{meta['logistic']['f1']:.2f}")
    c5.metric("ROC-AUC", f"{meta['logistic']['auc']:.2f}")
    st.caption("Recall matters most here: it reflects how many *actual leavers* the model correctly flags, "
               "which is critical for a retention early-warning tool.")

    cm = np.array(meta["logistic"]["confusion_matrix"])
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Stay", "Leave"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Stay", "Leave"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    st.markdown("#### 📁 Underlying Dataset Snapshot")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Full dataset: {df.shape[0]} employees, {df.shape[1]} columns.")

st.divider()
st.caption("ABC Ltd Predictive Analytics Tool | Built for Business Analytics Assignment 2 | "
           "Models: Linear & Logistic Regression | Deployed with Streamlit")
