# ABC Ltd — Predictive Analytics & Managerial AI Adoption
### Business Analytics — Assignment 2

## 1. Project Overview

This project builds, deploys, and reflects on two predictive models for a
fictional company, **ABC Ltd**, using an HR/employee dataset styled on
real-world industry data (deliberately anonymized per the assignment brief).

| Model | Type | Predicts | Business Use Case |
|---|---|---|---|
| Linear Regression | Regression | `MonthlyIncome_INR` (continuous) | Fair, data-driven salary benchmarking |
| Logistic Regression | Classification | `Attrition` (Yes/No) | Early warning for employee turnover risk |

## 2. Files in this project

```
ABC_Ltd_HR_Dataset.csv                    -> the dataset (1,500 employees, 15 columns)
generate_data.py                          -> script that generated the dataset
train_model.py                            -> trains both models, prints metrics, saves .pkl files
linear_regression_income_model.pkl        -> trained salary model (pipeline incl. preprocessing)
logistic_regression_attrition_model.pkl   -> trained attrition model (pipeline incl. preprocessing)
model_metadata.pkl                        -> saved metrics + feature lists used by the app
app.py                                    -> Streamlit app (the deployed tool)
requirements.txt                          -> Python dependencies
README.md                                 -> this file
```

## 3. How to run it yourself

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate the dataset and retrain models from scratch
python generate_data.py
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Streamlit will open automatically in your browser (usually `http://localhost:8501`).
No coding knowledge is needed to use the app itself — it's built entirely with
sliders, dropdowns, and buttons for non-technical users (HR managers, in this case).

## 4. Dataset Description

Since the assignment asks for a genuine industry-style dataset without naming
the real company, this dataset ("ABC Ltd") was synthetically generated to
**mirror the structure and statistical relationships of real HR attrition
datasets** (similar in spirit to the well-known IBM HR Analytics dataset on
Kaggle), with realistic correlations deliberately built in:

- Higher job level, tenure, age, education, and performance → higher salary
- Overtime, low job satisfaction, poor work-life balance, long commute,
  and lack of promotions → higher attrition risk
- Attrition rate calibrated to ~20%, matching typical real-world HR data

**Columns:** Age, Department, Education, JobLevel, YearsAtCompany,
DistanceFromHome_KM, OverTime, JobSatisfaction, WorkLifeBalance,
PerformanceRating, TrainingHoursLastYear, NumPromotionsLast5Yrs,
MonthlyIncome_INR (linear regression target), Attrition (logistic regression target)

## 5. Model Building Summary

Both models use a `scikit-learn` **Pipeline** (StandardScaler for numeric
features + OneHotEncoder for categorical features), so raw employee data
goes in and a clean prediction comes out — no manual preprocessing needed
at prediction time.

**Linear Regression (Salary):**
- R² ≈ 0.93 — the model explains ~93% of the variation in salary
- MAE ≈ ₹5,000 — typical prediction error
- Interpretation: salary is highly predictable from job level, tenure, age,
  education, and performance, which is expected in a structured pay system

**Logistic Regression (Attrition):**
- Accuracy ≈ 0.70, ROC-AUC ≈ 0.78, Recall ≈ 0.67
- `class_weight="balanced"` was used because attrition is a minority class
  (~20%) — without this, the model would be biased toward predicting "No"
- Recall is prioritized in the business framing: in a retention tool, missing
  an at-risk employee (false negative) is usually costlier than flagging a
  safe employee for a check-in (false positive)

## 6. Understanding Managerial AI Adoption (Reflection)

This section addresses the "understand" objective of the assignment —
thinking like a manager deciding whether and how to adopt this tool, not
just like a data scientist building it.

### Why this matters to managers
A predictive tool like this turns HR data into a decision-support system.
Instead of reacting to resignations after they happen, a manager can see
attrition risk scores proactively and intervene — a raise conversation,
workload adjustment, or career development plan — while retention is still
possible. Similarly, the salary model supports fairer, more consistent pay
decisions instead of relying purely on manager discretion, which can
introduce bias.

### Why a Streamlit deployment matters (not just a Jupyter notebook)
A model sitting in a notebook only a data scientist can run creates zero
business value. Deploying it as a simple, no-code interface is what actually
lets an HR manager or business owner use it directly. This is the real
bridge between "data science project" and "adopted AI capability."

### Adoption barriers a manager should anticipate
- **Trust:** managers may not trust a "black box" score; the Model
  Performance tab exists specifically to build transparency and let users
  judge model reliability instead of asking for blind trust.
- **Bias and fairness:** if historical data encodes bias (e.g., against
  certain departments or demographics), the model will learn and repeat it.
  Regular audits of predictions by department/demographic are essential
  before using this in real HR decisions.
- **Over-reliance:** the tool should support judgment, not replace it. A
  high attrition-risk score should trigger a conversation, not an automatic
  action (like denying a promotion).
- **Change management:** non-technical staff need training on what the
  output means (a probability, not a certainty) and its limitations.
- **Data quality and drift:** predictions are only as good as the data.
  Models should be retrained periodically as company conditions change.

### Governance recommendations
- Use predictions as one input among several, not the sole basis, for
  decisions about pay or termination
- Keep a human in the loop for any consequential decision
- Monitor model performance over time and retrain when accuracy drops
- Be transparent with employees that such models are used, per data
  ethics and (where applicable) legal requirements

## 7. Limitations
- Dataset is synthetic; a real deployment would need company data,
  proper consent, and legal/compliance review of use
- Logistic regression, while interpretable, may miss non-linear
  attrition patterns that tree-based models (e.g., Random Forest,
  XGBoost) could capture — a good next iteration
- Streamlit app runs locally in this submission; for company-wide use it
  would need to be hosted (e.g., Streamlit Community Cloud, an internal
  server, or a cloud VM) with access controls
