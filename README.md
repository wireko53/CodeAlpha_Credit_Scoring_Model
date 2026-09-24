# Credit Scoring Model

Predicting individual creditworthiness using historical financial and loan application data.

Built with Python, Pandas, and Scikit-learn as part of the CodeAlpha Machine Learning Internship.

---

## Overview

I built this project to tackle one of the most practical problems in finance — figuring out whether a loan applicant is likely to default. Using the Kaggle Credit Risk dataset, I put together a full pipeline that cleans the raw data, scales numeric features, encodes categorical variables, and trains a Random Forest classifier to flag high-risk applicants.

Along the way I ran into (and fixed) a few classic ML pitfalls — a mislabeled target column that was inflating my class count, and a data leakage issue that was giving me a suspicious 100% accuracy. The final model reflects a realistic, properly validated result.

---

## Model Performance

- **Algorithm:** Random Forest Classifier (`n_estimators=100`)
- **Accuracy:** 93.19%
- **ROC-AUC Score:** 0.9314
- **Precision / Recall / F1:** Evaluated separately for low-risk (`0`) and high-risk (`1`) borrowers via `classification_report`

---

## Dataset

The dataset holds financial and demographic details for each loan applicant:

- `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`
- `cb_person_default_on_file`, `cb_person_cred_hist_length`
- **Target (`loan_status`):** `0` = Non-Default (Good Credit), `1` = Default (High Risk)

---

## Pipeline

### 1. Data Preprocessing & Cleaning

- Filled missing numeric values with the column median
- Isolated `loan_status` as the target before any feature processing, to avoid leakage
- One-hot encoded categorical fields with `pd.get_dummies()`

### 2. Feature Scaling & Train-Test Split

- 80/20 train-test split with stratified sampling to preserve class balance
- Standardized numeric features with `StandardScaler`

### 3. Model Training & Evaluation

- Trained a `RandomForestClassifier`
- Evaluated with Precision, Recall, F1-Score, Accuracy, and ROC-AUC
- Plotted a confusion matrix with Seaborn to visualize true vs. false predictions

---

## Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
├── credit_risk_dataset.csv    # Dataset
├── credit_scoring.ipynb       # Main Jupyter Notebook
└── README.md                  # Project documentation
```

---

## Getting Started

### Requirements

- Python 3.x
- pandas, numpy, scikit-learn, matplotlib, seaborn

### Running It

```bash
git clone https://github.com/wireko53/CodeAlpha_Credit_Scoring_Model.git
cd CodeAlpha_Credit_Scoring_Model
```

Then open `credit_scoring.ipynb` in VS Code or Jupyter and run the cells top to bottom to reproduce the preprocessing, training, and evaluation steps.

---

## Notes

This was completed as Task 1 of the CodeAlpha Machine Learning Internship. It's a solid baseline — future improvements could include hyperparameter tuning (e.g. `GridSearchCV`) or testing gradient-boosted models like XGBoost against the current Random Forest.
