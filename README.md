# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## The Business Problem

Acquiring a new customer costs 5–7× more than keeping an existing one. A model that identifies at-risk customers before they leave gives a retention team something to act on - but only if it is interpretable enough to explain *why* a customer is flagged, not just *that* they are.

This project builds that model on the real IBM Telco Customer Churn dataset - 7,043 customers, 20 features, and a 26.5% churn rate that reflects the genuine imbalance a retention team would face in production.

> **Core question:** *Can we predict which customers will churn with enough accuracy and lead time to make proactive retention economically worthwhile - and which features tell us the most about why they leave?*

---

## Dashboard Preview

### Churn Distribution
<img width="508" height="527" alt="image" src="https://github.com/user-attachments/assets/f4c43eb8-3c0a-43c6-9935-9a8f58fcde7a" />

### Tenure & Monthly Charges by Churn
<img width="1016" height="502" alt="image" src="https://github.com/user-attachments/assets/a4d7e9f4-4c85-4680-ad2b-1d51b7efb6f5" />

<img width="1020" height="510" alt="image" src="https://github.com/user-attachments/assets/717ccbd4-f814-4ad4-b2f5-dbdb4f10c3ad" />

### Churn Rate by Contract Type & Internet Service
<img width="890" height="527" alt="image" src="https://github.com/user-attachments/assets/64cbde9e-8a23-489a-b093-9bbe53683a7d" />

<img width="891" height="531" alt="image" src="https://github.com/user-attachments/assets/5df3a422-edcd-4dad-9e0c-2aebe25f8412" />

### ROC Curves - Model Comparison
<img width="793" height="551" alt="image" src="https://github.com/user-attachments/assets/37f26f6e-f395-44cf-a44b-5b0a73c7c3ae" />

### Confusion Matrix - Best Model
<img width="579" height="553" alt="image" src="https://github.com/user-attachments/assets/f59b201e-ac3e-42d5-8df9-65a4bcc5da78" />

### Feature Importance - Random Forest
<img width="792" height="552" alt="image" src="https://github.com/user-attachments/assets/3034ce51-94f6-4b12-aff2-01ec165280f3" />

### Threshold Optimisation
<img width="954" height="391" alt="image" src="https://github.com/user-attachments/assets/ce7107cf-ad8c-4b30-9607-cafce6c2fe5a" />

---

## What the Notebook Covers

### Data Cleaning
The dataset has known quality issues handled explicitly:
- `TotalCharges` stored as a string - converted to numeric, 11 blank entries (new customers with zero tenure) filled with 0
- `customerID` dropped - not a predictive feature
- `No internet service` and `No phone service` values in service columns collapsed to `No` before binary encoding
- One-hot encoding applied to `Contract`, `InternetService`, and `PaymentMethod`

### Exploratory Analysis
Three key patterns emerge before any modelling:
- Churners skew heavily toward **short tenure** - the distribution is right-heavy for retained customers, left-heavy for churners
- Churners pay **higher monthly charges** on average - they are not the customers with the lowest engagement
- **Month-to-month contracts** churn at dramatically higher rates than one- or two-year contracts

### Model Comparison

Three models trained and evaluated with 5-fold stratified cross-validation:

| Model | AUC-ROC | CV AUC |
|---|---|---|
| **Logistic Regression** | **0.8418** | **0.8462** |
| Gradient Boosting | 0.8376 | 0.8432 |
| Random Forest | 0.8265 | 0.8261 |

**Logistic Regression wins** - not just on test AUC, but on cross-validation stability (0.8462 CV vs 0.8432 for GBM). For a retention use case where the model will be retrained periodically on new data, stability matters as much as peak performance. It is also the most interpretable - coefficients map directly to feature impact, which is what a retention team needs to act on.

### Feature Importance
Top 15 features from Random Forest importance scores. The three dominant predictors - tenure, total charges, and monthly charges - are continuous variables, confirming that *how long* a customer has been with the company and *how much* they are paying are the strongest churn signals. Contract type features appear next, followed by service add-ons.

### Threshold Optimisation
The default 0.5 probability threshold is not optimal for churn problems. Missing a churner (false negative) costs more than incorrectly flagging a loyal customer (false positive) - the retention team would rather make an unnecessary call than miss someone about to leave.

The notebook finds the threshold that maximises F1 score for the churned class, giving the retention team an operationally appropriate cutoff rather than a statistically default one.

---

## Key Findings

### What a churner looks like
- New customer - most likely in the first 6 months
- On a month-to-month contract - no long-term commitment
- Using Fiber optic internet - counterintuitively, Fiber customers churn *more* than DSL, pointing to a pricing or service quality problem specific to that tier
- Paying by electronic check - the highest-churn payment method, suggesting lower trust or engagement
- Paying high monthly charges without protective add-ons (TechSupport, OnlineSecurity)

### Why Logistic Regression outperformed tree models
The feature set is largely binary after encoding - most columns are 0/1 flags. Logistic Regression is well-suited to this structure and generalises cleanly. Tree models can overfit to the encoded dummy variables, which likely explains the gap in cross-validation stability.

---

## Recommended Actions

**1. Early tenure intervention**
The first 6 months are the highest-risk window. An automated check-in or onboarding programme triggered at day 30 and day 90 would target the highest-probability churners at the point where intervention is most effective.

**2. Contract upgrade offers**
Month-to-month customers churn at 3–4× the rate of two-year contract holders. Offering a discount or incentive for contract upgrades - particularly to customers in tenure months 1–9 - is the single highest-ROI retention lever in the data.

**3. Investigate Fiber optic**
The Fiber optic churn signal is strong and counterintuitive - these are presumably higher-paying customers who are still leaving. The data does not tell us why, but the pattern is clear enough to warrant a customer satisfaction investigation specific to that tier.

**4. Bundle protective services into onboarding**
Customers without TechSupport or OnlineSecurity churn at higher rates. Including these in a default onboarding bundle (even as a free trial) increases both perceived value and retention probability.

---

## Dataset

| Field | Detail |
|---|---|
| Source | [IBM Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 20 (demographics, services, contract, billing) |
| Target | Churn (Yes/No) - 26.5% positive rate |
| Known issues | TotalCharges as string, 11 blank entries for new customers |

---

## Tools & Skills Demonstrated

| Area | Detail |
|---|---|
| Data Cleaning | String-to-numeric conversion, null imputation, categorical encoding |
| EDA | Distribution analysis, churn rate by category, correlation patterns |
| Modelling | Logistic Regression, Random Forest, Gradient Boosting - stratified CV |
| Evaluation | AUC-ROC, confusion matrix, classification report, precision-recall curve |
| Threshold Analysis | F1-optimal threshold selection for retention business context |
| Interpretation | Feature importance, coefficient analysis, actionable business framing |

---

---

## About

Built by **Charles Edeki**
📧 charlesedeki093@gmail.com | [LinkedIn](https://www.linkedin.com/in/charles-edeki/) | [GitHub](https://github.com/charlesedeki)
