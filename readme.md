# 🫁 Lung Cancer Risk Pre-Screening Dashboard

An AI-powered clinical pre-screening dashboard built with **Streamlit** and **Scikit-Learn**. This application utilizes an **Ensemble Consensus Strategy** to evaluate patient survey responses (habits, medical history, and physical symptoms), mapping them to calculated risk categories via a dynamic tricolor indicator light system.

---

## 📊 The Machine Learning Core

This project evaluates six distinct algorithmic structures (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Support Vector Machine`, `K-Nearest Neighbors`, and `Gaussian Naive Bayes`) trained on clinical survey profiles balanced via **SMOTE (Synthetic Minority Over-sampling Technique)**.

### The Optimization Framework: Recall vs. Precision
In healthcare applications, **False Negatives** (missing a true cancer patient) carry far worse consequences than **False Positives** (causing temporary alarm cleared by a CT scan). Therefore, our pipeline implements a dual-model framework combining two contrasting strengths:
1. **Naive Bayes (The Clinical Safety Net):** Selected for peak **Recall (95.83%)** to capture the maximum number of true cases.
2. **Random Forest (The Precision Anchor):** Selected for peak **Precision (100.00%)** and **Accuracy (94.64%)** to minimize false alarms.

---

## 🛠️ Project Architecture & Consensus System

When a user submits a clinical profile through the Streamlit interface:
* **`StandardScaler` Validation:** Continuous numeric variables like `AGE` are scaled dynamically using the underlying training statistics to avoid data leakage.
* **Consensus Pipeline:** The features are sent to *both* algorithms simultaneously.
* **Tricolor Risk Engine:**
  * 🔴 **Red Light (High Risk):** Both models identify a severe risk profile. Immediate medical evaluation recommended.
  * 🟡 **Yellow Light (Borderline Risk):** The hypersensitive screening model flags indicators while the strict precision model clears them. Routine clinical review advised.
  * 🟢 **Green Light (Low Risk):** Both models clear the response metrics.

---

## 🚀 Getting Started Locally

### Prerequisites
Make sure you have Python 3.9+ installed. Clone or copy this repository into your working directory.

### 1. Installation
Install the required data science packages:
```bash
pip install streamlit pandas numpy scikit-learn