# 🫁 Lung Cancer Risk Prediction Portal

An end-to-end Machine Learning pipeline and interactive Streamlit web application designed to predict lung cancer risk profiles using optimized behavioral and clinical survey factors. 

This project solves a critical small-dataset challenge ($N=276$) where unscaled continuous features (`AGE`) initially caused severe model overfitting and baseline majority-class guessing. Through robust feature scaling, synthetic oversampling (SMOTE), and tree-depth regularization, the final pipeline delivers highly accurate, balanced, and clinically generalized predictions led by an optimized **XGBoost** engine.

## 🚀 Performance Metrics (Test Set Validation)

| Model Engine | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Optimized)** | **91.30%** | **94.92%** | **94.92%** | **94.92%** | **95.34%** |
| Support Vector Machine | 91.30% | 96.49% | 93.22% | 94.83% | 94.92% |
| Logistic Regression | 91.30% | 100.00% | 89.83% | 94.64% | 96.10% |
| Random Forest | 88.41% | 94.74% | 91.53% | 93.10% | 96.19% |

## 🛠️ Key Pipeline Features
* **Anti-Overfitting Design:** Resolves extreme feature weight dominance by scaling continuous age attributes relative to binary survey features using `StandardScaler`.
* **Imbalance Correction:** Utilizes `SMOTE` strictly within the training boundary to prevent synthetic data leakage into the test set.
* **Compact Model Packaging:** Bundles the best-performing models, feature array maps, and scaling configurations into a single compressed asset (`lung_cancer_models.pkl`) for immediate server side decompression.
* **Production UI:** An interactive web portal built with Streamlit featuring real-time risk calculations, individual model querying, and a cross-engine model consensus matrix.

## 📁 Repository Structure
```text
├── survey lung cancer.csv     # Raw clinical survey dataset
├── lung_cancer_pipeline.ipynb # Jupyter Notebook containing EDA, preprocessing, & training
├── lung_cancer_models.pkl     # Single serialized deployment artifact (Scaler + Models)
└── app.py                     # Streamlit web application source code