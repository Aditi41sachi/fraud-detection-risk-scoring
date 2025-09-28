# Fraud Detection Project

This project implements a **fraud detection system** using machine learning models. It includes data preprocessing, handling class imbalance, model training with hyperparameter tuning, evaluation, and visualization of results.

---

## 📂 Project Structure

fraud_detection_project/
│
├─ fraud_data - Sheet 1.csv # Dataset file
├─ best_xgb_model.pkl # Saved XGBoost model
├─ best_rf_model.pkl # Saved Random Forest model
└─ README.md # Project documentation



---

## 🧰 Libraries Used

- `pandas` – Data manipulation
- `numpy` – Numerical computations
- `matplotlib` & `seaborn` – Visualization
- `scikit-learn` – Preprocessing, model building, evaluation
- `xgboost` – Gradient boosting model
- `imblearn` – SMOTE for handling class imbalance
- `joblib` – Model serialization

---

## ⚙️ Steps Implemented

### 1. Load Dataset
The dataset `fraud_data - Sheet 1.csv` contains transactions with the target column `IsFraud`.

```python
X = df.drop(columns=["TransactionID", "IsFraud"])
y = df["IsFraud"]

2. Preprocessing
Numerical features: missing values imputed with median and scaled.

Categorical features: missing values imputed with most frequent value and encoded using one-hot encoding.

Combined using ColumnTransformer.

3. Train-Test Split
Split dataset into training (80%) and testing (20%) sets.

Stratified split to maintain class distribution.

4. Handle Class Imbalance
Applied SMOTE to oversample minority class in the training set.

5. Model Definition & Hyperparameter Tuning
Logistic Regression

SVM

Random Forest with RandomizedSearchCV

XGBoost with RandomizedSearchCV

6. Model Training & Evaluation
Evaluated on Precision, Recall, F1-Score, ROC-AUC

Confusion matrices plotted for each model.

ROC curves compared across all models.

7. Save Best Models
Best Random Forest and XGBoost models are saved as .pkl files for future use.

## 📊 Results
Models are compared using metrics and visualizations.

Bar chart and ROC curve plots provide an overview of model performance.

##💡 Key Features
Automated preprocessing pipeline

SMOTE for class imbalance

Hyperparameter tuning for Random Forest and XGBoost

Evaluation using multiple metrics and visualizations

Model persistence with joblib

## 🔗 How to Run
Clone the repository and place fraud_data - Sheet 1.csv in the folder.

Install dependencies:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib


## ⚠️ Notes
Ensure the dataset columns match the code (TransactionID, IsFraud, Location, MerchantCategory, Amount, Time, CardHolderAge).

Hyperparameter tuning may take time depending on dataset size and computing resources.

SMOTE is applied only to the training set to avoid data leakage.

## 📝 Author
Your Name – GitHub Profile

## 📌 References
Scikit-learn Documentation

XGBoost Documentation

imbalanced-learn Documentation