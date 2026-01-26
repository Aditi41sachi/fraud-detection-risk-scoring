# Fraud Detection Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Dataset](https://img.shields.io/badge/Dataset-Transaction%20Logs-green)

---

## 📌 Overview

This project implements an end-to-end **fraud detection machine learning pipeline**, covering data exploration, feature engineering, model training, evaluation, and final model selection. The objective is to identify fraudulent transactions using supervised learning techniques and to determine an optimal decision threshold that balances fraud detection effectiveness with false-positive control.

The project is structured as a series of modular Jupyter notebooks that follow a clear, logical workflow, emphasizing reproducibility, clarity, and real‑world machine learning practices.

```
Raw Transaction Data
        ↓
Exploratory Data Analysis (EDA)
        ↓
Business-Oriented Insights
        ↓
Feature Engineering
        ↓
Model Training & Evaluation
        ↓
Prediction & Risk Scoring Dataset
```
---

## ✨ Features

- End-to-end machine learning pipeline for fraud detection  
- Exploratory Data Analysis (EDA) with visual insights  
- Business-oriented fraud pattern analysis  
- Feature engineering including encoding and scaling  
- Training and comparison of multiple machine learning models  
- Decision threshold optimization for precision–recall trade-off  
- Persistent storage of trained models and optimized thresholds
- Risk scoring and error-based evaluation (TP / FP / FN / TN)
- Modular, notebook-driven workflow for clarity and reproducibility  

## 📂 Project Structure

```
fraud_detection/
│
├── 01_data_exploration.ipynb       # Initial data analysis and cleaning
├── 02_business_insights.ipynb      # Exploratory insights and observations
├── 03_feature_engineering.ipynb    # Feature creation and preprocessing
├── 04_model_training.ipynb         # Model training and comparison
├── 05_model_evaluation.ipynb       # Evaluation, threshold tuning, final selection
│
└── fraud_data - Sheet 1.csv        # Raw dataset

```
---

## 📓 Notebook Workflow

The project follows a structured, notebook-based workflow:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_exploration.ipynb` | Load data, inspect structure, handle missing values, and perform initial visual analysis |
| 2 | `02_business_insights.ipynb` | Analyze fraud patterns and extract key business insights |
| 3 | `03_feature_engineering.ipynb` | Perform feature transformations, encoding, scaling, and train-test split |
| 4 | `04_model_training.ipynb` | Train and tune machine learning models (Logistic Regression, Random Forest, XGBoost) |
| 5 | `05_model_evaluation.ipynb` | Compare models, evaluate metrics, optimize decision threshold, and select the final model |

---

> **Note**  
>The following files and folders are created automatically when the notebooks are executed and are intentionally excluded from version control:
>- `processed/` – train/test datasets and engineered features
>- `models/` – trained models and optimized decision threshold
>- `fraud_predictions.csv` – final prediction, risk scoring, and evaluation dataset
>
>This keeps the repository clean while ensuring full reproducibility.

## 📊 Dataset
- **File:** `fraud_data - Sheet 1.csv`
- **Description:** Transaction-level data containing both legitimate and fraudulent records.
- **Target Variable:** Binary fraud indicator (fraud vs. non-fraud)
- **Challenges:**
  - Class imbalance
  - Mixed numerical and categorical features
  - Real-world noise and outliers
The dataset is used **as-is for EDA** and then processed during feature engineering.

## 📊 Prediction Dataset (`fraud_predictions.csv`)
- The prediction dataset is generated after model evaluation and contains **only test-set predictions (unseen data)** to ensure unbiased evaluation and avoid data leakage.
- It includes:
   - Original feature values
   - Actual fraud labels
   - Predicted fraud labels
   - Fraud probability scores
   - Risk level classification
   - Evaluation results (TP / FP / FN / TN)

> **Note:** The prediction dataset has fewer records than the raw dataset because it represents the **test split only**, reflecting realistic model performance.

## 📊 Models
The project trains and evaluates multiple models:
- **Logistic Regression**
- **Random Forest**
- **XGBoost (Final Selected Model)**

Model selection criteria included:
- Precision, recall, F1-score, ROC-AUC
- Business-oriented cost optimization

After execution, trained models are saved locally along with the optimized decision threshold for fraud detection.

## 📈 Model Evaluation & Analysis

Model evaluation focuses on both statistical performance and business impact:

* Confusion matrix analysis (TP, FP, FN, TN)
* False positive vs false negative tradeoff analysis
* Fraud probability–based risk scoring
* Error distribution across transaction attributes (e.g., location, category)

All evaluation insights are embedded directly in `fraud_predictions.csv`, enabling transparent downstream analysis.

## 🎯 Risk Segmentation

Each transaction is categorized into a risk level based on predicted fraud probability:

* **Low Risk**
* **Medium Risk**
* **High Risk**

This segmentation supports business-friendly interpretation and prioritization of fraud investigation efforts.

## 📊 Analytical & Dashboard Readiness

The prediction dataset is designed as a **single, clean analytical table** suitable for visualization or reporting tools. It supports:

* Confusion matrix visualization
* FP vs FN tradeoff analysis
* Risk-level vs error analysis
* Location- or category-based error distribution

## ⚙️ Installation

1. **Prerequisites**

* Python **3.8+**
* `pip` or `conda`
* Jupyter Notebook or JupyterLab

2.**Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

3.**Using `requirements.txt`**

```bash
pip install -r requirements.txt
```


## ▶️ How to Run the Project

**Step 1: Clone the Repository**

```bash
git clone <repository-url>
cd fraud_detection
```

**Step 2: Start Jupyter Notebook**

```bash
jupyter notebook
```

**Step 3: Run Notebooks in Order**

Execute the notebooks sequentially:

1. `01_data_exploration.ipynb`
2. `02_business_insights.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_training.ipynb`
5. `05_model_evaluation.ipynb`

**Step 4: Outputs**

After successful execution:
* Processed datasets are saved in `processed/`
* Trained models and the optimized threshold are saved in `models/`
* * `fraud_predictions.csv` is generated as the final prediction, risk scoring, and evaluation dataset

## 📈 Results

* **XGBoost** achieved the best overall performance.
* Decision threshold tuning significantly improved fraud recall.
* The final model balances fraud detection effectiveness with false-positive control.

## ⚡ For Experienced Users

If you are already familiar with ML workflows, you can:

* Skip directly to `03_feature_engineering.ipynb`
* Modify hyperparameters in `04_model_training.ipynb`
* Adjust classification thresholds in `05_model_evaluation.ipynb`
* Integrate trained models into downstream applications (API, batch scoring)
* Replace the dataset with a new fraud dataset (same schema required)

## 🚀 Future Improvements

* Model monitoring and retraining for data drift
* API deployment using FastAPI or Flask
* Model explainability with SHAP or LIME
* Experiment tracking with MLflow
* Pipeline automation using CI/CD

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.
