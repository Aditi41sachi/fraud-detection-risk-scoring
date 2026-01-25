# 🕵️ Fraud Detection Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fraud%20Detection-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

## 📌 Overview

This project implements an end-to-end **fraud detection machine learning pipeline**, covering data exploration, feature engineering, model training, evaluation, and final model selection. The goal is to identify fraudulent transactions using supervised learning techniques and to determine an optimal decision threshold for classification.

The project is structured as a series of Jupyter notebooks that follow a logical workflow, making it suitable for academic review, technical assessment, or portfolio demonstration.

---

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

> 📌 **Note**  
> The following folders will be **automatically created after running the notebooks**:
> - `processed/` → train/test datasets  
> - `models/` → trained models and selected decision threshold  

---

## 📊 Dataset
- **File:** `fraud_data - Sheet 1.csv`
- **Description:** Transaction-level data containing both legitimate and fraudulent records.
- **Target Variable:** Binary fraud indicator (fraud vs. non-fraud)
- **Challenges:**
  - Class imbalance
  - Mixed numerical and categorical features
  - Real-world noise and outliers

The dataset is used **as-is for EDA** and then processed during feature engineering.
---

## 🧠 Workflow

The notebooks should be run in the following order:

1. **01_data_exploration.ipynb**

   * Data loading
   * Missing value analysis
   * Basic statistics and visualizations

2. **02_business_insights.ipynb**

   * Exploratory findings
   * Fraud patterns and observations

3. **03_feature_engineering.ipynb**

   * Feature transformation
   * Encoding and scaling
   * Train/test split

4. **04_model_training.ipynb**

   * Model training (Logistic Regression, XGBoost)
   * Hyperparameter tuning

5. **05_model_evaluation.ipynb**

   * Performance comparison
   * Precision/Recall trade-off
   * Threshold optimization
   * Final model selection

---

## 📊 Models
The project trains and evaluates multiple models:
- **Logistic Regression**
- **Random Forest**
- **XGBoost (Final Selected Model)**

After execution, trained models are saved locally along with the optimized decision threshold for fraud detection.

---

## ⚙️ Installation

### Prerequisites

* Python **3.8+**
* `pip` or `conda`
* Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Using `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fraud_detection
```

### Step 2: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Run Notebooks in Order

Execute the notebooks sequentially:

1. `01_data_exploration.ipynb`
2. `02_business_insights.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_training.ipynb`
5. `05_model_evaluation.ipynb`

### Step 4: Outputs

After successful execution:

* Processed datasets are saved in `processed/`
* Trained models and the optimized threshold are saved in `models/`

---

## 📈 Results

* **XGBoost** achieved the best overall performance.
* Decision threshold tuning significantly improved fraud recall.
* The final model balances fraud detection effectiveness with false-positive control.

---

## ⚡ For Experienced Users

If you are already familiar with ML workflows, you can:

* Skip directly to `03_feature_engineering.ipynb`
* Modify hyperparameters in `04_model_training.ipynb`
* Adjust classification thresholds in `05_model_evaluation.ipynb`
* Integrate trained models into downstream applications (API, batch scoring)
* Replace the dataset with a new fraud dataset (same schema required)

---

## 🚀 Future Improvements

* Model monitoring and retraining for data drift
* API deployment using FastAPI or Flask
* Model explainability with SHAP or LIME
* Experiment tracking with MLflow
* Pipeline automation using CI/CD

---

## 📝 Author

Prepared as a fraud detection machine learning project for analysis, evaluation, and demonstration purposes.

---

## 📄 License

This project is intended for **educational and research purposes only**.

---

*Prepared as a fraud detection machine learning project for analysis, evaluation, and demonstration purposes.*
