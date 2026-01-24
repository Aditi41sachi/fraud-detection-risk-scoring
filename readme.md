# Fraud Detection Project

## Overview

This project implements an end-to-end **fraud detection machine learning pipeline**, covering data exploration, feature engineering, model training, evaluation, and final model selection. The goal is to identify fraudulent transactions using supervised learning techniques and to determine an optimal decision threshold for classification.

The project is structured as a series of Jupyter notebooks that follow a logical workflow, making it suitable for academic review, technical assessment, or portfolio demonstration.

---

## Project Structure

```
fraud_detection/
│
├── fraud_data - Sheet 1.csv        # Raw dataset
│
├── 01_data_exploration.ipynb       # Initial data analysis and cleaning
├── 02_business_insights.ipynb      # Exploratory insights and observations
├── 03_feature_engineering.ipynb    # Feature creation and preprocessing
├── 04_model_training.ipynb         # Model training and comparison
├── 05_model_evaluation.ipynb       # Evaluation, threshold tuning, final selection
│
├── models/
│   ├── best_logistic_model.pkl     # Trained Logistic Regression model
│   └── best_xgb_model.pkl          # Trained XGBoost model (final model)
│
└── final_threshold.txt             # Selected probability threshold for classification
```

---

## Dataset

* **File:** `fraud_data - Sheet 1.csv`
* The dataset contains transaction-level data with a binary target indicating whether a transaction is fraudulent.
* Preprocessing steps and feature definitions are documented within the notebooks.

> Note: A formal data dictionary is not provided; feature meanings can be inferred from the exploration and feature engineering notebooks.

---

## Workflow

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

## Final Model

* **Selected Model:** XGBoost Classifier
* **Saved Model:** `models/best_xgb_model.pkl`
* **Decision Threshold:** Stored in `final_threshold.txt`

Predictions should be generated as probabilities and converted to class labels using the saved threshold.

---

## How to Use the Model

Example workflow:

1. Load the trained model from `models/best_xgb_model.pkl`
2. Prepare input data using the same features and preprocessing steps
3. Generate predicted probabilities
4. Apply the threshold from `final_threshold.txt` to classify transactions

---

## Requirements

This project requires Python and common data science libraries. A sample dependency list:

* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
* seaborn

> It is recommended to create a virtual environment and install dependencies before running the notebooks.

---

## Notes and Limitations

* Models are saved using Python pickle files, which are dependent on library versions.
* The project is notebook-driven and intended for analysis and demonstration purposes.
* The processed datasets are included for convenience and reproducibility.

---

## Future Improvements

* Add a formal data dictionary
* Provide a standalone inference script
* Add dependency pinning via `requirements.txt`
* Improve model deployment readiness

---

## Author

*Prepared as a fraud detection machine learning project for analysis, evaluation, and demonstration purposes.*
