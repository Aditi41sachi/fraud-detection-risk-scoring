# =========================
# FRAUD DETECTION PROJECT
# =========================

import joblib
import matplotlib.pyplot as plt
# 📦 Imports
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ================================
# Load Dataset
# ================================
file_path = "fraud_data - Sheet 1.csv"   # change path if needed
df = pd.read_csv(file_path)

X = df.drop(columns=["TransactionID", "IsFraud"])
y = df["IsFraud"]

cat_cols = ["Location", "MerchantCategory"]
num_cols = ["Amount", "Time", "CardHolderAge"]

# ================================
# Preprocessing
# ================================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# SMOTE for imbalance handling
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_prep, y_train)

# ================================
# Define Models
# ================================
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
svm = SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42)

# Random Forest with tuning
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, scoring="roc_auc",
                               cv=3, random_state=42, n_jobs=-1)

# XGBoost with tuning
xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=10, scoring="roc_auc",
                                cv=3, random_state=42, n_jobs=-1)

models = {
    "Logistic Regression": log_reg,
    "SVM": svm,
    "Random Forest (Tuned)": rf_search,
    "XGBoost (Tuned)": xgb_search
}

# ================================
# Train + Evaluate
# ================================
results = []
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train
    model.fit(X_train_bal, y_train_bal)
    
    # Best model if tuned
    best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    
    # Predict
    y_pred = best_model.predict(X_test_prep)
    y_prob = best_model.predict_proba(X_test_prep)[:, 1]
    
    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
        "ROC-AUC": roc_auc
    })
    
    # 🎨 Confusion Matrix (styled heatmap)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    
    # Save best model
    if name == "XGBoost (Tuned)":
        joblib.dump(best_model, "best_xgb_model.pkl")
    elif name == "Random Forest (Tuned)":
        joblib.dump(best_model, "best_rf_model.pkl")

# ================================
# Results
# ================================
results_df = pd.DataFrame(results)
print("\n🔎 Fraud Detection Model Comparison (with tuning):\n")
print(results_df.sort_values(by="ROC-AUC", ascending=False))

# 🎨 ROC Curve Comparison (styled)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison (Tuned Models)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# ================================
# Bar Chart Comparison
# ================================
metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]

plt.figure(figsize=(10, 6))
results_melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Fraud Detection Model Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)
plt.legend(title="Metric")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
