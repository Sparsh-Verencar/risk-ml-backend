import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

# ===========================================================
# CONFIGURATION
# ===========================================================
model_path = r'C:\Users\shaun\Downloads\xgb_model_without_delay.pkl'
test_csv_path = r'C:\Users\shaun\Downloads\processed_dataset.csv'  # 🔹 Change to your CSV path
target_column = 'Supply_Risk_Flag'  # 🔹 Replace with your actual target column name

# ===========================================================
# LOAD TRAINED XGBOOST MODEL
# ===========================================================
xgb_model = joblib.load(model_path)
print("✅ XGBoost model loaded successfully!")

# ===========================================================
# LOAD TEST DATA
# ===========================================================
df = pd.read_csv(test_csv_path)
print(f"\n📘 Loaded test data: {df.shape[0]} rows, {df.shape[1]} columns")

# Split features and target
if target_column in df.columns:
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]
    print("🎯 Target column found and separated for evaluation.")
else:
    X_test = df
    y_test = None
    print("⚠️ No target column found — running predictions only (no accuracy metrics).")

# 🔹 Remove 'Delay_Days' if it exists (model was trained without it)
if 'Delay_Days' in X_test.columns:
    X_test = X_test.drop(columns=['Delay_Days'])
    print("📝 Removed 'Delay_Days' from test features to match model training data.")

# ===========================================================
# MAKE PREDICTIONS
# ===========================================================
predicted_class = xgb_model.predict(X_test)
predicted_proba = xgb_model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    'Predicted_Class': predicted_class,
    'Predicted_Probability': predicted_proba
})

print("\n=========== MODEL PREDICTIONS (first 10) ===========")
print(results.head(10))

# ===========================================================
# EVALUATE MODEL (if target is available)
# ===========================================================
if y_test is not None:
    accuracy = accuracy_score(y_test, predicted_class)
    f1 = f1_score(y_test, predicted_class)
    roc = roc_auc_score(y_test, predicted_proba)
    cm = confusion_matrix(y_test, predicted_class)

    print(f"\n🎯 Model Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predicted_class))

    print("\nConfusion Matrix:")
    print(cm)
else:
    print("\n⚠️ Skipping evaluation — no true labels provided in CSV.")

# ===========================================================
# FEATURE IMPORTANCES (XGBOOST)
# ===========================================================
importances = xgb_model.feature_importances_
features = X_test.columns
sorted_idx = np.argsort(importances)[::-1]

print("\n📊 Feature Importances (XGBoost):")
for i in sorted_idx:
    print(f"{features[i]}: {importances[i]:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), features[sorted_idx], rotation=45, ha="right")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()
