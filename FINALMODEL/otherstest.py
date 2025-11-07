import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

# ===========================================================
# CONFIGURATION
# ===========================================================
model_path = r'D:\MODELTEST\testtrain\normalized_random_forest.pkl'
test_csv_path = r'C:\Users\shaun\Downloads\processed_dataset.csv'
target_column = 'Supply_Risk_Flag'  # Change if needed

# ===========================================================
# LOAD TRAINED MODEL
# ===========================================================
xgb_model = joblib.load(model_path)
print("✅ XGBoost model loaded successfully!")

# ===========================================================
# GET TRAINED FEATURE NAMES
# ===========================================================
booster = xgb_model.get_booster()
trained_features = booster.feature_names
print("\n📋 Model was trained on the following features:")
print(trained_features)

# ===========================================================
# LOAD TEST DATA
# ===========================================================
df = pd.read_csv(test_csv_path)
print(f"\n📘 Loaded test data: {df.shape[0]} rows, {df.shape[1]} columns")

# ===========================================================
# COMPARE TRAINED FEATURES VS TEST CSV
# ===========================================================
test_features = list(df.columns)

if target_column in test_features:
    test_features.remove(target_column)

missing_in_test = [f for f in trained_features if f not in test_features]
extra_in_test = [f for f in test_features if f not in trained_features]

print("\n🔍 Feature comparison:")
print(f"✅ Columns expected by model: {len(trained_features)}")
print(f"🧩 Columns found in test CSV: {len(test_features)}")

if missing_in_test:
    print(f"⚠️ Missing in test data: {missing_in_test}")
else:
    print("✅ No missing features in test data.")

if extra_in_test:
    print(f"⚠️ Extra columns in test data: {extra_in_test}")
else:
    print("✅ No extra columns in test data.")

# ===========================================================
# PREPARE TEST DATA
# ===========================================================
if target_column in df.columns:
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]
    print("\n🎯 Target column found and separated for evaluation.")
else:
    X_test = df
    y_test = None
    print("\n⚠️ No target column found — running predictions only (no accuracy metrics).")

# Remove any extra columns not seen by the model
for col in extra_in_test:
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])
        print(f"📝 Removed extra column '{col}' not used during training.")

# Ensure order of columns matches the training order
X_test = X_test[trained_features]

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
# EVALUATE MODEL
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
# FEATURE IMPORTANCES
# ===========================================================
importances = xgb_model.feature_importances_
features = X_test.columns
sorted_idx = np.argsort(importances)[::-1]

print("\n📊 Feature Importances (XGBoost):")
for i in sorted_idx:
    print(f"{features[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), features[sorted_idx], rotation=45, ha="right")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()
