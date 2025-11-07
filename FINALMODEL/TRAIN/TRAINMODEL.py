import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# LOAD DATA
# ===========================================================
df = pd.read_csv(r'C:\Users\shaun\Downloads\processed_dataset.csv')
print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# ===========================================================
# DEFINE FEATURES AND TARGET
# ===========================================================
target_col = 'Supply_Risk_Flag'

# Exclude 'Delay_Days' from training
exclude_cols = ['Delay_Days', target_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

print(f"\nFeature count (excluding Delay_Days): {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")

# ===========================================================
# TRAIN-TEST SPLIT
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split:")
print(f"  Train: {X_train.shape[0]} rows")
print(f"  Test:  {X_test.shape[0]} rows")

# ===========================================================
# RANDOM FOREST MODEL
# ===========================================================
print("\n🚀 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
print("✅ Model training complete!")

# ===========================================================
# EVALUATION
# ===========================================================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("\n=========== MODEL EVALUATION ===========")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC:  {roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===========================================================
# FEATURE IMPORTANCE
# ===========================================================
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importances.head(10))

# ===========================================================
# SAVE MODEL
# ===========================================================
save_path = r'D:\MODELTEST\testtrain\normalized_random_forest.pkl'
joblib.dump(rf_model, save_path)
print(f"\n💾 Model saved successfully to {save_path}")
