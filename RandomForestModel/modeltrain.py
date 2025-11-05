import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# LOAD DATA
# ====================================================================
df = pd.read_csv(r'D:\risk-ml-backend\RandomForestModel\preprocessed_supply_chain_resilience_dataset.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# ====================================================================
# DEFINE COLUMNS
# ====================================================================
target_col = 'Supply_Risk_Flag'
exclude_cols = ['Delay_Days', 'Disruption_Type', 'Disruption_Severity']  # Excluded
date_cols = ['Order_Date', 'Dispatch_Date', 'Delivery_Date']
id_cols = ['Buyer_ID', 'Supplier_ID', 'Organization_ID']

# Get feature columns (excluding target, dates, IDs, and excluded)
all_cols = df.columns.tolist()
feature_cols = [col for col in all_cols 
                if col not in date_cols + id_cols + [target_col] + exclude_cols]

# Separate numerical and categorical features
numerical_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

print(f"\nFeatures to use: {len(feature_cols)}")
print(f"  - Numerical: {len(numerical_cols)}")
print(f"  - Categorical: {len(categorical_cols)}")
print(f"\nExcluded columns: {exclude_cols}")

# ====================================================================
# DATA PREPROCESSING
# ====================================================================
df_model = df.copy()

# Handle missing values in categorical columns (fill with 'None')
for col in categorical_cols:
    if df_model[col].isnull().sum() > 0:
        df_model[col] = df_model[col].fillna('None')
        print(f"Filled {df_model[col].isnull().sum()} missing values in {col}")

# Encode categorical variables using Label Encoding (overwrite original columns)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {df_model[col].nunique()} unique categories")

# ====================================================================
# PREPARE FEATURES AND TARGET
# ====================================================================
X = df_model[feature_cols]
y = df_model[target_col]

print(f"\nFinal feature set: {X.shape[1]} features")
print("\nTarget variable distribution:")
print(y.value_counts())
print(f"  Class 0 (No Risk): {(y==0).sum()/len(y)*100:.1f}%")
print(f"  Class 1 (Risk): {(y==1).sum()/len(y)*100:.1f}%")

# ====================================================================
# TRAIN-TEST SPLIT WITH STRATIFICATION
# ====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nData split:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# ====================================================================
# FEATURE SCALING
# ====================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling applied (StandardScaler)")

# ====================================================================
# RANDOM FOREST MODEL
# ====================================================================
print("\nTraining Random Forest model...")
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
rf_model.fit(X_train_scaled, y_train)
print("✓ Model training completed!")

# ====================================================================
# MODEL EVALUATION
# ====================================================================
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)
y_train_proba = rf_model.predict_proba(X_train_scaled)[:,1]
y_test_proba = rf_model.predict_proba(X_test_scaled)[:,1]

print("\n--- Training Set Performance ---")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_train, y_train_proba):.4f}")

print("\n--- Test Set Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_test_proba):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# ====================================================================
# FEATURE IMPORTANCE
# ====================================================================
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop features:")
print(feature_importance.head(10))

# ====================================================================
# SAVE ARTIFACTS
# ====================================================================
import joblib
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\n✓ Model artifacts saved to disk")
