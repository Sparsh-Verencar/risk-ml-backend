import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# =====================================================
# 1️⃣ LOAD THE NEW MODEL (8 FEATURES)
# =====================================================
rf_model = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\random_forest_model.pkl')
scaler = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\scaler.pkl')
label_encoders = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\label_encoders.pkl')

print("✅ NEW Model loaded successfully!")

# =====================================================
# 2️⃣ LOAD TEST DATA
# =====================================================
test_data = pd.read_csv('preprocessed_supply_chain_resilience_dataset.csv')
print(f"Test data loaded — shape: {test_data.shape}")

# =====================================================
# 3️⃣ USE THE SAME 8 FEATURES AS TRAINING
# =====================================================
target_col = 'Supply_Risk_Flag'

# These should match EXACTLY what your training code uses
feature_cols = [
    'Product_Category', 
    'Quantity_Ordered', 
    'Shipping_Mode', 
    'Order_Value_USD', 
    'Historical_Disruption_Count', 
    'Supplier_Reliability_Score', 
    'Dominant_Buyer_Flag', 
    'Available_Historical_Records'
]

print(f"\n🔍 Using {len(feature_cols)} features:")
print(feature_cols)

# =====================================================
# 4️⃣ PREPARE TEST DATA (IDENTICAL TO TRAINING)
# =====================================================
X_test = test_data[feature_cols].copy()
y_test = test_data[target_col]

print(f"\n🔍 Test data prepared:")
print(f"Features: {X_test.shape}")
print(f"Target distribution: {y_test.value_counts().to_dict()}")

# =====================================================
# 5️⃣ APPLY PREPROCESSING (IDENTICAL TO TRAINING)
# =====================================================
# Handle missing values
categorical_cols = ['Product_Category', 'Shipping_Mode']
for col in categorical_cols:
    if X_test[col].isnull().any():
        X_test.loc[:, col] = X_test[col].fillna('None')
        print(f"✅ Filled missing values in {col}")

# Apply encoding
for col in categorical_cols:
    if col in label_encoders:
        le = label_encoders[col]
        X_test.loc[:, col] = X_test[col].astype(str)
        
        # Handle unseen categories
        valid_categories = set(le.classes_)
        mask = ~X_test[col].isin(valid_categories)
        if mask.any():
            replacement = le.classes_[0]
            X_test.loc[mask, col] = replacement
            print(f"⚠️ Replaced {mask.sum()} unseen values in '{col}'")
        
        X_test.loc[:, col] = le.transform(X_test[col])
        print(f"✅ Encoded: {col}")

# =====================================================
# 6️⃣ CHECK FEATURE DIMENSIONS BEFORE SCALING
# =====================================================
print(f"\n🔍 Feature dimensions check:")
print(f"X_test shape: {X_test.shape}")
print(f"Scaler expects: {scaler.n_features_in_} features")

if X_test.shape[1] != scaler.n_features_in_:
    print(f"❌ FEATURE MISMATCH: Test has {X_test.shape[1]} features, Scaler expects {scaler.n_features_in_}")
    print("This means your model was trained with different features!")
    exit()

# =====================================================
# 7️⃣ APPLY SCALING AND PREDICT
# =====================================================
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling applied successfully")

y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# 8️⃣ EVALUATE
# =====================================================
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*60)
print("📊 FINAL MODEL EVALUATION (8 FEATURES)")
print("="*60)
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ F1 Score:  {f1:.4f}")
print(f"✅ ROC-AUC:   {roc_auc:.4f}")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =====================================================
# 9️⃣ CHECK FOR BALANCED PREDICTIONS
# =====================================================
print(f"\n🔍 Prediction Distribution:")
print(f"Class 0 predictions: {sum(y_pred == 0)} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"Class 1 predictions: {sum(y_pred == 1)} ({(y_pred == 1).mean()*100:.1f}%)")

if sum(y_pred == 0) == 0:
    print("🚨 CRITICAL: Model is predicting ALL samples as Class 1!")
    print("This suggests the model training had issues.")
elif sum(y_pred == 1) == 0:
    print("🚨 CRITICAL: Model is predicting ALL samples as Class 0!")
else:
    print("✅ Model is predicting both classes (good!)")

# =====================================================
# 🔟 SAMPLE PREDICTIONS ANALYSIS
# =====================================================
print(f"\n📋 First 10 predictions:")
sample_results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Risk_Probability': [f"{p:.3f}" for p in y_pred_proba[:10]],
    'Correct': (y_test.values[:10] == y_pred[:10])
})
print(sample_results)

print(f"\n🎯 Performance Summary:")
print(f"Correct predictions: {sum(y_test.values == y_pred)}/{len(y_test)} ({accuracy*100:.1f}%)")

print("\n✅ TESTING COMPLETED SUCCESSFULLY!")