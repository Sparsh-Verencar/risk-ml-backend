import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD TRAINED MODEL ARTIFACTS
# ============================================================
rf_model = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\random_forest_model.pkl')
scaler = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\scaler.pkl')
label_encoders = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\label_encoders.pkl')
print("✓ Model, scaler, and encoders loaded successfully!")

# ============================================================
# PREPARE RAW NEW DATA FOR PREDICTION
# ============================================================
raw_new_data = [
   {
        'Product_Category': 'Machinery',
        'Shipping_Mode': 'Road',
        'Dominant_Buyer_Flag': 0,
        'Quantity_Ordered': 800,           # High quantity
        'Order_Value_USD': 80000,          # High value
        'Historical_Disruption_Count': 20, # Moderate disruptions  
        'Supplier_Reliability_Score': 0.3, # Poor reliability
        'Available_Historical_Records': 100 # Limited history
    },
    # This should be LOW RISK (despite some disruptions)
    {
        'Product_Category': 'Machinery', 
        'Shipping_Mode': 'Road',
        'Dominant_Buyer_Flag': 0,
        'Quantity_Ordered': 365,
        'Order_Value_USD': 34780.36,
        'Historical_Disruption_Count': 500, # High disruptions
        'Supplier_Reliability_Score': 0,  # Good reliability
        'Available_Historical_Records': 700 # Good track record
    },
    {
        'Product_Category': 'Machinery',
        'Shipping_Mode': 'Rail',
        'Dominant_Buyer_Flag': 1,
        'Quantity_Ordered': 142,
        'Order_Value_USD': 15320.08,
        'Historical_Disruption_Count': 17,
        'Supplier_Reliability_Score': 0.92,  # High reliability
        'Available_Historical_Records': 807
    }
]

new_data = pd.DataFrame(raw_new_data)

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================
categorical_cols = ['Product_Category', 'Shipping_Mode']

for col in categorical_cols:
    le = label_encoders[col]
    # Replace unseen categories with most frequent
    new_data[col] = new_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    new_data[col] = le.transform(new_data[col])
    print(f"Encoded {col}: {dict(zip(new_data[col], (raw_new_data[i][col] for i in range(len(new_data)))))}")


# ============================================================
# ALIGN FEATURES WITH TRAINING
# ============================================================
expected_features = list(scaler.feature_names_in_)
print(f"\nExpected features: {expected_features}")
print(f"Available features: {new_data.columns.tolist()}")

# Add any missing features as zero
for col in expected_features:
    if col not in new_data.columns:
        new_data[col] = 0
        print(f"Added missing feature: {col}")

# Reorder columns
X_new = new_data[expected_features]
print(f"\nFinal feature matrix shape: {X_new.shape}")

# ============================================================
# SCALE FEATURES AND PREDICT
# ============================================================
X_new_scaled = scaler.transform(X_new)
y_pred = rf_model.predict(X_new_scaled)
y_pred_proba = rf_model.predict_proba(X_new_scaled)[:, 1]

# ============================================================
# DISPLAY DETAILED PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("DETAILED MODEL PREDICTIONS")
print("="*60)

for i in range(len(y_pred)):
    risk_label = '🚨 HIGH RISK' if y_pred[i] == 1 else '✅ LOW RISK'
    risk_prob = y_pred_proba[i]
    
    print(f"\n📦 SAMPLE {i+1}:")
    print(f"   Prediction: {risk_label} (Probability: {risk_prob:.3f})")
    print(f"   Key Risk Factors:")
    
    # Show the actual values for this sample
    sample_data = raw_new_data[i]
    print(f"     - Historical Disruptions: {sample_data['Historical_Disruption_Count']}")
    print(f"     - Supplier Reliability: {sample_data['Supplier_Reliability_Score']:.2f}")
    print(f"     - Dominant Buyer: {'Yes' if sample_data['Dominant_Buyer_Flag'] else 'No'}")
    print(f"     - Order Value: ${sample_data['Order_Value_USD']:,.2f}")
    print(f"     - Product: {sample_data['Product_Category']}")
    print(f"     - Shipping: {sample_data['Shipping_Mode']}")
    
    # Risk interpretation
    if risk_prob > 0.7:
        print(f"   🔴 HIGH CONFIDENCE RISK")
    elif risk_prob > 0.3:
        print(f"   🟡 MEDIUM RISK")
    else:
        print(f"   🟢 LOW RISK")

print("\n" + "="*60)
print("RISK THRESHOLD ANALYSIS")
print("="*60)
print("Risk probability interpretation:")
print("  < 0.3: 🟢 LOW RISK")
print("  0.3-0.7: 🟡 MEDIUM RISK") 
print("  > 0.7: 🔴 HIGH RISK")

# Check the problematic sample specifically
problem_idx = 1
print(f"\n🔍 ANALYZING SAMPLE {problem_idx + 1} (Should be HIGH RISK):")
problem_sample = raw_new_data[problem_idx]
print(f"  - Historical Disruptions: {problem_sample['Historical_Disruption_Count']} ← EXTREMELY HIGH!")
print(f"  - Supplier Reliability: {problem_sample['Supplier_Reliability_Score']:.2f}")
print(f"  - Dominant Buyer: {'Yes' if problem_sample['Dominant_Buyer_Flag'] else 'No'}")
print(f"  - Actual Prediction: {y_pred[problem_idx]} (Probability: {y_pred_proba[problem_idx]:.3f})")

if y_pred[problem_idx] == 0 and y_pred_proba[problem_idx] < 0.5:
    print(f"  ❗ UNEXPECTED: This should be predicting RISK!")
    print(f"  Possible reasons:")
    print(f"    1. Model learned different risk patterns")
    print(f"    2. Other features are compensating")
    print(f"    3. Training data didn't have such extreme values")