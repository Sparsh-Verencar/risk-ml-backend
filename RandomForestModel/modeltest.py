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
        'Product_Category': 'Food',
        'Shipping_Mode': 'Air',
        'Dominant_Buyer_Flag': 0,
        'Quantity_Ordered': 469,
        'Order_Value_USD': 36273.99,
        'Historical_Disruption_Count': 1,
        'Supplier_Reliability_Score': 0.05,
        'Available_Historical_Records': 900,
    },
    {
        'Product_Category': 'Machinery',
        'Shipping_Mode': 'Road',
        'Dominant_Buyer_Flag': 0,
        'Quantity_Ordered': 365,
        'Order_Value_USD': 34780.36,
        'Historical_Disruption_Count': 1,
        'Supplier_Reliability_Score': 0.1,
        'Available_Historical_Records': 909
    },
    {
        'Product_Category': 'Food',
        'Shipping_Mode': 'Rail',
        'Dominant_Buyer_Flag': 0,
        'Quantity_Ordered': 365,
        'Order_Value_USD': 7154.54,
        'Historical_Disruption_Count': 19,
        'Supplier_Reliability_Score': 0.8,
        'Available_Historical_Records': 9000
    },
    {
        'Product_Category': 'Machinery',
        'Shipping_Mode': 'Rail',
        'Dominant_Buyer_Flag': 1,
        'Quantity_Ordered': 142,
        'Order_Value_USD': 15320.08,
        'Historical_Disruption_Count': 17,
        'Supplier_Reliability_Score': 0.92,
        'Available_Historical_Records': 807
    }
]

new_data = pd.DataFrame(raw_new_data)

# ============================================================
# ENCODE CATEGORICAL FEATURES (matching training process)
# ============================================================
categorical_cols = ['Product_Category', 'Shipping_Mode']

for col in categorical_cols:
    le = label_encoders[col]
    # Replace unseen categories with 'None'
    new_data[col] = new_data[col].apply(lambda x: x if x in le.classes_ else 'None')
    if 'None' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'None')
    # Transform
    new_data[col] = le.transform(new_data[col])

# ============================================================
# ALIGN FEATURES WITH TRAINING
# ============================================================
expected_features = list(scaler.feature_names_in_)

# Add any missing features as zero
for col in expected_features:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns
X_new = new_data[expected_features]

# ============================================================
# SCALE FEATURES AND PREDICT
# ============================================================
X_new_scaled = scaler.transform(X_new)
y_pred = rf_model.predict(X_new_scaled)
y_pred_proba = rf_model.predict_proba(X_new_scaled)[:, 1]

# ============================================================
# DISPLAY PREDICTIONS
# ============================================================
print("\n=== MODEL PREDICTIONS ===")
for i in range(len(y_pred)):
    label = 'RISK' if y_pred[i] == 1 else 'NO RISK'
    print(f"Sample {i+1}: {label} (Probability of Risk = {y_pred_proba[i]:.3f})")
