import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# Load model artifacts
# -----------------------------
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# -----------------------------
# Prepare a base sample
# -----------------------------
base_sample = {
    'Product_Category': 'Food',
    'Shipping_Mode': 'Rail',
    'Dominant_Buyer_Flag': 0,
    'Quantity_Ordered': 365,
    'Order_Value_USD': 7154.54,
    'Historical_Disruption_Count': 19,
    'Supplier_Reliability_Score': 0.8,
    'Available_Historical_Records': 9000
}

# -----------------------------
# Choose a feature to vary
# -----------------------------
feature_to_vary = 'Supplier_Reliability_Score'
values = np.linspace(0, 1, 50)  # Example: vary from 0 to 1

probs = []
for val in values:
    sample = base_sample.copy()
    sample[feature_to_vary] = val
    
    # Convert to DataFrame
    df_sample = pd.DataFrame([sample])
    
    # Encode categorical features
    categorical_cols = ['Product_Category', 'Shipping_Mode']
    for col in categorical_cols:
        le = label_encoders[col]
        df_sample[col] = df_sample[col].apply(lambda x: x if x in le.classes_ else 'None')
        if 'None' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'None')
        df_sample[col + '_encoded'] = le.transform(df_sample[col])
        df_sample.drop(columns=[col], inplace=True)
    
    # Ensure all expected columns are present
    expected_features = list(scaler.feature_names_in_)
    for col in expected_features:
        if col not in df_sample.columns:
            df_sample[col] = 0
    df_sample = df_sample[expected_features]
    
    # Scale and predict
    X_scaled = scaler.transform(df_sample)
    prob = rf_model.predict_proba(X_scaled)[:, 1][0]
    probs.append(prob)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,5))
sns.lineplot(x=values, y=probs)
plt.xlabel(feature_to_vary)
plt.ylabel('Predicted Probability of Risk')
plt.title(f'Effect of {feature_to_vary} on Predicted Risk')
plt.grid(True)
plt.show()
