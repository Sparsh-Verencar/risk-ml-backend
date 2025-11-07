import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# LOAD TRAINED MODEL
# ===========================================================
model_path = r'D:\Risk_ml_backend\risk-ml-backend\FINALMODEL\RandomForest\normalized_random_forest.pkl'
rf_model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# ===========================================================
# DEFINE FEATURE COLUMNS (must match training!)
# ===========================================================
feature_cols = [
    'Buyer_ID', 'Supplier_ID', 'Product_Category', 'Quantity_Ordered',
    'Order_Date', 'Dispatch_Date', 'Delivery_Date', 'Shipping_Mode',
    'Order_Value_USD', 'Disruption_Type', 'Disruption_Severity',
    'Historical_Disruption_Count', 'Supplier_Reliability_Score',
    'Organization_ID', 'Dominant_Buyer_Flag', 'Available_Historical_Records'
]

# ===========================================================
# FEATURE IMPORTANCE
# ===========================================================
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=========== FEATURE IMPORTANCES ===========")
print(importances)

# ===========================================================
# VISUALIZE TOP FEATURES
# ===========================================================
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'][:10], importances['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
