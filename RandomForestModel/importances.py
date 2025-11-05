import joblib

# Load trained model and scaler
rf_model = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\random_forest_model.pkl')
scaler = joblib.load(r'D:\risk-ml-backend\RandomForestModel\trainedmodel\scaler.pkl')

# Get the feature names in exact order used for training
feature_names = list(scaler.feature_names_in_)

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Print importances by feature name
print("=== Feature Importances ===")
for fname, imp in zip(feature_names, importances):
    print(f"{fname:30s}: {imp:.6f}")
