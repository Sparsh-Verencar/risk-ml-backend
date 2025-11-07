# ===================================================
# Check skewness of numerical attributes in CSV
# ===================================================

import pandas as pd

# ---- Load your CSV file ----
file_path = r"D:\MODELTEST\preprocessed_supply_chain_resilience_dataset.csv"
data = pd.read_csv(file_path)

# ---- Select only numeric columns ----
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# ---- Calculate skewness ----
skewness = numeric_data.skew()

# ---- Sort by absolute skewness (most skewed first) ----
skewness_sorted = skewness.reindex(skewness.abs().sort_values(ascending=False).index)

# ---- Display results ----
print("=== Skewness of Numeric Columns ===")
print(skewness_sorted)

# ---- Optional: Flag highly skewed columns ----
print("\n=== Highly Skewed Columns (|skew| > 1) ===")
highly_skewed = skewness_sorted[abs(skewness_sorted) > 1]
print(highly_skewed)
