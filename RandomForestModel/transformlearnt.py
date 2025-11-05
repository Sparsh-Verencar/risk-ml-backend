import joblib
import os

# Path to your saved transformers dictionary
transformer_path = r"D:\MODELTEST\trainedmodel\yeo_johnson_transformers.pkl"

if not os.path.exists(transformer_path):
    print("⚠️ Transformer file not found at:", transformer_path)
else:
    pt_dict = joblib.load(transformer_path)
    
    if isinstance(pt_dict, dict) and len(pt_dict) > 0:
        print(f"✅ Loaded {len(pt_dict)} Yeo–Johnson transformers.\n")
        print("=== Columns with learned transformations ===")
        for i, col in enumerate(pt_dict.keys(), start=1):
            print(f"{i}. {col}")
    else:
        print("ℹ️ No transformers found or file is empty.")
