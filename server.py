from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

# =====================================================
# ✅ CORS
# =====================================================
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ✅ Load Model & Schema
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

# Load models
rfModel = joblib.load(BASE_DIR / "FINALMODEL/RandomForest/normalized_random_forest.pkl")
xgModel = joblib.load(BASE_DIR / "FINALMODEL/XGBOOST/xgb_model_without_delay.pkl")

# =====================================================
# ✅ Load Mappings from training data
# These are numeric/scaled values for all categorical columns
# You must create these mappings once from your processed dataset
# Example: {'B33': 0.530612245, 'B34': 0.612244898, ...}
# =====================================================
# Load processed dataset
DATA_PATH = BASE_DIR / "FINALMODEL/processed_dataset.csv"
df_processed = pd.read_csv(DATA_PATH)

columns_to_drop = [
    "Communication_Cost_MB",
    "Parameter_Change_Magnitude",
    "Energy_Consumption_Joules",
    "Federated_Round",
    "Order_ID",
    "Data_Sharing_Consent",
    "Delay_Days",
    "Supply_Risk_Flag"
]

df_processed = df_processed.drop(columns=columns_to_drop, errors="ignore")

# Detect numeric columns (all columns are numeric after processing)
numeric_cols = df_processed.columns.tolist()

# Build mapping dictionaries for categorical-like columns
categorical_maps = {}
categorical_like_cols = [
    "Buyer_ID",
    "Supplier_ID",
    "Product_Category",
    "Order_Date",
    "Dispatch_Date",
    "Delivery_Date",
    "Shipping_Mode",
    "Disruption_Type",
    "Disruption_Severity",
    "Organization_ID"
]

for col in categorical_like_cols:
    unique_vals = df_processed[col].unique()
    # Map each raw value (from df_processed as string) to numeric value
    mapping = dict(zip(df_processed[col].astype(str).unique(), unique_vals))
    categorical_maps[col] = mapping


print("✅ Model loaded and mappings prepared.")

# =====================================================
# ✅ Routes
# =====================================================
@app.get("/")
def root():
    return {"message": "API running 🚀"}

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        data = body.get("data", body)
        df = pd.DataFrame([data])

        # Drop unused columns
        drop_cols = [
            "Communication_Cost_MB",
            "Parameter_Change_Magnitude",
            "Energy_Consumption_Joules",
            "Federated_Round",
            "Order_ID",
            "Data_Sharing_Consent",
            "Delay_Days",
            "Supply_Risk_Flag"
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Map categorical-like columns to numeric values
        for col, mapping in categorical_maps.items():
            if col in df.columns:
                raw_val = str(df.at[0, col])
                df.at[0, col] = mapping.get(raw_val, 0)  # default 0 if missing

        # Ensure all numeric columns exist
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Reorder columns exactly as model expects
        df = df[numeric_cols]

        print("\n✅ INPUT DF READY FOR PREDICTION:")
        print(df)

        # Predict
        rf_pred = rfModel.predict(df)[0]
        xgb_pred = xgModel.predict(df)[0]

        print("\n✅ PREDICTIONS:")
        print("RandomForest:", rf_pred)
        print("XGBoost:", xgb_pred)

        return {
            "status": "success ✅",
            "processed_input": df.to_dict(orient="records")[0],
            "RandomForest": str(rf_pred),
            "XGBoost": str(xgb_pred),
        }

    except Exception as e:
        print("❌ Prediction Error:", e)
        return {"error": str(e)}

# =====================================================
# ✅ Print model feature order for debugging
# =====================================================
print("RandomForest expected feature order:")
print(rfModel.feature_names_in_)

print("\nXGBoost expected feature order:")
print(xgModel.feature_names_in_)
