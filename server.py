from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ===============================
# ✅ Enable CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ✅ Load model + encoders
# ===============================
with open("xgb_model.pkl", "rb") as f:
    model_data = pickle.load(f)

if isinstance(model_data, dict) and "model" in model_data:
    model = model_data["model"]
else:
    model = model_data

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_info = pickle.load(f)

numeric_cols = feature_info["numeric_cols"]
categorical_cols = feature_info["categorical_cols"]

# Keep only necessary columns
required_columns = numeric_cols + categorical_cols

# ===============================
# ✅ Request schema
# ===============================
class InputData(BaseModel):
    data: dict

# ===============================
# ✅ Prediction endpoint
# ===============================
@app.post("/predict")
def predict(payload: InputData):
    request_dict = payload.data

    # Convert to DataFrame
    df = pd.DataFrame([request_dict])

    # ✅ Filter only required columns
    df = df[required_columns]

    # ✅ Convert numeric columns to float
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # ===============================
    # Encode categorical
    # ===============================
    X_cat = encoder.transform(df[categorical_cols])

    X_cat_df = pd.DataFrame(
        X_cat,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    # Combine numeric + encoded cat
    X_final = pd.concat([df[numeric_cols], X_cat_df], axis=1)

    # ✅ Ensure correct numeric dtype
    X_final = X_final.apply(pd.to_numeric, errors="coerce")

    # Predict
    pred = model.predict(X_final)[0]

    return {
        "received_body": request_dict,
        "prediction": int(pred)
    }
