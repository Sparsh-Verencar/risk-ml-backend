from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# ===============================
# ✅ Load model + encoders
# ===============================
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_info = pickle.load(f)

numeric_cols     = feature_info["numeric_cols"]
categorical_cols = feature_info["categorical_cols"]

# Keep only necessary columns
required_columns = numeric_cols + categorical_cols

# ===============================
# ✅ FastAPI app
# ===============================
app = FastAPI()

# =======================================
# ✅ Request schema
# =======================================
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

    # ===============================
    # Encode categorical
    # ===============================
    X_cat = encoder.transform(df[categorical_cols])

    X_cat_df = pd.DataFrame(
        X_cat,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Combine numeric + encoded cat
    X_final = pd.concat([df[numeric_cols], X_cat_df], axis=1)

    # Predict
    pred = model.predict(X_final)[0]

    return {
        "received_body": request_dict,
        "prediction": int(pred)
    }
