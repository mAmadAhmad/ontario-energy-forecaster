from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
import os

# 1. Initialize the App
app = FastAPI(title="Ontario Energy Demand Forecaster")


class ForecastInput(BaseModel):
    timestamp: str       # e.g., "2025-11-24 15:00:00"
    demand_lag_24hr: float  # The demand exactly 24 hours ago
    demand_lag_1year: float # The demand exactly 1 year ago

model = xgb.XGBRegressor()
feature_order = []

@app.on_event("startup")
def load_artifacts():
    global model, feature_order
    
    if not os.path.exists("energy_model.json"):
        raise RuntimeError("Model file not found!")
    
    model.load_model("energy_model.json")
    
    with open("model_features.json", "r") as f:
        feature_order = json.load(f)
    
    print("Model and features loaded successfully.")

# 4. The Prediction Endpoint
@app.post("/predict")
def predict_demand(data: ForecastInput):
    try:
        dt = pd.to_datetime(data.timestamp)
        
        # Recreate the exact features the model was trained on.
        input_data = {
            'hour': dt.hour,
            'day_of_week': dt.dayofweek,
            'quarter': dt.quarter,
            'month': dt.month,
            'year': dt.year,
            'day_of_year': dt.dayofyear,
            'is_weekend': int(dt.dayofweek >= 5),
            'demand_lag_24hr': data.demand_lag_24hr,
            'demand_lag_1year': data.demand_lag_1year
        }
        
        # Convert to DataFrame and ensure columns match the training order EXACTLY
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order]
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return {
            "timestamp": data.timestamp,
            "predicted_demand_MW": float(prediction),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Root endpoint (Health Check)
@app.get("/")
def health_check():
    return {"status": "running", "message": "Energy Forecaster is Online"}