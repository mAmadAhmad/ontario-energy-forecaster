from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
import os
import joblib

# 1. Initialize the App
app = FastAPI(title="Ontario Energy Demand Forecaster")

class ForecastInput(BaseModel):
    timestamp: str       # e.g., "2025-11-24 15:00:00"
    demand_lag_24hr: float  # The demand exactly 24 hours ago
    demand_lag_1year: float # The demand exactly 1 year ago

# Global Variables
model = xgb.XGBRegressor()
feature_order = []
anomaly_model = None

@app.on_event("startup")
def load_artifacts():
    global model, feature_order, anomaly_model

    # Load the Anomaly Detector (Grid Watchdog)
    # Ensure 'anomaly_model.pkl' is in the same folder!
    if os.path.exists("anomaly_model.pkl"):
        anomaly_model = joblib.load('anomaly_model.pkl')
        print("Anomaly model loaded successfully.")
    else:
        print("Warning: anomaly_model.pkl not found. Grid status will be unavailable.")

    # Load the XGBoost Regressor (Demand Predictor)
    if not os.path.exists("energy_model.json"):
        raise RuntimeError("Model file not found!")
    
    model.load_model("energy_model.json")
    
    with open("model_features.json", "r") as f:
        feature_order = json.load(f)
    
    print("Main model and features loaded successfully.")

# 4. The Prediction Endpoint
@app.post("/predict")
def predict_demand(data: ForecastInput):
    try:
        dt = pd.to_datetime(data.timestamp)
        
        # A. Feature Engineering (for XGBoost)
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
        
        # B. XGBoost Prediction
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order] # Ensure correct order
        prediction = model.predict(input_df)[0]
        
        # C. Anomaly Detection (Grid Watchdog)
        grid_status = "UNKNOWN"
        if anomaly_model:
            # The Isolation Forest expects ONLY these 4 features in this order
            # (Based on how we trained it in the notebook)
            anomaly_features = ['hour', 'day_of_week', 'month', 'demand_lag_24hr']
            anomaly_input = pd.DataFrame([input_data])[anomaly_features]
            
            # Predict returns: 1 (Normal), -1 (Anomaly)
            anomaly_score = anomaly_model.predict(anomaly_input)[0]
            grid_status = "CRITICAL" if anomaly_score == -1 else "NORMAL"
        
        return {
            "timestamp": data.timestamp,
            "predicted_demand_MW": float(prediction),
            "grid_status": grid_status,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Root endpoint (Health Check)
@app.get("/")
def health_check():
    return {"status": "running", "message": "Energy Forecaster is Online"}