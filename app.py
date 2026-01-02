import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime, time

# 1. Configuration
st.set_page_config(
    page_title="Ontario Energy Brain", 
    layout="centered", 
    page_icon="⚡"
)

# Defaults to localhost if not found
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# --- REAL WORLD DATA EXAMPLES ---
EXAMPLES = {
    "Custom (Manual Input)": None,
    "Scenario 1: Evening Peak (High Load)": {
        "date": "2023-04-03",
        "hour": "19:00", 
        "lag_24": 16440.0, 
        "lag_1y": 16321.0,
        "actual_mw": 16590.0  # The Ground Truth
    },
    "Scenario 2: Night Drop (Transition)": {
        "date": "2023-04-03",
        "hour": "21:00", 
        "lag_24": 15187.0, 
        "lag_1y": 15066.0,
        "actual_mw": 15434.0
    },
    "Scenario 3: Late Night (Low Load)": {
        "date": "2023-04-03",
        "hour": "23:00", 
        "lag_24": 13753.0, 
        "lag_1y": 13779.0,
        "actual_mw": 13755.0
    }
}

# --- HEADER ---
st.title("⚡ Ontario Energy Demand")

with st.expander("ℹ️ How this model works (Model Architecture)"):
    st.markdown("""
    This system uses two models working in parallel:
    1.  **Forecaster (XGBoost):** Predicts the exact MW demand using 9 temporal and lag features.
    2.  **Watchdog (Isolation Forest):** Detects if the current combination of time and demand is "Anomalous" (Unsupervised Learning).
    """)

st.divider()

# --- SCENARIO SELECTOR ---
st.subheader("1. Configure Scenario")

def update_inputs():
    selected = st.session_state.scenario_selector
    if selected != "Custom (Manual Input)":
        data = EXAMPLES[selected]
        st.session_state.date_val = pd.to_datetime(data["date"]).date()
        st.session_state.hour_val = data["hour"]
        st.session_state.l24_val = data["lag_24"]
        st.session_state.l1y_val = data["lag_1y"]

if 'date_val' not in st.session_state:
    st.session_state.date_val = pd.to_datetime("today").date()
    st.session_state.hour_val = "12:00"
    st.session_state.l24_val = 15000.0
    st.session_state.l1y_val = 14500.0

scenario = st.selectbox(
    "📂 Load a Real-World Test Case (Optional)",
    options=list(EXAMPLES.keys()),
    key="scenario_selector",
    on_change=update_inputs
)

# --- INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📅 Time Target")
    date_input = st.date_input("Select Date", key="date_val")
    hour_options = [f"{i:02d}:00" for i in range(24)]
    
    try:
        h_index = hour_options.index(st.session_state.hour_val)
    except:
        h_index = 12
        
    time_input = st.selectbox("Select Hour", hour_options, index=h_index, key="hour_val")
    timestamp = f"{date_input} {time_input}:00"

with col2:
    st.markdown("#### 🔌 Grid Context")
    lag_24hr = st.number_input("Demand 24hr ago (MW)", step=100.0, key="l24_val")
    lag_1y = st.number_input("Demand 1 year ago (MW)", step=100.0, key="l1y_val")

# --- ACTION ---
st.markdown("###")
predict_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

# --- RESULTS ---
tab1, tab2 = st.tabs(["📈 Forecast Result", "🛠️ Model Internals"])

with tab1:
    if predict_btn:
        with st.spinner("Crunching numbers..."):
            try:
                # 1. Get Prediction from API
                payload = {
                    "timestamp": str(timestamp),
                    "demand_lag_24hr": lag_24hr, 
                    "demand_lag_1year": lag_1y
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                
                pred = data["predicted_demand_MW"]
                grid_status = data.get("grid_status", "UNKNOWN")
                
                # 2. Check for Ground Truth
                actual_val = None
                if scenario != "Custom (Manual Input)":
                    actual_val = EXAMPLES[scenario]["actual_mw"]

                # 3. Display Results
                st.success("Prediction Complete")
                
                # A. Metrics Row
                if actual_val:
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Demand", f"{pred:,.0f} MW")
                    with m2:
                        st.metric("Actual Demand", f"{actual_val:,.0f} MW", delta_color="off")
                    with m3:
                        error = pred - actual_val
                        st.metric("Prediction Error", f"{error:,.0f} MW", delta_color="inverse")
                else:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Predicted Demand", f"{pred:,.0f} MW")
                    with m2:
                        delta = pred - lag_24hr
                        st.metric("vs 24hr Ago", f"{delta:,.0f} MW")
                
                # B. Anomaly Status Row (NEW)
                st.markdown("---")
                if grid_status == "NORMAL":
                    st.success("✅ **Grid Status: STABLE** (Pattern matches historical norms)")
                elif grid_status == "CRITICAL":
                    st.error("⚠️ **Grid Status: ABNORMAL** (Unusual demand pattern detected!)")
                else:
                    st.warning("⚠️ **Grid Status: UNKNOWN** (Watchdog model not loaded)")

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("MLOps Monitor")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature Vector")
        st.write("Model features:")
        features = ["hour", "day_of_week", "quarter", "month", "year", 
             "day_of_year", "is_weekend", "demand_lag_24hr", "demand_lag_1year"]
        st.dataframe(pd.DataFrame(features, columns=["Feature Name"]), hide_index=True, use_container_width=True)
        
    with c2:
        st.subheader("Performance Metrics")
        st.metric("Test Set MAPE", "3.95%", delta="-1.2% vs Baseline", delta_color="inverse")
        st.metric("Anomaly Detection", "Isolation Forest", "Unsupervised")