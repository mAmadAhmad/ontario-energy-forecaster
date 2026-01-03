# ⚡ Ontario Energy Demand Forecaster

A production-grade Machine Learning microservice that forecasts hourly energy demand for the Ontario power grid. This system combines **XGBoost** for accurate regression with **Isolation Forests** for unsupervised anomaly detection to ensure grid reliability.

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ontario-energy-forecaster.streamlit.app/)

## 🚀 Key Features

* **High-Accuracy Forecasting:** Predicts grid load (MW) with **<4% MAPE** (Mean Absolute Percentage Error).
* **Grid Watchdog (Anomaly Detection):** Uses unsupervised learning to flag abnormal grid conditions (e.g., extreme weather events, outages) in real-time.
* **Scenario Simulation:** Frontend interface allows users to load historical "Ground Truth" scenarios to unit-test the model live.
* **Microservices Architecture:** Decoupled backend (FastAPI) and frontend (Streamlit) for independent scaling.

## 🛠️ Tech Stack

* **Modeling:** XGBoost, Scikit-Learn (Isolation Forest), Pandas
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Plotly
* **DevOps:** Docker, Render (Cloud Deployment), Git

## 🧠 Model Architecture

The system utilizes a dual-model approach:

1.  **The Forecaster (XGBoost Regressor):**
    * **Objective:** Minimize Mean Absolute Error (MAE).
    * **Features:** Temporal signals (Hour, Day of Week, Seasonality) + Inertia signals (24h Lag, 1-Year Lag).
    * **Performance:** Achieved **3.95% MAPE** on the 2021-2023 Test Set.

2.  **The Watchdog (Isolation Forest):**
    * **Objective:** Detect Contextual Anomalies.
    * **Logic:** Flags data points that are statistically rare based on the combination of Time and Lag Load.
    * **Status:** Returns `NORMAL` or `CRITICAL` flags with every prediction.

## 📦 Installation & Local Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/mAmadAhmad/ontario-energy-forecaster.git](https://github.com/mAmadAhmad/ontario-energy-forecaster.git)
    cd ontario-energy-forecaster
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Backend (Terminal 1)**
    ```bash
    uvicorn main:app --reload
    ```

5.  **Run the Frontend (Terminal 2)**
    ```bash
    streamlit run app.py
    ```

## 📊 Performance Metrics

| Metric | Baseline (Persistence) | Our Model (XGBoost) | Improvement |
| :--- | :--- | :--- | :--- |
| **MAPE** | 5.15% | **3.95%** | **+1.2%** |
| **MAE** | 832 MW | **~600 MW** | **~230 MW Saved** |

## 📁 Project Structure

```bash
├── app.py                 # Streamlit Frontend (The User Interface)
├── main.py                # FastAPI Backend (The Inference Engine)
├── energy_model.json      # Trained XGBoost Artifact
├── anomaly_model.pkl      # Trained Isolation Forest Artifact
├── model_features.json    # Feature consistency map
├── requirements.txt       # Project dependencies
└── Procfile               # Cloud deployment configuration