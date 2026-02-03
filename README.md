# ⚡ Ontario Energy Demand Forecaster

A production-grade Machine Learning microservice that forecasts hourly energy demand for the Ontario power grid. This system combines **XGBoost** for accurate regression with **Isolation Forests** for unsupervised anomaly detection to ensure grid reliability. It is hard or impossible for me to get real time data from Ontario Energy Provider company IESO than I could have automated model retraining and provide real time predictions for current hour based on real data of previous day, however they do provide real time forecasts themselves along with prices here [![IESO Market Data]](https://www.ieso.ca/market-data),   

Check my live application here->[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ontario-energy-forecaster.streamlit.app/)  
The application might take some time to load frontend(streamlit hosted) and backend(Render hosted) as they are hosted on free tiers for demo purposes, servers turn down with no use for free tiers. Click on "Yes, get this app backup" thanks for your patience and time.  

* **Scenario Simulation:** Frontend interface allows users to load historical "Ground Truth" scenarios to unit-test the model live, this happens on real historic data and you can also see how much the model drifts away from actual demands for that time of the day in MW. I picked few examples from the test set to showcase real scenario, rather than putting data randomly, If you have accurate new data you are most welcome. 
* **Microservices Architecture:** Decoupled backend (FastAPI) and frontend (Streamlit) for independent scaling.

## 🛠️ Tech Stack

* **Modeling:** XGBoost, Scikit-Learn (Isolation Forest), Pandas
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Plotly
* **DevOps:** Render (Cloud Deployment), Git, CI/CD automated, Render and Streamlit auto detects new commits and deploy latest version of application.

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
Baseline was basically predicting that if previous day for hour 4 we had i.e. 14000MW demand than today for hour 4 we also have 14000MW demand, this dumb model had 5.15% MAPE. Here are my metrics.  
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
