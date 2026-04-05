# Ontario Energy Demand Forecaster

A machine learning microservice that forecasts hourly energy demand for the Ontario power grid, combining XGBoost regression with Isolation Forest anomaly detection.

It is hard or impossible for me to get real time data from Ontario Energy Provider company IESO than I could have automated model retraining and provide real time predictions for current hour based on real data of previous day, however they do provide real time forecasts themselves along with prices here [![IESO Market Data]](https://www.ieso.ca/market-data),

Check my live application here->[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ontario-energy-forecaster.streamlit.app/)  
The application might take some time to load frontend(streamlit hosted) and backend(Render hosted) as they are hosted on free tiers for demo purposes. Click on "Yes, get this app backup" thanks for your patience and time.  

**Scenario Simulation:** Since live data isn't available, the frontend lets you load real historical scenarios from the test set to unit-test the model and see how much it drifts from actual demand. If you have accurate new data, the app accepts it. 
**Microservices Architecture:** Decoupled backend (FastAPI) and frontend (Streamlit) for independent scaling.

## Tech Stack

XGBoost, Scikit-Learn, FastAPI, Streamlit, Plotly, Docker, Render (Cloud Deployment), CI/CD (Render and Streamlit auto detects new commits and deploy latest version of application)

## Model Architecture

The system uses two models:

1. Forecaster (XGBoost): Uses temporal features (hour, day of week, seasonality) plus lag features (24h and 1-year) to predict demand. Achieved 3.95% MAPE on the 2021-2023 test set.

2. Watchdog (Isolation Forest): Flags contextual anomalies based on the combination of time-of-day and lag load, not just raw value outliers. Returns NORMAL or CRITICAL with every prediction.


## Performance Metrics
Baseline was basically predicting that if previous day for hour 4 we had i.e. 14000MW demand than today for hour 4 we also have 14000MW demand, this dumb model had 5.15% MAPE. Here is the comparison between dumb and XGBoost model.  
| Metric | Baseline (Dumb model) | Our Model (XGBoost) | Improvement |
| :--- | :--- | :--- | :--- |
| **MAPE** | 5.15% | **3.95%** | **+1.2%** |
| **MAE** | 832 MW | **~600 MW** | **~230 MW Saved** |

## Project Structure

```bash
├── app.py                 # Streamlit Frontend (The User Interface)
├── main.py                # FastAPI Backend (The Inference Engine)
├── energy_model.json      # Trained XGBoost Artifact
├── anomaly_model.pkl      # Trained Isolation Forest Artifact
├── model_features.json    # Feature consistency map
├── requirements.txt       # Project dependencies
└── Procfile               # Cloud deployment configuration
