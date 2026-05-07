# End-to-End Time Series Forecasting System 📈

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

A multi-model time series forecasting pipeline designed to predict state-level sales data. This system automates feature engineering, evaluates multiple state-of-the-art models per state, and serves recursive 8-week predictions via a fast REST API.

## 🌟 Key Features

- **Automated Feature Engineering**: Dynamically generates lag features (`lag_1`, `lag_2`, `lag_4`, `lag_8`), rolling statistics (`rolling_mean_4`, `rolling_std_4`), and calendar features (month, quarter, week of year).
- **Multi-Model Evaluation**: Trains and evaluates various machine learning and statistical models (XGBoost, Prophet, SARIMA/pmdarima, LightGBM, CatBoost) to find the best performer per state.
- **Recursive Forecasting**: The API uses a recursive forecasting strategy to predict the next 8 weeks of sales based on engineered features.
- **Production API**: Deploys the best pre-trained models via a lightning-fast REST API using FastAPI.

## 🏗️ Project Architecture

```text
Forecasting/
├── app/
│   ├── api/
│   │   └── routes.py               # FastAPI endpoints (/health, /forecast)
│   ├── models/                     # Model implementation and wrappers
│   ├── services/
│   │   ├── preprocessing.py        # Data loading and cleaning
│   │   ├── feature_engineering.py  # Lag and rolling feature generation
│   │   └── forecasting_service.py  # Model training and evaluation logic
│   ├── utils/
│   └── main.py                     # FastAPI application entry point
├── data/
│   └── sales.csv                   # Raw state-level sales data
├── saved_models/                   # Directory where serialized models (e.g., .pkl) are stored
├── requirements.txt                # Project dependencies
├── train.py                        # Main script to train models and generate model_comparison.csv
└── README.md                       # You are here!
```

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Models

Run the training pipeline. This will load the data, train models for every state, save the best models into the `saved_models/` folder, and generate a `model_comparison.csv` tracking the best model per state.

```bash
python train.py
```

### 3. Running the REST API

Once the models are trained, start the FastAPI production server:

```bash
uvicorn app.main:app --reload 
```

## 🔌 API Usage

The API is accessible at `http://localhost:8000`.
Interactive API documentation (Swagger UI) is automatically available at `http://localhost:8000/docs`.

### 1. Health Check
**`GET /health`**
Returns the status of the API.
```json
{
  "status": "running"
}
```

### 2. Predict Forecast
**`POST /forecast`**
Generates an 8-week recursive forecast for the specified state. It dynamically looks up the best model evaluated for the state and serves the prediction (currently optimized to serve via XGBoost).

**Request Body (JSON):**
```json
{
  "state": "California"
}
```

**Response (JSON):**
```json
{
  "state": "California",
  "selected_best_model": "LightGBM",
  "api_serving_model": "XGBoost",
  "forecast_horizon_weeks": 8,
  "next_8_weeks_forecast": [
    827200064,
    829007488,
    830200512,
    830200512,
    830200512,
    830200512,
    830200512,
    830200512
  ]
}
```
