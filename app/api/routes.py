from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from app.services.preprocessing import load_data
from app.services.feature_engineering import create_features

router = APIRouter()


class ForecastRequest(BaseModel):
    state: str


@router.get("/health")
def health_check():
    return {
        "status": "running"
    }


@router.post("/forecast")
def forecast(req: ForecastRequest):
    state_name = req.state

    # Step 1: Load model comparison file
    comparison_df = pd.read_csv("model_comparison.csv")

    row = comparison_df[
        comparison_df["state"].str.lower() == state_name.lower()
    ]

    if row.empty:
        return {
            "error": f"State '{state_name}' not found"
        }

    best_model = row.iloc[0]["best_model"]

    # Step 2: Force XGBoost for real API prediction
    # because it is easiest and strongest for serving
    model_path = f"saved_models/{state_name}_xgboost.pkl"

    if not os.path.exists(model_path):
        return {
            "error": f"Saved model not found for {state_name}"
        }

    model = joblib.load(model_path)

    # Step 3: Load original dataset
    df = load_data("data/sales.csv")

    state_df = df[
        df["State"].str.lower() == state_name.lower()
    ].copy()

    if state_df.empty:
        return {
            "error": f"No data found for {state_name}"
        }

    # Step 4: Feature Engineering
    state_df = create_features(state_df)

    latest_row = state_df.iloc[-1:].copy()

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "rolling_mean_4",
        "rolling_std_4",
        "month",
        "quarter",
        "week_of_year",
        "year",
        "time_index"
    ]

    predictions = []

    # Step 5: Recursive Forecast for 8 Weeks
    for _ in range(8):
        X_future = latest_row[feature_columns]

        next_pred = model.predict(X_future)[0]
        predictions.append(round(float(next_pred), 2))

        # update for next step
        latest_row["lag_8"] = latest_row["lag_4"]
        latest_row["lag_4"] = latest_row["lag_2"]
        latest_row["lag_2"] = latest_row["lag_1"]
        latest_row["lag_1"] = next_pred

        latest_row["rolling_mean_4"] = (
            latest_row[
                ["lag_1", "lag_2", "lag_4", "lag_8"]
            ].mean(axis=1)
        )

        latest_row["time_index"] += 1

    return {
        "state": state_name,
        "selected_best_model": best_model,
        "api_serving_model": "XGBoost",
        "forecast_horizon_weeks": 8,
        "next_8_weeks_forecast": predictions
    }