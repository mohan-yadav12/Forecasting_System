import os
import joblib

from app.services.feature_engineering import create_features
from app.models.xgboost_model import train_xgboost
from app.models.prophet_model import train_prophet
from app.models.arima_model import train_sarima
from app.models.lstm_model import train_lstm
from app.services.model_selection import select_best_model
from app.models.lightgbm_model import train_lightgbm
from app.models.catboost_model import train_catboost
from app.models.holtwinters_model import train_holtwinters


SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def train_for_state(df, state_name):
    state_df = df[df["State"] == state_name].copy()
    state_df = create_features(state_df)

    results = {}
    xgb_model, results["XGBoost"] = train_xgboost(state_df)
    prophet_model, results["Prophet"] = train_prophet(state_df)
    sarima_model, results["SARIMA"] = train_sarima(state_df)
    _, results["LSTM"] = train_lstm(state_df)
    lightgbm_model, results["LightGBM"] = train_lightgbm(state_df)
    catboost_model, results["CatBoost"] = train_catboost(state_df)
    holt_model, results["HoltWinters"] = train_holtwinters(state_df)

    joblib.dump(xgb_model, f"{SAVE_DIR}/{state_name}_xgboost.pkl")
    joblib.dump(prophet_model, f"{SAVE_DIR}/{state_name}_prophet.pkl")
    joblib.dump(sarima_model, f"{SAVE_DIR}/{state_name}_sarima.pkl")
    joblib.dump(
        lightgbm_model,
        f"{SAVE_DIR}/{state_name}_lightgbm.pkl"
    )

    joblib.dump(
        catboost_model,
        f"{SAVE_DIR}/{state_name}_catboost.pkl"
    )

    joblib.dump(
        holt_model,
        f"{SAVE_DIR}/{state_name}_holtwinters.pkl"
    )

    best_model = select_best_model(results)

    return {
        "state": state_name,
        "best_model": best_model,
        "metrics": results
    }

