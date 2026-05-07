from prophet import Prophet
from app.utils.metrics import evaluate


def train_prophet(df, validation_weeks=8):
    prophet_df = df[["Date", "Total"]].copy()
    prophet_df.columns = ["ds", "y"]

    train = prophet_df.iloc[:-validation_weeks]
    valid = prophet_df.iloc[-validation_weeks:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(train)

    future = model.make_future_dataframe(
        periods=validation_weeks,
        freq="W"
    )

    forecast = model.predict(future)
    predictions = forecast.tail(validation_weeks)["yhat"].values

    score = evaluate(valid["y"].values, predictions)
    return model, score
