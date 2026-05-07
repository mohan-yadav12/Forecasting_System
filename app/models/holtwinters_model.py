from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.utils.metrics import evaluate


def train_holtwinters(df, validation_weeks=8):
    series = df["Total"]

    train = series.iloc[:-validation_weeks]
    valid = series.iloc[-validation_weeks:]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    )

    fitted_model = model.fit()

    predictions = fitted_model.forecast(validation_weeks)

    score = evaluate(valid.values, predictions.values)

    return fitted_model, score