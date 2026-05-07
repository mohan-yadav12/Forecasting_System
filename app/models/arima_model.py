from statsmodels.tsa.statespace.sarimax import SARIMAX
from app.utils.metrics import evaluate


def train_sarima(df, validation_weeks=8):
    series = df["Total"]

    train = series.iloc[:-validation_weeks]
    valid = series.iloc[-validation_weeks:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    fitted_model = model.fit(disp=False)
    predictions = fitted_model.forecast(steps=validation_weeks)

    score = evaluate(valid.values, predictions.values)
    return fitted_model, score
