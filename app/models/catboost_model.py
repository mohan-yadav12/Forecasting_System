from catboost import CatBoostRegressor
from app.utils.metrics import evaluate

FEATURES = [
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


def train_catboost(df, validation_weeks=8):
    train = df.iloc[:-validation_weeks]
    valid = df.iloc[-validation_weeks:]

    X_train = train[FEATURES]
    y_train = train["Total"]

    X_valid = valid[FEATURES]
    y_valid = valid["Total"]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    score = evaluate(y_valid, predictions)

    return model, score