from lightgbm import LGBMRegressor
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


def train_lightgbm(df, validation_weeks=8):
    train = df.iloc[:-validation_weeks]
    valid = df.iloc[-validation_weeks:]

    X_train = train[FEATURES]
    y_train = train["Total"]

    X_valid = valid[FEATURES]
    y_valid = valid["Total"]

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_samples=5,
        num_leaves=20,
        force_col_wise=True,
        verbose=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    score = evaluate(y_valid, predictions)

    return model, score