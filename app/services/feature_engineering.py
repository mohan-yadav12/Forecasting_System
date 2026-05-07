import numpy as np


def create_features(df):
    df = df.copy()

    df["lag_1"] = df["Total"].shift(1)
    df["lag_2"] = df["Total"].shift(2)
    df["lag_4"] = df["Total"].shift(4)
    df["lag_8"] = df["Total"].shift(8)

    df["rolling_mean_4"] = df["Total"].shift(1).rolling(4).mean()
    df["rolling_std_4"] = df["Total"].shift(1).rolling(4).std()

    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year
    df["time_index"] = np.arange(len(df))

    df = df.dropna().reset_index(drop=True)
    return df
