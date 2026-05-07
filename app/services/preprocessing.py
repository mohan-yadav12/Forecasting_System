import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Total"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df = df.dropna(subset=["Date", "State", "Total"])
    df = df.sort_values(["State", "Date"])

    return df