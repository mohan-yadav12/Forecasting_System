import pandas as pd
from app.services.preprocessing import load_data
from app.services.forecasting_service import train_for_state


def main():
    df = load_data("data/sales.csv")

    all_results = []

    for state in df["State"].unique():
        print(f"Training for: {state}")

        result = train_for_state(df, state)

        all_results.append({
            "state": result["state"],
            "best_model": result["best_model"]
        })

        print(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_comparison.csv", index=False)

    print("\nSaved model_comparison.csv successfully")


if __name__ == "__main__":
    main()