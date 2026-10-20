"""Battery-level generalization evaluation.

Leave one battery ID out per fold. battery_id is used only to define groups and is
never included among model input features.
Run from the project root: python src/model_evaluation.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Battery_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES = ["cycle", "chI", "chV", "chT", "disI", "disV", "disT", "BCt", "SOH"]
TARGET = "RUL"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    required = {"battery_id", *FEATURES, TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    rows = []
    predictions = []
    for battery_id in sorted(data["battery_id"].unique()):
        train = data[data["battery_id"] != battery_id]
        test = data[data["battery_id"] == battery_id]
        for name, features, model in [
            ("Linear Regression (all features)", FEATURES, LinearRegression()),
            ("Linear Regression (cycle only)", ["cycle"], LinearRegression()),
            ("Random Forest", FEATURES, RandomForestRegressor(
                n_estimators=100, random_state=42
            )),
        ]:
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            rows.append({
                "Held-out battery": battery_id,
                "Model": name,
                "MAE": mean_absolute_error(test[TARGET], pred),
                "MSE": mean_squared_error(test[TARGET], pred),
                "R2": r2_score(test[TARGET], pred),
                "Train batteries": ", ".join(sorted(train["battery_id"].unique())),
                "Test records": len(test),
            })
            for actual, predicted in zip(test[TARGET], pred):
                predictions.append({
                    "held_out_battery": battery_id,
                    "model": name,
                    "actual_RUL": actual,
                    "predicted_RUL": predicted,
                })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "leave_one_battery_out_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        RESULTS_DIR / "leave_one_battery_out_predictions.csv", index=False
    )

    print("Leave-one-battery-out results")
    print(results.to_string(index=False, formatters={
        "MAE": "{:.3f}".format, "MSE": "{:.3f}".format, "R2": "{:.4f}".format
    }))

    pivot = results.pivot(index="Held-out battery", columns="Model", values="MAE")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("MAE (cycles)")
    ax.set_title("Leave-One-Battery-Out Evaluation")
    ax.set_xlabel("Held-out battery")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS_DIR / "unseen_battery_mae.png", dpi=180)
    plt.close(ax.figure)

    print("\nInterpretation note: only three battery groups are available, so this is a")
    print("useful stress test but not a statistically broad estimate of generalization.")


if __name__ == "__main__":
    main()
