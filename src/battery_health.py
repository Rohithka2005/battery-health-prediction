"""Battery RUL prediction and rule-based status demonstration.

Run from the project root:
    python src/battery_health.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from protection_system import classify_battery
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Battery_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = ["cycle", "chI", "chV", "chT", "disI", "disV", "disT", "BCt", "SOH"]
TARGET = "RUL"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and validate the columns required by this project."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Battery_dataset.csv in data/."
        )
    data = pd.read_csv(path)
    required = ["battery_id", *FEATURES, TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Required dataset columns contain missing values.")
    return data


def regression_metrics(y_true, y_pred) -> dict:
    """Calculate common regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def save_baseline_results(data: pd.DataFrame) -> pd.DataFrame:
    """Compare full-feature LR, cycle-only LR, RF, and tuned RF on one split."""
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression (all features)": LinearRegression(),
        "Linear Regression (cycle only)": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
    }
    predictions = {}
    fitted = {}

    models["Linear Regression (all features)"].fit(X_train, y_train)
    predictions["Linear Regression (all features)"] = models[
        "Linear Regression (all features)"
    ].predict(X_test)
    fitted["Linear Regression (all features)"] = models["Linear Regression (all features)"]

    models["Linear Regression (cycle only)"].fit(
        X_train[["cycle"]], y_train
    )
    predictions["Linear Regression (cycle only)"] = models[
        "Linear Regression (cycle only)"
    ].predict(X_test[["cycle"]])
    fitted["Linear Regression (cycle only)"] = models["Linear Regression (cycle only)"]

    models["Random Forest"].fit(X_train, y_train)
    predictions["Random Forest"] = models["Random Forest"].predict(X_test)
    fitted["Random Forest"] = models["Random Forest"]

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [50], "max_depth": [None, 10]},
        cv=3,
        scoring="r2",
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    predictions["Tuned Random Forest"] = grid.best_estimator_.predict(X_test)
    fitted["Tuned Random Forest"] = grid.best_estimator_

    rows = []
    for name, pred in predictions.items():
        rows.append({"Model": name, **regression_metrics(y_test, pred)})
    metrics = pd.DataFrame(rows).sort_values("R2", ascending=False)
    metrics.to_csv(RESULTS_DIR / "random_split_model_comparison.csv", index=False)

    # Save predictions for the best holdout R² model.
    best_name = metrics.iloc[0]["Model"]
    best_pred = predictions[best_name]
    pred_frame = pd.DataFrame(
        {"actual_RUL": y_test.to_numpy(), "predicted_RUL": best_pred}
    )
    pred_frame.to_csv(RESULTS_DIR / "holdout_predictions.csv", index=False)

    # Actual vs predicted.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, best_pred, alpha=0.8)
    lo = min(float(y_test.min()), float(np.min(best_pred)))
    hi = max(float(y_test.max()), float(np.max(best_pred)))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="Ideal prediction")
    ax.set(xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)",
           title=f"Actual vs Predicted RUL — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    # Residuals.
    residuals = y_test.to_numpy() - best_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(best_pred, residuals, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Predicted RUL (cycles)", ylabel="Residual (actual - predicted)",
           title=f"Residual Analysis — {best_name}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "residual_analysis.png", dpi=180)
    plt.close(fig)

    # Model comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics.set_index("Model")["R2"].sort_values().plot(kind="barh", ax=ax)
    ax.set(xlabel="R² score", title="Random-Split Model Comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Standardized coefficient magnitude is more comparable than raw coefficient magnitude.
    lr = LinearRegression().fit(X_train, y_train)
    std = X_train.std(ddof=0).replace(0, np.nan)
    standardized_effect = (lr.coef_ * std.to_numpy())
    coef = pd.Series(standardized_effect, index=FEATURES).abs().sort_values()
    coef.rename("absolute_standardized_linear_effect").to_csv(
        RESULTS_DIR / "linear_model_feature_analysis.csv", header=True
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    coef.plot(kind="barh", ax=ax)
    ax.set(xlabel="Absolute standardized linear effect",
           title="Linear Regression Feature Analysis")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_analysis.png", dpi=180)
    plt.close(fig)

    print("\nRandom-split model comparison")
    print(metrics.to_string(index=False, formatters={
        "MAE": "{:.3f}".format, "MSE": "{:.3f}".format, "R2": "{:.4f}".format
    }))
    print("\nBest tuned Random Forest parameters:", grid.best_params_)
    print(f"Best holdout model for diagnostic plots: {best_name}")
    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    print("Dataset loaded:", data.shape)
    print("Battery counts:", data["battery_id"].value_counts().sort_index().to_dict())
    metrics = save_baseline_results(data)

    # Demonstration input: these are user-entered example values, not a sensor connection.
    print("\n===== Battery RUL Prediction Demo =====")
    try:
        sample = {}
        prompts = {
            "cycle": "Cycle number: ",
            "chI": "Charge current (chI): ",
            "chV": "Charge voltage (chV): ",
            "chT": "Charge temperature (chT): ",
            "disI": "Discharge current (disI): ",
            "disV": "Discharge voltage (disV): ",
            "disT": "Discharge temperature (disT): ",
            "BCt": "Battery capacity (BCt): ",
            "SOH": "State of health (SOH): ",
        }
        for column, prompt in prompts.items():
            sample[column] = float(input(prompt))
            if not np.isfinite(sample[column]):
                raise ValueError(f"{column} must be finite.")
        if sample["cycle"] < 0 or sample["BCt"] < 0:
            raise ValueError("Cycle number and battery capacity cannot be negative.")
        if not 0 <= sample["SOH"] <= 100:
            raise ValueError("SOH should be between 0 and 100.")
    except (ValueError, EOFError) as exc:
        print(f"Input not accepted: {exc}")
        return

    # Use the best model according to random-split R² for the interactive demo.
    X = data[FEATURES]
    y = data[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression (all features)": (LinearRegression(), FEATURES),
        "Linear Regression (cycle only)": (LinearRegression(), ["cycle"]),
        "Random Forest": (RandomForestRegressor(random_state=42), FEATURES),
    }
    for model, feature_names in candidates.values():
        model.fit(X_train[feature_names], y_train)
    selected_name = str(metrics.iloc[0]["Model"])
    if selected_name == "Tuned Random Forest":
        selected_name = "Random Forest"
    selected_model, selected_features = candidates[selected_name]
    sample_frame = pd.DataFrame([sample], columns=FEATURES)
    predicted_rul = float(selected_model.predict(sample_frame[selected_features])[0])
    print(f"\nPredicted Remaining Useful Life: {predicted_rul:.2f} cycles")

    # Use project-defined thresholds, not safety-certified limits.
    status, action = classify_battery(max(0.0, predicted_rul))
    print("Battery Status:", status)
    print("Suggested Action:", action)
    print("\nPlots and metrics saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
