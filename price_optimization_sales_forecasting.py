"""Product price optimisation and sales forecasting.

Designed for ``price_demand_sales.csv`` with these columns:
``Months``, ``Product_Code`` (optional), ``Price``, and ``Demand``.

Outputs are written to an ``outputs`` directory beside this script.
The neural-network comparison runs when TensorFlow is installed; otherwise,
the rest of the analysis completes and reports that deep learning was skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
CURRENCY = "Rs"
FORECAST_MONTHS = 6
TEST_FRACTION = 0.20

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

np.random.seed(RANDOM_SEED)


def resolve_input_file() -> Path:
    """Pair the CSV with this script, including browser suffixes such as ``(1)``.

    For example, ``price_optimization_sales_forecasting(1).py`` is paired with
    ``price_demand_sales(1).csv``. The canonical CSV name remains the fallback.
    """
    script_stem = Path(__file__).stem
    expected_stem = "price_optimization_sales_forecasting"
    suffix = script_stem[len(expected_stem):] if script_stem.startswith(expected_stem) else ""
    paired_file = BASE_DIR / f"price_demand_sales{suffix}.csv"
    canonical_file = BASE_DIR / "price_demand_sales.csv"

    if paired_file.exists():
        return paired_file
    if canonical_file.exists():
        return canonical_file
    return paired_file


INPUT_FILE = resolve_input_file()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return common regression metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
    }


def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Load, clean, validate, and chronologically sort the input data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Place price_demand_sales.csv in the same directory as this script."
        )

    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    month_column = "Months" if "Months" in data.columns else "Month"
    required = {month_column, "Price", "Demand"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # The supplied NIELIT data uses DD-MM-YYYY. A second parse supports ISO dates.
    parsed_month = pd.to_datetime(data[month_column], format="%d-%m-%Y", errors="coerce")
    unresolved = parsed_month.isna()
    if unresolved.any():
        parsed_month.loc[unresolved] = pd.to_datetime(
            data.loc[unresolved, month_column], errors="coerce"
        )

    cleaned = pd.DataFrame(
        {
            "Month": parsed_month,
            "Product_Code": (
                data["Product_Code"].astype(str).str.strip()
                if "Product_Code" in data.columns
                else "Unknown"
            ),
            "Price": pd.to_numeric(data["Price"], errors="coerce"),
            "Demand": pd.to_numeric(data["Demand"], errors="coerce"),
        }
    ).dropna(subset=["Month", "Price", "Demand"])

    cleaned = cleaned[(cleaned["Price"] > 0) & (cleaned["Demand"] >= 0)]
    cleaned = cleaned.drop_duplicates().sort_values("Month").reset_index(drop=True)

    if len(cleaned) < 24:
        raise ValueError(
            f"At least 24 valid monthly rows are required; found {len(cleaned)}."
        )
    if cleaned["Price"].nunique() < 3:
        raise ValueError("At least three distinct prices are required for optimisation.")

    # Do not silently delete unusual demand observations. Flag them for reporting.
    q1, q3 = cleaned["Demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    cleaned["Demand_Outlier"] = ~cleaned["Demand"].between(lower, upper)
    cleaned["Revenue"] = cleaned["Price"] * cleaned["Demand"]
    return cleaned


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add trend and cyclic calendar features."""
    featured = data.copy()
    month_number = featured["Month"].dt.month
    featured["Trend"] = np.arange(len(featured), dtype=float)
    featured["Month_Sin"] = np.sin(2 * np.pi * month_number / 12)
    featured["Month_Cos"] = np.cos(2 * np.pi * month_number / 12)
    return featured


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest observations for honest out-of-time evaluation."""
    test_size = max(12, int(np.ceil(len(data) * TEST_FRACTION)))
    if len(data) - test_size < 12:
        test_size = max(2, len(data) // 3)
    return data.iloc[:-test_size].copy(), data.iloc[-test_size:].copy()


def train_price_models(data: pd.DataFrame) -> tuple[Any, Any | None, pd.DataFrame]:
    """Train and compare machine-learning and optional neural-network models."""
    featured = add_calendar_features(data)
    train, test = chronological_split(featured)
    features = ["Price", "Trend", "Month_Sin", "Month_Cos"]

    x_train, y_train = train[features], train["Demand"]
    x_test, y_test = test[features], test["Demand"]

    ml_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=10,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    ml_model.fit(x_train, y_train)
    ml_predictions = np.maximum(ml_model.predict(x_test), 0)

    rows = [
        {
            "Model": "Gradient Boosting (ML)",
            **regression_metrics(y_test.to_numpy(), ml_predictions),
            "Status": "trained",
        }
    ]

    dl_model = None
    dl_predictions = None
    try:
        import tensorflow as tf
        from tensorflow.keras import Input, Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(RANDOM_SEED)
        dl_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "network",
                    _KerasDemandRegressor(
                        tf=tf,
                        Input=Input,
                        Sequential=Sequential,
                        Dense=Dense,
                        Adam=Adam,
                        EarlyStopping=EarlyStopping,
                    ),
                ),
            ]
        )
        dl_model.fit(x_train, y_train)
        dl_predictions = np.maximum(dl_model.predict(x_test), 0)
        rows.append(
            {
                "Model": "Neural Network (DL)",
                **regression_metrics(y_test.to_numpy(), dl_predictions),
                "Status": "trained",
            }
        )
    except (ImportError, ModuleNotFoundError):
        rows.append(
            {
                "Model": "Neural Network (DL)",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "Status": "skipped: TensorFlow is not installed",
            }
        )

    comparison = pd.DataFrame(rows)
    prediction_table = pd.DataFrame(
        {
            "Month": test["Month"].to_numpy(),
            "Actual_Demand": y_test.to_numpy(),
            "ML_Predicted_Demand": ml_predictions,
        }
    )
    if dl_predictions is not None:
        prediction_table["DL_Predicted_Demand"] = dl_predictions
    prediction_table.to_csv(OUTPUT_DIR / "price_model_test_predictions.csv", index=False)
    return ml_model, dl_model, comparison


class _KerasDemandRegressor:
    """Small sklearn-compatible Keras regressor with target scaling."""

    def __init__(self, *, tf: Any, Input: Any, Sequential: Any, Dense: Any, Adam: Any, EarlyStopping: Any):
        self.tf = tf
        self.Input = Input
        self.Sequential = Sequential
        self.Dense = Dense
        self.Adam = Adam
        self.EarlyStopping = EarlyStopping
        self.target_mean = 0.0
        self.target_std = 1.0
        self.model = None

    def fit(self, x: Any, y: Any) -> "_KerasDemandRegressor":
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        self.target_mean = float(y_array.mean())
        self.target_std = float(y_array.std()) or 1.0
        y_scaled = (y_array - self.target_mean) / self.target_std
        self.model = self.Sequential(
            [
                self.Input(shape=(x_array.shape[1],)),
                self.Dense(32, activation="relu"),
                self.Dense(16, activation="relu"),
                self.Dense(1),
            ]
        )
        self.model.compile(optimizer=self.Adam(learning_rate=0.003), loss="mse")
        self.model.fit(
            x_array,
            y_scaled,
            epochs=400,
            batch_size=min(16, len(x_array)),
            validation_split=0.20,
            callbacks=[self.EarlyStopping(patience=35, restore_best_weights=True)],
            verbose=0,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The neural network has not been fitted.")
        scaled = self.model.predict(np.asarray(x, dtype=np.float32), verbose=0).ravel()
        return scaled * self.target_std + self.target_mean


def optimise_prices(
    data: pd.DataFrame, ml_model: Any, dl_model: Any | None, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate revenue over observed prices for the month after the dataset."""
    next_month = data["Month"].max() + pd.offsets.MonthEnd(1)
    candidate_prices = np.sort(data["Price"].unique()).astype(float)
    future_trend = float(len(data))
    month_number = next_month.month
    scenarios = pd.DataFrame(
        {
            "Price": candidate_prices,
            "Trend": future_trend,
            "Month_Sin": np.sin(2 * np.pi * month_number / 12),
            "Month_Cos": np.cos(2 * np.pi * month_number / 12),
        }
    )

    results = []
    models = [("Gradient Boosting (ML)", ml_model)]
    if dl_model is not None:
        models.append(("Neural Network (DL)", dl_model))

    for model_name, model in models:
        predicted_demand = np.maximum(model.predict(scenarios), 0)
        revenue = candidate_prices * predicted_demand
        best = int(np.argmax(revenue))
        results.append(
            {
                "Model": model_name,
                "Forecast_Month": next_month.strftime("%Y-%m-%d"),
                "Recommended_Price": candidate_prices[best],
                "Predicted_Demand": predicted_demand[best],
                "Predicted_Revenue": revenue[best],
                "Test_RMSE": float(
                    comparison.loc[comparison["Model"] == model_name, "RMSE"].iloc[0]
                ),
            }
        )
        plt.plot(candidate_prices, revenue, marker="o", label=model_name)

    plt.title("Predicted Revenue Across Historically Observed Prices")
    plt.xlabel(f"Price ({CURRENCY})")
    plt.ylabel(f"Predicted Revenue ({CURRENCY})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_optimisation_comparison.png", dpi=300)
    plt.close()

    result_table = pd.DataFrame(results).sort_values("Test_RMSE", na_position="last")
    result_table.to_csv(OUTPUT_DIR / "price_optimisation_results.csv", index=False)
    return result_table


def build_sales_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lag, rolling, trend, and seasonality features."""
    sales = data[["Month", "Demand"]].rename(columns={"Demand": "Sales"}).copy()
    for lag in (1, 2, 3, 6, 12):
        sales[f"Lag_{lag}"] = sales["Sales"].shift(lag)
    sales["Rolling_Mean_3"] = sales["Sales"].shift(1).rolling(3).mean()
    sales["Rolling_Mean_6"] = sales["Sales"].shift(1).rolling(6).mean()
    sales["Trend"] = np.arange(len(sales), dtype=float)
    month_number = sales["Month"].dt.month
    sales["Month_Sin"] = np.sin(2 * np.pi * month_number / 12)
    sales["Month_Cos"] = np.cos(2 * np.pi * month_number / 12)
    return sales.dropna().reset_index(drop=True)


def train_and_forecast_sales(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate against a naive baseline and recursively forecast future sales."""
    featured = build_sales_features(data)
    train, test = chronological_split(featured)
    feature_columns = [c for c in featured.columns if c not in {"Month", "Sales"}]

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(train[feature_columns], train["Sales"])
    predictions = np.maximum(model.predict(test[feature_columns]), 0)
    baseline_predictions = test["Lag_1"].to_numpy()

    metrics = pd.DataFrame(
        [
            {"Model": "Random Forest Forecast", **regression_metrics(test["Sales"], predictions)},
            {"Model": "Naive Previous-Month Baseline", **regression_metrics(test["Sales"], baseline_predictions)},
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "sales_forecast_metrics.csv", index=False)

    evaluation = pd.DataFrame(
        {
            "Month": test["Month"],
            "Actual_Sales": test["Sales"],
            "Predicted_Sales": predictions,
            "Baseline_Prediction": baseline_predictions,
        }
    )
    evaluation.to_csv(OUTPUT_DIR / "sales_forecast_test_predictions.csv", index=False)

    history = data[["Month", "Demand"]].rename(columns={"Demand": "Sales"}).copy()
    future_rows = []
    for _ in range(FORECAST_MONTHS):
        next_month = history["Month"].max() + pd.offsets.MonthEnd(1)
        values = history["Sales"].to_numpy(dtype=float)
        month_number = next_month.month
        row = {
            "Lag_1": values[-1],
            "Lag_2": values[-2],
            "Lag_3": values[-3],
            "Lag_6": values[-6],
            "Lag_12": values[-12],
            "Rolling_Mean_3": values[-3:].mean(),
            "Rolling_Mean_6": values[-6:].mean(),
            "Trend": float(len(history)),
            "Month_Sin": np.sin(2 * np.pi * month_number / 12),
            "Month_Cos": np.cos(2 * np.pi * month_number / 12),
        }
        prediction = max(float(model.predict(pd.DataFrame([row])[feature_columns])[0]), 0.0)
        future_rows.append({"Month": next_month, "Forecast_Sales": prediction})
        history.loc[len(history)] = [next_month, prediction]

    future = pd.DataFrame(future_rows)
    future.to_csv(OUTPUT_DIR / "future_sales_forecast.csv", index=False)

    plt.plot(evaluation["Month"], evaluation["Actual_Sales"], marker="o", label="Actual")
    plt.plot(evaluation["Month"], evaluation["Predicted_Sales"], marker="o", label="Model")
    plt.plot(evaluation["Month"], evaluation["Baseline_Prediction"], linestyle="--", label="Naive baseline")
    plt.title("Out-of-Time Sales Forecast Evaluation")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_forecast_evaluation.png", dpi=300)
    plt.close()

    plt.plot(data["Month"], data["Demand"], label="Historical sales")
    plt.plot(future["Month"], future["Forecast_Sales"], marker="o", label="Six-month forecast")
    plt.title("Future Sales Forecast")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_sales_forecast.png", dpi=300)
    plt.close()
    return metrics, future


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_and_validate_data(INPUT_FILE)
    data.to_csv(OUTPUT_DIR / "validated_data.csv", index=False)

    summary = {
        "rows": len(data),
        "products": int(data["Product_Code"].nunique()),
        "unique_prices": int(data["Price"].nunique()),
        "start_month": data["Month"].min().strftime("%Y-%m-%d"),
        "end_month": data["Month"].max().strftime("%Y-%m-%d"),
        "demand_outliers_flagged": int(data["Demand_Outlier"].sum()),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    ml_model, dl_model, price_metrics = train_price_models(data)
    price_metrics.to_csv(OUTPUT_DIR / "price_model_metrics.csv", index=False)
    price_results = optimise_prices(data, ml_model, dl_model, price_metrics)
    sales_metrics, future_forecast = train_and_forecast_sales(data)

    print("\nDATA SUMMARY")
    print(pd.Series(summary).to_string())
    print("\nPRICE-MODEL METRICS")
    print(price_metrics.to_string(index=False))
    print("\nPRICE-OPTIMISATION RESULTS")
    print(price_results.to_string(index=False))
    print("\nSALES-FORECAST METRICS")
    print(sales_metrics.to_string(index=False))
    print("\nFUTURE SALES FORECAST")
    print(future_forecast.to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
