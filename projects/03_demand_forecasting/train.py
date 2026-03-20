from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from portfolio_ml.common import (
    build_preprocessor,
    evaluate_regression_model,
    save_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "demand" / "daily_demand.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET = "demand"


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_category", "date"])

    grouped = df.groupby(["store_id", "product_category"])["demand"]
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = grouped.shift(1).rolling(7).mean()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    return df.dropna().drop(columns=["date"])


def main() -> None:
    dataframe = build_features(pd.read_csv(DATA_PATH))
    preprocessor, _, _ = build_preprocessor(dataframe, TARGET)

    x = dataframe.drop(columns=[TARGET])
    y = dataframe[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        ),
    }

    metrics = {}
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[model_name] = evaluate_regression_model(y_test, predictions)
        trained_pipelines[model_name] = pipeline

    best_model_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    save_artifacts(
        OUTPUT_DIR,
        trained_pipelines[best_model_name],
        metrics,
        {
            "project": "demand_forecasting",
            "target_column": TARGET,
            "best_model": best_model_name,
        },
    )

    print("Best model:", best_model_name)
    print("Metrics:", metrics[best_model_name])


if __name__ == "__main__":
    main()
