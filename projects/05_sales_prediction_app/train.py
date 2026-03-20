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
DATA_PATH = ROOT / "data" / "sales" / "sales_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET = "sales"


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)
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
            "project": "sales_prediction_app",
            "target_column": TARGET,
            "best_model": best_model_name,
        },
    )

    print("Best model:", best_model_name)
    print("Metrics:", metrics[best_model_name])


if __name__ == "__main__":
    main()
