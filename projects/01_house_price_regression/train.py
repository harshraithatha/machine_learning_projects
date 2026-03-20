from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from portfolio_ml.common import (
    build_preprocessor,
    evaluate_regression_model,
    regression_models,
    save_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "housing" / "housing_prices.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET = "price"


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    preprocessor, _, _ = build_preprocessor(dataframe, TARGET)

    x = dataframe.drop(columns=[TARGET])
    y = dataframe[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    metrics = {}
    trained_pipelines = {}

    for model_name, model in regression_models().items():
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
            "project": "house_price_regression",
            "target_column": TARGET,
            "best_model": best_model_name,
        },
    )

    print("Best model:", best_model_name)
    print("Metrics:", metrics[best_model_name])


if __name__ == "__main__":
    main()
