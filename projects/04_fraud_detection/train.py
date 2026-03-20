from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from portfolio_ml.common import (
    build_preprocessor,
    classification_models,
    evaluate_classification_model,
    save_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "fraud" / "fraud_transactions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET = "fraud"


def main() -> None:
    dataframe = pd.read_csv(DATA_PATH)
    preprocessor, _, _ = build_preprocessor(dataframe, TARGET)

    x = dataframe.drop(columns=[TARGET])
    y = dataframe[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count

    metrics = {}
    trained_pipelines = {}

    for model_name, model in classification_models(scale_pos_weight=scale_pos_weight).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics[model_name] = evaluate_classification_model(
            y_test,
            predictions,
            probabilities,
        )
        trained_pipelines[model_name] = pipeline

    best_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    save_artifacts(
        OUTPUT_DIR,
        trained_pipelines[best_model_name],
        metrics,
        {
            "project": "fraud_detection",
            "target_column": TARGET,
            "best_model": best_model_name,
        },
    )

    print("Best model:", best_model_name)
    print("Metrics:", metrics[best_model_name])


if __name__ == "__main__":
    main()
