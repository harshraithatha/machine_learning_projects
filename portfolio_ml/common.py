from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_preprocessor(
    dataframe: pd.DataFrame,
    target_column: str,
    drop_columns: Iterable[str] | None = None,
) -> Tuple[ColumnTransformer, list[str], list[str]]:
    drop_columns = list(drop_columns or [])
    feature_frame = dataframe.drop(columns=[target_column] + drop_columns, errors="ignore")

    numeric_columns = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor, numeric_columns, categorical_columns


def regression_models() -> Dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
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


def classification_models(scale_pos_weight: float | None = None) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    xgb_params = {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = scale_pos_weight

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "xgboost": XGBClassifier(**xgb_params),
    }


def evaluate_regression_model(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, predictions, squared=False)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_classification_model(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def save_artifacts(output_dir: Path, model: object, metrics: dict, metadata: dict) -> None:
    ensure_directory(output_dir)
    joblib.dump(model, output_dir / "best_model.joblib")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
