from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import build_preprocessor, get_feature_lists, train_test_split_data
from src.feature_engineering import engineer_features


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    numeric_features, categorical_features = get_feature_lists()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    return {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for name, model in models.items()
    }


def evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def select_best(results: Dict[str, ModelResult]) -> ModelResult:
    # Primary: F1, Secondary: ROC-AUC
    return sorted(
        results.values(),
        key=lambda r: (r.metrics.get("f1", 0.0), r.metrics.get("roc_auc", 0.0)),
        reverse=True,
    )[0]


def train_and_select(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[ModelResult, Dict[str, ModelResult]]:
    df_fe = engineer_features(df)
    split = train_test_split_data(df_fe, target_col="fraud", random_state=random_state)

    pipelines = build_model_pipelines(random_state=random_state)
    results: Dict[str, ModelResult] = {}

    for name, pipe in pipelines.items():
        pipe.fit(split.X_train, split.y_train)
        metrics = evaluate_pipeline(pipe, split.X_test, split.y_test)
        results[name] = ModelResult(name=name, pipeline=pipe, metrics=metrics)

    best = select_best(results)
    return best, results


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "transactions.csv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    best, all_results = train_and_select(df)

    model_path = models_dir / "fraud_model.pkl"
    joblib.dump(best.pipeline, model_path)

    print("Model comparison (higher is better):")
    for name, res in all_results.items():
        m = res.metrics
        print(
            f"- {name:18s} | acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f}"
        )

    print(f"\nSelected best model: {best.name}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
