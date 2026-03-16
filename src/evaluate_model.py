from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import train_test_split_data
from src.feature_engineering import engineer_features


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "transactions.csv"
    model_path = PROJECT_ROOT / "models" / "fraud_model.pkl"
    outputs_dir = PROJECT_ROOT / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df_fe = engineer_features(df)
    split = train_test_split_data(df_fe, target_col="fraud", random_state=42)

    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(split.X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(split.X_test)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(split.y_test, y_pred)
    prec = precision_score(split.y_test, y_pred, zero_division=0)
    rec = recall_score(split.y_test, y_pred, zero_division=0)
    f1 = f1_score(split.y_test, y_pred, zero_division=0)

    print("Evaluation Metrics:")
    print(f"- Accuracy : {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall   : {rec:.4f}")
    print(f"- F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(split.y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(split.y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = outputs_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    if y_proba is not None:
        roc_auc = roc_auc_score(split.y_test, y_proba)
        fpr, tpr, _ = roc_curve(split.y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_path = outputs_dir / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=160)
        plt.close()
        print(f"Saved ROC curve to: {roc_path}")


if __name__ == "__main__":
    main()

