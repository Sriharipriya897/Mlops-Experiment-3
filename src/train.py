"""
Training and experiment tracking pipeline using MLflow.
Trains multiple classification models across varied hyperparameters,
computes evaluation metrics, generates visual diagnostic artifacts,
and logs models with signatures.
"""

import sys
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    auc,
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PLOTS_DIR,
    RANDOM_STATE,
)
from src.data import load_and_preprocess_data

# Suppress minor external optimization warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def generate_and_log_artifacts(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: List[str],
    run_id: str
) -> List[str]:
    """
    Generates and logs visual diagnostic artifacts:
    1. Confusion Matrix heatmap
    2. ROC-AUC curve
    3. Feature Importance or Coefficient Magnitude plot
    """
    saved_artifacts = []

    # 1. Confusion Matrix
    cm_path = PLOTS_DIR / f"{run_id}_confusion_matrix.png"
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(cm_path), artifact_path="plots")
    saved_artifacts.append("confusion_matrix.png")

    # 2. ROC-AUC Curve
    roc_path = PLOTS_DIR / f"{run_id}_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC Curve (AUC = {roc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(roc_path), artifact_path="plots")
    saved_artifacts.append("roc_curve.png")

    # 3. Feature Importance or Coefficient Magnitude
    fi_path = PLOTS_DIR / f"{run_id}_feature_importance.png"
    importances = None
    fi_title = "Feature Importance"

    # Check if Pipeline or raw model
    estimator = model.named_steps["classifier"] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        fi_title = f"Top 10 Feature Importances - {model_name}"
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
        fi_title = f"Top 10 Coefficient Magnitudes (|coef|) - {model_name}"

    if importances is not None:
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title(fi_title)
        plt.xlabel("Relative Importance / Magnitude")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(fi_path), artifact_path="plots")
        saved_artifacts.append("feature_importance.png")

    return saved_artifacts


def train_and_track_experiments() -> List[Dict[str, Any]]:
    """
    Executes multiple experiment runs with varied classifiers and hyperparameters,
    tracking all runs, metrics, parameters, and artifacts in MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, data_info = load_and_preprocess_data()
    feature_names = data_info["feature_names"]

    # Define experiment configurations with multiple classifiers & hyperparameters
    experiments_config = [
        {
            "model_type": "Logistic Regression",
            "run_name": "LogisticRegression_L2_C1.0",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE))
            ]),
            "params": {"C": 1.0, "max_iter": 500, "penalty": "l2", "solver": "lbfgs", "scaled": True}
        },
        {
            "model_type": "Logistic Regression",
            "run_name": "LogisticRegression_L2_C0.1",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_STATE))
            ]),
            "params": {"C": 0.1, "max_iter": 500, "penalty": "l2", "solver": "lbfgs", "scaled": True}
        },
        {
            "model_type": "Random Forest",
            "run_name": "RandomForest_n100_d5",
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=2, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "criterion": "gini"}
        },
        {
            "model_type": "Random Forest",
            "run_name": "RandomForest_n200_d10",
            "estimator": RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=4, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4, "criterion": "gini"}
        },
        {
            "model_type": "Gradient Boosting",
            "run_name": "GradientBoosting_n100_lr0.1",
            "estimator": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "loss": "log_loss"}
        },
        {
            "model_type": "Gradient Boosting",
            "run_name": "GradientBoosting_n150_lr0.05",
            "estimator": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4, "loss": "log_loss"}
        },
    ]

    print("========================================")
    print("MLflow Training & Experiment Tracking")
    print("========================================")
    print(f"Experiment Name : {EXPERIMENT_NAME}")
    print(f"Tracking URI    : {MLFLOW_TRACKING_URI}")
    print(f"Total Runs      : {len(experiments_config)}")
    print("----------------------------------------")

    results_summary = []

    for cfg in experiments_config:
        model_type = cfg["model_type"]
        run_name = cfg["run_name"]
        model = cfg["estimator"]
        params = cfg["params"]

        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
            run_id = run.info.run_id

            # Set descriptive tags
            mlflow.set_tags({
                "model_type": model_type,
                "dataset": "breast_cancer",
                "experiment_purpose": "MLOps Experiment 3 - Model Tracking & Evaluation",
                "task": "binary_classification",
            })

            # Log Hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("model_family", model_type)

            # Train Model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate True Metrics
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, y_prob))
            loss = float(log_loss(y_test, y_prob))

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "log_loss": round(loss, 4),
            }

            # Log Metrics to MLflow
            mlflow.log_metrics(metrics)

            # Generate and log visual diagnostic artifacts
            artifacts = generate_and_log_artifacts(
                model=model,
                model_name=run_name,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                feature_names=feature_names,
                run_id=run_id
            )

            # Infer Signature and Log Model with cloudpickle serialization format
            signature = infer_signature(X_test, y_pred)
            input_example = X_test.iloc[:2]

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle"
            )

            print(f"Model     : {model_type} ({run_name})")
            print(f"Run ID    : {run_id}")
            print(f"Accuracy  : {accuracy:.4f}")
            print(f"F1 Score  : {f1:.4f}")
            print(f"ROC-AUC   : {roc_auc:.4f}")
            print(f"Log Loss  : {loss:.4f}")
            print(f"Artifacts : {', '.join(artifacts)}")
            print("----------------------------------------")

            results_summary.append({
                "run_id": run_id,
                "model_type": model_type,
                "run_name": run_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "log_loss": loss,
            })

    # Print Comparison Table
    print("\n==================================================")
    print("MODEL EVALUATION COMPARISON TABLE")
    print("==================================================")
    df_results = pd.DataFrame(results_summary)
    df_display = df_results[["run_name", "model_type", "accuracy", "f1_score", "roc_auc", "log_loss"]]
    print(df_display.to_string(index=False))
    print("==================================================\n")

    return results_summary


if __name__ == "__main__":
    train_and_track_experiments()
