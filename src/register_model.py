"""
Champion model selection, registration, and alias management using MLflow Model Registry.
Searches completed experiment runs, selects champion based on primary/secondary metrics,
registers the model into MLflow Model Registry, and assigns the 'champion' alias.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
    CHAMPION_ALIAS,
    STAGING_ALIAS,
)


def select_champion_and_register() -> Dict[str, Any]:
    """
    Queries MLflow experiment runs, evaluates metrics to select the champion model,
    registers it into the MLflow Model Registry, and sets the 'champion' alias.

    Returns:
        champion_info (dict): Metadata and evaluation metrics of the registered champion model
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Retrieve experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' was not found in tracking store.")

    # Search completed runs in experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{PRIMARY_METRIC} DESC", f"metrics.{SECONDARY_METRIC} DESC"],
    )

    if not runs:
        raise RuntimeError(f"No completed runs found for experiment '{EXPERIMENT_NAME}'.")

    # Select the champion run based on actual recorded metrics
    champion_run = runs[0]
    run_id = champion_run.info.run_id
    metrics = champion_run.data.metrics
    tags = champion_run.data.tags
    params = champion_run.data.params

    model_type = tags.get("model_type", params.get("model_family", "Unknown Classifier"))
    run_name = tags.get("mlflow.runName", run_id)
    f1 = metrics.get(PRIMARY_METRIC, 0.0)
    roc_auc = metrics.get(SECONDARY_METRIC, 0.0)
    accuracy = metrics.get("accuracy", 0.0)

    print("========================================")
    print("CHAMPION MODEL SELECTION")
    print("========================================")
    print(f"Champion Model : {run_name}")
    print(f"Model Type     : {model_type}")
    print(f"Run ID         : {run_id}")
    print(f"F1 Score       : {f1:.4f} (Primary Metric)")
    print(f"ROC-AUC        : {roc_auc:.4f} (Secondary Metric)")
    print(f"Accuracy       : {accuracy:.4f}")
    print("========================================")

    # Model URI from the selected run
    model_uri = f"runs:/{run_id}/model"

    # Register the model into MLflow Model Registry
    print(f"\nRegistering model '{REGISTERED_MODEL_NAME}' from URI: {model_uri}...")
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        tags={
            "dataset": "breast_cancer",
            "task": "binary_classification",
            "selection_metric": PRIMARY_METRIC,
            "model_type": model_type,
            "run_name": run_name,
            "run_id": run_id,
        }
    )

    version_number = model_version.version
    print(f"Registered Model Version: {version_number}")

    # Assign Aliases
    # 1. Assign required champion alias
    alias_assigned = False
    try:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=version_number
        )
        print(f"[SUCCESS] Assigned alias '@{CHAMPION_ALIAS}' to version {version_number}")
        alias_assigned = True
    except Exception as e:
        print(f"[WARNING] Could not set alias '{CHAMPION_ALIAS}': {e}")

    # 2. Assign optional staging alias if supported
    try:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=STAGING_ALIAS,
            version=version_number
        )
        print(f"[SUCCESS] Assigned optional alias '@{STAGING_ALIAS}' to version {version_number}")
    except Exception as e:
        print(f"[INFO] Optional staging alias not set: {e}")

    champion_summary = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": version_number,
        "run_id": run_id,
        "run_name": run_name,
        "model_type": model_type,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "model_uri": model_uri,
        "champion_alias": CHAMPION_ALIAS if alias_assigned else None,
    }

    return champion_summary


if __name__ == "__main__":
    select_champion_and_register()
