"""
Inference module demonstrating model loading and prediction from MLflow Model Registry.
Dynamically loads the champion model using the 'champion' alias from the MLflow registry:
models:/BreastCancerClassifier@champion
Performs batch prediction, single-sample inference, and probability estimation.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
)
from src.data import load_and_preprocess_data


def run_prediction_pipeline(sample_index: int = 0) -> Dict[str, Any]:
    """
    Loads champion model directly from MLflow Model Registry using alias
    and executes batch and single-sample inference.

    Args:
        sample_index (int): Index of test sample to use for single inference demo.

    Returns:
        prediction_results (dict): Dictionary with single and batch inference details.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Dynamic model URI targeting champion alias in MLflow Registry
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    print("========================================")
    print("Registry Prediction")
    print("========================================")
    print(f"Model : {REGISTERED_MODEL_NAME}")
    print(f"Alias : {CHAMPION_ALIAS}")
    print(f"URI   : {model_uri}")
    print("----------------------------------------")

    # Load model directly from MLflow Registry
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load registered model from '{model_uri}': {e}")

    # Load test dataset for real inference
    _, X_test, _, y_test, data_info = load_and_preprocess_data()
    target_names = data_info["target_names"]

    # 1. Batch Prediction Demonstration
    batch_size = min(5, len(X_test))
    X_batch = X_test.iloc[:batch_size]
    y_batch_true = y_test.iloc[:batch_size].values

    batch_preds = model.predict(X_batch)
    batch_probs = model.predict_proba(X_batch) if hasattr(model, "predict_proba") else None

    print(f"\n--- Batch Prediction (First {batch_size} Test Samples) ---")
    batch_records = []
    for i in range(batch_size):
        pred_class = int(batch_preds[i])
        true_class = int(y_batch_true[i])
        pred_label = target_names[pred_class]
        true_label = target_names[true_class]
        match_str = "[CORRECT]" if pred_class == true_class else "[MISMATCH]"
        prob_str = f"P(0)={batch_probs[i][0]:.4f}, P(1)={batch_probs[i][1]:.4f}" if batch_probs is not None else "N/A"
        print(f"Sample {i+1}: True={true_class} ({true_label}) | Pred={pred_class} ({pred_label}) | {prob_str} {match_str}")
        batch_records.append({
            "sample": i + 1,
            "true_class": true_class,
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": batch_probs[i].tolist() if batch_probs is not None else None,
        })

    # 2. Single-Sample Prediction Demonstration
    sample = X_test.iloc[[sample_index]]
    true_label = int(y_test.iloc[sample_index])

    single_pred = int(model.predict(sample)[0])
    single_label = target_names[single_pred]

    prob_0 = None
    prob_1 = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        prob_0 = float(probs[0])
        prob_1 = float(probs[1])

    print("\n----------------------------------------")
    print("Single-Sample Inference Result")
    print("----------------------------------------")
    print(f"Sample Index         : {sample_index}")
    print(f"Actual True Label    : {true_label} ({target_names[true_label]})")
    print(f"Prediction           : {single_pred} ({single_label})")
    if prob_0 is not None and prob_1 is not None:
        print(f"Probability class 0  : {prob_0:.6f} ({target_names[0]})")
        print(f"Probability class 1  : {prob_1:.6f} ({target_names[1]})")
        confidence = max(prob_0, prob_1) * 100
        print(f"Model Confidence     : {confidence:.2f}%")
    print("========================================\n")

    return {
        "model_uri": model_uri,
        "sample_index": sample_index,
        "true_label": true_label,
        "prediction": single_pred,
        "predicted_label": single_label,
        "prob_class_0": prob_0,
        "prob_class_1": prob_1,
        "batch_records": batch_records,
    }


if __name__ == "__main__":
    run_prediction_pipeline()
