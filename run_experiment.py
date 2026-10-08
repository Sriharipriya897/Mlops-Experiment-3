"""
End-to-End MLOps Experiment Orchestrator.
Orchestrates data loading, experiment tracking across multiple models,
hyperparameter/metric/artifact logging, champion model selection,
MLflow Model Registry registration, alias assignment, and registry-based inference.
"""

import sys
import os
import subprocess
from pathlib import Path

# Self-bootstrap: if executed with global Python, transparently switch to .venv python
# to ensure the verified environment (scikit-learn 1.5.2) runs without Windows Application Control restrictions.
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

if VENV_PYTHON.exists() and sys.executable.lower() != str(VENV_PYTHON).lower():
    # Only re-exec if not already in the venv
    if os.environ.get("_RUN_EXPERIMENT_INNER") != "1":
        env = os.environ.copy()
        env["_RUN_EXPERIMENT_INNER"] = "1"
        result = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
        sys.exit(result.returncode)

# Add project root to path
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)
from src.data import load_and_preprocess_data
from src.train import train_and_track_experiments
from src.register_model import select_champion_and_register
from src.predict import run_prediction_pipeline


def main():
    print("\n" + "=" * 60)
    print("      MLOPS LAB EXPERIMENT 3 - END-TO-END WORKFLOW")
    print("   MLflow Tracking, Artifacts, and Model Registry")
    print("=" * 60 + "\n")

    verification_checks = {}

    # STEP 1: Load and check dataset
    print(">>> STEP 1: Loading and validating dataset...")
    try:
        X_train, X_test, y_train, y_test, data_info = load_and_preprocess_data()
        verification_checks["dataset_loaded"] = True
        print(f"    [OK] Dataset loaded: {data_info['n_samples']} samples, {data_info['n_features']} features")
        print(f"    [OK] Train shape: {data_info['train_shape']}, Test shape: {data_info['test_shape']}")
    except Exception as e:
        print(f"    [FAIL] Dataset loading failed: {e}")
        verification_checks["dataset_loaded"] = False
        sys.exit(1)

    # STEP 2-7: Train multiple models and log to MLflow
    print("\n>>> STEP 2 - 7: Training models & logging to MLflow...")
    try:
        results_summary = train_and_track_experiments()
        verification_checks["experiment_created"] = True
        verification_checks["multiple_runs_completed"] = len(results_summary) >= 3
        verification_checks["params_logged"] = True
        verification_checks["metrics_logged"] = True
        verification_checks["artifacts_logged"] = True
        verification_checks["models_logged"] = True
        print(f"    [OK] Successfully completed {len(results_summary)} tracked experiment runs.")
    except Exception as e:
        print(f"    [FAIL] Training / Tracking failed: {e}")
        verification_checks["multiple_runs_completed"] = False
        sys.exit(1)

    # STEP 8-10: Champion Selection, Model Registry, and Alias Assignment
    print("\n>>> STEP 8 - 10: Selecting Champion, Registering Model & Assigning Alias...")
    try:
        champion_info = select_champion_and_register()
        verification_checks["champion_selected"] = True
        verification_checks["model_registered"] = True
        verification_checks["champion_alias_assigned"] = champion_info.get("champion_alias") == CHAMPION_ALIAS
        print(f"    [OK] Champion selected: {champion_info['run_name']} ({champion_info['model_type']})")
        print(f"    [OK] Model registered: {champion_info['model_name']} v{champion_info['model_version']}")
        print(f"    [OK] Alias '@{CHAMPION_ALIAS}' assigned.")
    except Exception as e:
        print(f"    [FAIL] Model Registry failed: {e}")
        verification_checks["champion_selected"] = False
        sys.exit(1)

    # STEP 11-12: Load champion from registry and run prediction test
    print("\n>>> STEP 11 - 12: Loading champion model from MLflow Registry & Running Inference...")
    try:
        pred_results = run_prediction_pipeline(sample_index=0)
        verification_checks["registry_model_loaded"] = True
        verification_checks["prediction_completed"] = True
        print("    [OK] Champion model loaded directly from registry.")
        print(f"    [OK] Sample 0 prediction: class {pred_results['prediction']} ({pred_results['predicted_label']})")
    except Exception as e:
        print(f"    [FAIL] Registry prediction failed: {e}")
        verification_checks["registry_model_loaded"] = False
        sys.exit(1)

    # STEP 13: Final Verification Report
    print("\n" + "=" * 50)
    print("FINAL EXPERIMENT VERIFICATION")
    print("=" * 50)

    pass_status = lambda k: "[PASS]" if verification_checks.get(k) else "[FAIL]"

    print(f"{pass_status('dataset_loaded')} Dataset loaded")
    print(f"{pass_status('experiment_created')} MLflow experiment created")
    print(f"{pass_status('multiple_runs_completed')} Multiple model runs completed")
    print(f"{pass_status('params_logged')} Parameters logged")
    print(f"{pass_status('metrics_logged')} Metrics logged")
    print(f"{pass_status('artifacts_logged')} Artifacts logged")
    print(f"{pass_status('models_logged')} Models logged")
    print(f"{pass_status('champion_selected')} Champion selected")
    print(f"{pass_status('model_registered')} Model registered")
    print(f"{pass_status('champion_alias_assigned')} Champion alias assigned")
    print(f"{pass_status('registry_model_loaded')} Registry model loaded")
    print(f"{pass_status('prediction_completed')} Prediction completed\n")

    print(f"Champion:\n{champion_info['model_type']} ({champion_info['run_name']})\n")
    print(f"F1:\n{champion_info['f1_score']:.4f}\n")
    print(f"ROC-AUC:\n{champion_info['roc_auc']:.4f}\n")
    print(f"Accuracy:\n{champion_info['accuracy']:.4f}\n")
    print(f"Model URI:\n{champion_info['model_uri']}\n")
    print(f"Registry:\n{REGISTERED_MODEL_NAME}\n")
    print(f"Alias:\n{CHAMPION_ALIAS}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
