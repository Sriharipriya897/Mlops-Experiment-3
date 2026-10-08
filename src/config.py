"""
Central configuration module for MLOps Experiment 3.
Centralizes experiment parameters, paths, MLflow tracking URI, and model registry settings.
"""

import os
from pathlib import Path

# Enable filesystem tracking backend and silence hints in MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow Tracking Configuration
# Use local mlruns directory as tracking store
MLFLOW_TRACKING_DIR = BASE_DIR / "mlruns"
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_TRACKING_DIR.as_posix()}"
EXPERIMENT_NAME = "Breast Cancer Classification"

# Model Registry Configuration
REGISTERED_MODEL_NAME = "BreastCancerClassifier"
PRIMARY_METRIC = "f1_score"
SECONDARY_METRIC = "roc_auc"

# Model Aliases
CHAMPION_ALIAS = "champion"
STAGING_ALIAS = "staging"

# Dataset & Preprocessing Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFY = True
