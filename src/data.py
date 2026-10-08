"""
Data loading and preprocessing module for Breast Cancer Wisconsin classification dataset.
Provides clean, reusable dataset loading and stratified splitting logic.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RANDOM_STATE, TEST_SIZE, STRATIFY


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Loads Breast Cancer dataset, converts to pandas DataFrame,
    and performs stratified train-test split.

    Returns:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        data_info (dict): Metadata including feature names, target names, and shapes
    """
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.data
    y = cancer.target
    feature_names = list(cancer.feature_names)
    target_names = list(cancer.target_names)

    stratify_target = y if STRATIFY else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target
    )

    data_info = {
        "feature_names": feature_names,
        "target_names": target_names,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "class_distribution": y.value_counts().to_dict(),
    }

    return X_train, X_test, y_train, y_test, data_info


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, info = load_and_preprocess_data()
    print("Dataset loaded successfully.")
    print(f"Features ({info['n_features']}): {info['feature_names'][:5]}...")
    print(f"Target classes: {info['target_names']}")
    print(f"Train shape: {info['train_shape']}, Test shape: {info['test_shape']}")
