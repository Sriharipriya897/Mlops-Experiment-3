# MLOps Lab Experiment 3: MLflow Experiment Tracking & Model Registry

## Experiment Title

**Experiment Tracking, Hyperparameter Logging, Artifact Management, and Model Registry using MLflow**

---

## Aim

To implement a complete MLOps experiment tracking workflow using MLflow that demonstrates experiment tracking, hyperparameter logging, metric evaluation, artifact management, model logging with signatures, model registration, alias-based lifecycle management, and production-style inference from the MLflow Model Registry.

---

## Objectives

1. Load and preprocess the Breast Cancer Wisconsin classification dataset
2. Train multiple classification models (Logistic Regression, Random Forest, Gradient Boosting) with varied hyperparameters
3. Track all experiments, parameters, and metrics using MLflow Tracking
4. Generate and log visual diagnostic artifacts (Confusion Matrix, ROC Curve, Feature Importance)
5. Log trained models with inferred signatures using MLflow's scikit-learn flavor
6. Select the champion model based on actual F1 score and ROC-AUC
7. Register the champion model into the MLflow Model Registry
8. Assign the `champion` alias to the registered model version
9. Demonstrate production-style inference by loading the model directly from the registry using `models:/BreastCancerClassifier@champion`

---

## Requirements

### Software Requirements

| Software         | Version          |
|------------------|------------------|
| Python           | 3.13             |
| MLflow           | >= 2.15.0        |
| scikit-learn     | 1.5.2            |
| pandas           | >= 2.2.0         |
| numpy            | >= 1.26.0        |
| matplotlib       | >= 3.8.0         |
| seaborn          | >= 0.13.0        |
| Operating System | Windows 10/11    |

### Hardware Requirements

| Component | Minimum Requirement |
|-----------|---------------------|
| Processor | Intel i3 or equivalent |
| RAM       | 4 GB |
| Storage   | 500 MB free disk space |

---

## MLOps Theory

### What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently. It addresses the lifecycle management of ML models including:

- **Experiment Tracking** — Recording and comparing experiments
- **Model Versioning** — Managing different versions of trained models
- **Model Deployment** — Serving models in production environments
- **Monitoring** — Tracking model performance over time

### What is MLflow?

**MLflow** is an open-source platform for managing the end-to-end machine learning lifecycle. It provides four core components:

1. **MLflow Tracking** — Logs parameters, metrics, and artifacts from experiments
2. **MLflow Projects** — Packages ML code in a reusable, reproducible format
3. **MLflow Models** — Manages and deploys models from diverse ML libraries
4. **MLflow Model Registry** — Central hub for model versioning, staging, and deployment

### MLflow Tracking

MLflow Tracking enables you to log and query experiments using Python, REST, R, and Java APIs. Each experiment run records:

- **Parameters**: Input configuration values (e.g., `C=1.0`, `n_estimators=100`)
- **Metrics**: Output evaluation values (e.g., `accuracy=0.98`, `f1_score=0.99`)
- **Artifacts**: Output files such as plots, serialized models, or data files
- **Tags**: Key-value metadata about the run

### Parameters

Parameters are the hyperparameters and configuration values used to train a model. Examples:

```
C = 1.0          # Regularization strength for Logistic Regression
n_estimators = 100  # Number of trees in Random Forest
learning_rate = 0.1 # Step size for Gradient Boosting
max_depth = 5       # Maximum tree depth
```

### Metrics

Metrics are the quantitative evaluation measures computed on the test set:

| Metric    | Description |
|-----------|-------------|
| Accuracy  | Ratio of correct predictions to total predictions |
| Precision | Ratio of true positives to predicted positives |
| Recall    | Ratio of true positives to actual positives |
| F1 Score  | Harmonic mean of precision and recall |
| ROC-AUC   | Area under the Receiver Operating Characteristic curve |
| Log Loss  | Negative log-likelihood of the true labels |

### Artifacts

Artifacts are files generated during a run, such as:

- **Confusion Matrix** — Visual heatmap of prediction vs. actual classes
- **ROC Curve** — Plot of True Positive Rate vs. False Positive Rate
- **Feature Importance** — Bar chart of top contributing features

### Model Logging

MLflow logs trained models with:

- **Model flavor** (e.g., `mlflow.sklearn`)
- **Input/output signature** via `mlflow.models.infer_signature()`
- **Input example** for schema documentation
- **Serialization format** (cloudpickle for compatibility)

### Model Registry

The MLflow Model Registry is a centralized store for managing model versions:

- **Register** a model from a logged run
- **Version** models automatically (v1, v2, ...)
- **Tag** models with metadata
- **Alias** models for lifecycle management

### Model Aliases

Aliases are mutable, named references to specific model versions:

- `@champion` — The production-ready model
- `@staging` — A candidate model under evaluation

Load a model by alias:
```python
model = mlflow.sklearn.load_model("models:/BreastCancerClassifier@champion")
```

### Champion Model

The champion model is the best-performing model selected from all experiment runs based on a primary evaluation metric (F1 Score) with a secondary tiebreaker (ROC-AUC).

---

## Dataset Description

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Property       | Value |
|----------------|-------|
| Source          | `sklearn.datasets.load_breast_cancer()` |
| Total Samples  | 569 |
| Features       | 30 (real-valued) |
| Classes        | 2 (malignant = 0, benign = 1) |
| Class Distribution | 212 malignant, 357 benign |
| Train/Test Split | 80/20 (stratified) |
| Random State   | 42 |

Features include mean, standard error, and worst-case values for:
radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

---

## Project Structure

```
MLOPS 3 EXP/
│
├── .gitignore              # Git exclusions
├── .venv/                  # Python virtual environment (auto-created)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── run_experiment.py       # End-to-end orchestrator script
├── launch_ui.bat           # One-click launcher for MLflow UI
│
├── src/
│   ├── __init__.py         # Package marker
│   ├── config.py           # Central configuration (paths, URIs, constants)
│   ├── data.py             # Dataset loader and preprocessor
│   ├── train.py            # Multi-model training and MLflow tracking
│   ├── register_model.py   # Champion selection and Model Registry
│   └── predict.py          # Registry-based inference demonstration
│
├── mlruns/                 # MLflow tracking data (auto-created)
└── plots/                  # Generated plot artifacts (auto-created)
```

---

## Installation

### Step 1: Open Terminal in Project Directory

```powershell
cd "C:\Users\user\OneDrive\Desktop\MLOPS 3 EXP"
```

### Step 2: Create Virtual Environment

```powershell
py -3.13 -m venv .venv
```

### Step 3: Activate Virtual Environment

```powershell
.\.venv\Scripts\activate
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

Or without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Execution Procedure

### Run the Complete Experiment

```powershell
py -3.13 run_experiment.py
```

This single command executes the entire workflow:

1. **STEP 1** — Load and validate the Breast Cancer dataset
2. **STEP 2-3** — Create MLflow experiment and train multiple models
3. **STEP 4-5** — Log all hyperparameters and evaluation metrics
4. **STEP 6** — Generate and log artifact plots (Confusion Matrix, ROC Curve, Feature Importance)
5. **STEP 7** — Log models with signatures using MLflow sklearn flavor
6. **STEP 8** — Search runs and find the best model by F1 score
7. **STEP 9** — Register the champion model in MLflow Model Registry
8. **STEP 10** — Assign the `champion` alias to the registered version
9. **STEP 11** — Load the champion model from registry using alias URI
10. **STEP 12** — Run batch and single-sample inference
11. **STEP 13** — Print final verification summary

> **Note:** The orchestrator auto-detects the `.venv` and uses it even when invoked with `py -3.13`.

---

## Expected Output

### Training Output

```
========================================
MLflow Training & Experiment Tracking
========================================
Experiment Name : Breast Cancer Classification
Tracking URI    : file:///C:/.../MLOPS 3 EXP/mlruns
Total Runs      : 6
----------------------------------------
Model     : Logistic Regression (LogisticRegression_L2_C1.0)
Run ID    : <actual_run_id>
Accuracy  : 0.9825
F1 Score  : 0.9861
ROC-AUC   : 0.9954
Log Loss  : 0.0777
Artifacts : confusion_matrix.png, roc_curve.png, feature_importance.png
----------------------------------------
... (5 more models)
```

### Comparison Table

```
                    run_name          model_type  accuracy  f1_score  roc_auc  log_loss
  LogisticRegression_L2_C1.0 Logistic Regression  0.982456  0.986111 0.995370  0.077746
  LogisticRegression_L2_C0.1 Logistic Regression  0.973684  0.979310 0.995701  0.106785
        RandomForest_n100_d5       Random Forest  0.956140  0.965517 0.993386  0.109513
       RandomForest_n200_d10       Random Forest  0.947368  0.958333 0.993717  0.112362
 GradientBoosting_n100_lr0.1   Gradient Boosting  0.956140  0.965986 0.990741  0.131651
GradientBoosting_n150_lr0.05   Gradient Boosting  0.929825  0.944444 0.983466  0.284405
```

### Champion Selection

```
Champion Model : LogisticRegression_L2_C1.0
Model Type     : Logistic Regression
F1 Score       : 0.9861 (Primary Metric)
ROC-AUC        : 0.9954 (Secondary Metric)
Accuracy       : 0.9825
```

### Final Verification

```
==================================================
FINAL EXPERIMENT VERIFICATION
==================================================
[PASS] Dataset loaded
[PASS] MLflow experiment created
[PASS] Multiple model runs completed
[PASS] Parameters logged
[PASS] Metrics logged
[PASS] Artifacts logged
[PASS] Models logged
[PASS] Champion selected
[PASS] Model registered
[PASS] Champion alias assigned
[PASS] Registry model loaded
[PASS] Prediction completed
```

---

## MLflow UI Procedure

### Launch the Tracking UI

**Option 1**: Double-click `launch_ui.bat`

**Option 2**: Run manually:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "file:///C:/Users/user/OneDrive/Desktop/MLOPS 3 EXP/mlruns" --host 127.0.0.1 --port 5000
```

> Set environment variables before running:
> ```powershell
> $env:MLFLOW_ALLOW_FILE_STORE="true"
> $env:MLFLOW_DISABLE_AGENT_HINT="1"
> ```

### Access the UI

Open a browser and navigate to:

```
http://127.0.0.1:5000
```

### What to Verify in MLflow UI

**Experiments Tab:**
- Select "Breast Cancer Classification" experiment
- View all 6 runs with parameters, metrics, and artifacts
- Compare metrics across runs using the Compare button
- Click any run to see detailed parameters, metrics, and logged artifacts

**Artifacts Tab (per run):**
- `plots/confusion_matrix.png` — Classification performance heatmap
- `plots/roc_curve.png` — ROC curve with AUC score
- `plots/feature_importance.png` — Top 10 feature importances
- `model/` — Logged MLflow model with signature and input example

---

## Model Registry Verification

In the MLflow UI:

1. Click **Models** in the navigation bar
2. Find `BreastCancerClassifier`
3. Verify:
   - Model version(s) exist
   - `@champion` alias is assigned to the best version
   - Tags include `dataset`, `task`, `model_type`, `selection_metric`
4. Click the version to see source run, signature, and metadata

---

## Prediction Verification

The prediction module demonstrates:

1. **Registry Loading**: Model loaded via `models:/BreastCancerClassifier@champion`
2. **Batch Prediction**: First 5 test samples with class labels and probabilities
3. **Single-Sample Prediction**: Individual sample with confidence percentage

Example output:
```
Sample 1: True=0 (malignant) | Pred=0 (malignant) | P(0)=1.0000, P(1)=0.0000 [CORRECT]
Sample 2: True=1 (benign)    | Pred=1 (benign)    | P(0)=0.0000, P(1)=1.0000 [CORRECT]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root directory |
| `MlflowException: filesystem tracking backend is in maintenance mode` | Set `MLFLOW_ALLOW_FILE_STORE=true` (already handled in `config.py`) |
| `DLL load failed: Application Control policy` | Use the `.venv` with `scikit-learn==1.5.2` (already configured) |
| `UntrustedTypesFoundException` with tree models | Use `serialization_format="cloudpickle"` (already configured) |
| MLflow UI doesn't start | Check port 5000 is free: `netstat -an | findstr 5000` |
| `py -3.13` not found | Use `.\.venv\Scripts\python.exe` directly |

---

## Result

The experiment was successfully executed with the following verified results:

| Check | Status |
|-------|--------|
| Dataset loaded (569 samples, 30 features) | ✅ PASS |
| MLflow experiment created | ✅ PASS |
| 6 model runs completed | ✅ PASS |
| Parameters logged | ✅ PASS |
| Metrics logged (accuracy, precision, recall, F1, ROC-AUC, log loss) | ✅ PASS |
| Artifacts logged (confusion matrix, ROC curve, feature importance) | ✅ PASS |
| Models logged with signatures | ✅ PASS |
| Champion selected (LogisticRegression_L2_C1.0, F1=0.9861) | ✅ PASS |
| Model registered (BreastCancerClassifier) | ✅ PASS |
| Champion alias assigned | ✅ PASS |
| Registry model loaded via alias | ✅ PASS |
| Prediction completed with probabilities | ✅ PASS |

**Champion Model**: Logistic Regression (C=1.0, L2 regularization)
- F1 Score: **0.9861**
- ROC-AUC: **0.9954**
- Accuracy: **0.9825**

---

## Conclusion

This experiment demonstrated a complete MLOps workflow using MLflow for experiment tracking and model registry management. Key takeaways:

1. **MLflow Tracking** enables systematic comparison of multiple models with different hyperparameters by logging parameters, metrics, and artifacts in a centralized store.

2. **Model Logging** with inferred signatures ensures that model inputs and outputs are well-documented, enabling safe deployment.

3. **Visual Artifacts** (confusion matrices, ROC curves, feature importance plots) provide interpretable diagnostics that help in model selection and validation.

4. **MLflow Model Registry** provides a centralized hub for model lifecycle management with versioning, tagging, and alias-based deployment.

5. **Alias-based Inference** (`models:/ModelName@champion`) decouples deployment from specific model versions, allowing seamless model updates without changing application code.

6. The **Logistic Regression** model with C=1.0 and L2 regularization was automatically selected as champion, demonstrating that simpler models can outperform ensemble methods on well-structured datasets.

---

## Viva Questions and Answers

### Q1. What is MLflow and what are its components?
**A:** MLflow is an open-source platform for managing the ML lifecycle. Its four components are: MLflow Tracking (logging experiments), MLflow Projects (packaging code), MLflow Models (model deployment), and MLflow Model Registry (model versioning and lifecycle management).

### Q2. What is experiment tracking and why is it important?
**A:** Experiment tracking is the process of recording parameters, metrics, code versions, and artifacts for each training run. It enables reproducibility, comparison of different approaches, and audit trails for model development.

### Q3. What is the difference between parameters and metrics in MLflow?
**A:** Parameters are input configuration values set before training (e.g., learning_rate=0.1, n_estimators=100). Metrics are output evaluation values computed after training (e.g., accuracy=0.98, f1_score=0.99).

### Q4. What are artifacts in MLflow?
**A:** Artifacts are output files associated with a run, such as trained model files, plots (confusion matrices, ROC curves), data files, or any other files generated during training.

### Q5. What is model signature in MLflow?
**A:** A model signature defines the expected input schema and output schema of a model. It is inferred using `mlflow.models.infer_signature()` and stored with the model to validate inputs during inference.

### Q6. What is the MLflow Model Registry?
**A:** The Model Registry is a centralized store for managing model versions. It supports model registration, versioning, tagging with metadata, and alias assignment for lifecycle management (champion, staging, etc.).

### Q7. What are model aliases in MLflow?
**A:** Aliases are mutable, named references to specific model versions. For example, the `champion` alias points to the current production model. Loading `models:/ModelName@champion` always returns the latest champion version.

### Q8. How does champion model selection work?
**A:** The system searches all completed experiment runs, sorts them by the primary metric (F1 Score) with a secondary tiebreaker (ROC-AUC), and selects the top-performing run as the champion.

### Q9. What is the difference between `mlflow.log_param()` and `mlflow.log_metric()`?
**A:** `log_param()` records a single input parameter (string key-value pair). `log_metric()` records a single numeric metric. Parameters are typically set once, while metrics can be logged at multiple steps.

### Q10. What is the tracking URI in MLflow?
**A:** The tracking URI specifies where MLflow stores experiment data. It can be a local file path (`file:./mlruns`), a SQLite database (`sqlite:///mlflow.db`), or a remote server (`http://mlflow-server:5000`).

### Q11. Why was a virtual environment used in this experiment?
**A:** A virtual environment isolates project dependencies from the system Python. In this case, it was also necessary to use scikit-learn 1.5.2 which is compatible with the Windows Application Control policy on the lab machine.

### Q12. How do you load a model from the MLflow Model Registry?
**A:** Use `mlflow.sklearn.load_model("models:/ModelName@alias")` or `mlflow.pyfunc.load_model("models:/ModelName@alias")`. The URI format `models:/` indicates registry access, and `@alias` specifies which version to load.

### Q13. What is the significance of ROC-AUC in binary classification?
**A:** ROC-AUC measures the model's ability to distinguish between classes across all probability thresholds. A score of 1.0 indicates perfect discrimination, while 0.5 indicates random guessing.

### Q14. What is F1 Score and why is it preferred over accuracy?
**A:** F1 Score is the harmonic mean of precision and recall. It is preferred over accuracy for imbalanced datasets because accuracy can be misleadingly high when the model simply predicts the majority class.

### Q15. What serialization format was used for model logging and why?
**A:** The `cloudpickle` format was used because the default `skops` format requires explicit trust declarations for tree-based models (`sklearn.tree._tree.Tree`), which adds complexity. Cloudpickle provides broader compatibility.
