@echo off
title MLflow Tracking UI - MLOps Experiment 3
echo ===================================================
echo Starting MLflow Tracking Server on Port 5000...
echo UI URL: http://127.0.0.1:5000
echo ===================================================

cd /d "%~dp0"
set MLFLOW_ALLOW_FILE_STORE=true
set MLFLOW_DISABLE_AGENT_HINT=1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri "file:///%~dp0mlruns" --host 127.0.0.1 --port 5000
) else (
    py -3.13 -m mlflow ui --backend-store-uri "file:///%~dp0mlruns" --host 127.0.0.1 --port 5000
)

pause
