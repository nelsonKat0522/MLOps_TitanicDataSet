# MLOps Lifecycle Project

## Project Overview
This project builds an end-to-end machine learning pipeline for classification using the Titanic dataset as a proxy for patient readmission risk.

## Tools Used
Python, FastAPI, MLflow, Evidently AI, GitHub Actions

## Workflow
1. Data exploration
2. Data preprocessing
3. Model training
4. Experiment tracking with MLflow
5. Data and model versioning with Mlflow
6. REST API deployment with FastAPI
7. CI/CD with GitHub Actions
8. Monitoring with Evidently AI

## How to Run
pip install -r requirements.txt
python Unit5_MLOps_endProject_NganHuynh.py
uvicorn app.main:app --reload

## Monitoring
python monitoring/monitor.py

## Best Practices
- Track code with Git
- Track data and models with Mlflow
- Track experiments with MLflow
- Test API before deployment
- Monitor data drift regularly
