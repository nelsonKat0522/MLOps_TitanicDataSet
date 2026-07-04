"""
Function-based training script for Titanic survival prediction.

This file is cleaned from the notebook-converted Python script.
Notebook/EDA plots were removed so the file can run automatically in CI/CD.

Pipeline:
1. load_data()
2. preprocess_data(data)
3. train_model(X_train, y_train)
4. evaluate_model(model, X_test, y_test)
5. log_to_mlflow(model, metrics)
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "titanic.csv"
MODEL_PATH = "models/titanic_logistic_regression.pkl"
SCALER_PATH = "models/scaler.pkl"
EXPERIMENT_NAME = "Titanic_Survived_Prediction"
REGISTERED_MODEL_NAME = "TitanicLogisticRegression"
RANDOM_STATE = 42


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load Titanic dataset from CSV file.

    Args:
        data_path: Path to the Titanic CSV file.

    Returns:
        pandas DataFrame.
    """
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data: pd.DataFrame):
    """
    Clean, encode, split, balance, and scale the dataset.

    Steps:
    - Fill missing Age with mean.
    - Fill missing Cabin with "Unknown".
    - Fill missing Embarked with the most common value "S".
    - Drop unnecessary high-cardinality or ID columns.
    - Encode Sex and Embarked.
    - Split into train/test sets.
    - Apply SMOTE only to training data.
    - Scale numeric features.

    Args:
        data: Raw Titanic DataFrame.

    Returns:
        X_train_scale, X_test_scale, y_train_smote, y_test
    """
    data = data.copy()

    # Fill missing values
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Cabin"] = data["Cabin"].fillna("Unknown")
    data["Embarked"] = data["Embarked"].fillna("S")

    # Remove unnecessary columns
    train_data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Convert categorical features to numeric features
    train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
    train_data = pd.get_dummies(train_data, columns=["Embarked"], drop_first=True, dtype=int)

    # Split features and target
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train_smote)
    X_test_scale = scaler.transform(X_test)

    # Save scaler for FastAPI prediction later
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_train_scale, X_test_scale, y_train_smote, y_test


def train_model(X_train, y_train) -> LogisticRegression:
    """
    Train Logistic Regression model.

    Args:
        X_train: Scaled training features.
        y_train: Balanced training target.

    Returns:
        Trained Logistic Regression model.
    """
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1500)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate trained model.

    Args:
        model: Trained model.
        X_test: Scaled testing features.
        y_test: Testing target.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
    }

    print("Model Evaluation Metrics")
    print("------------------------")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


def log_to_mlflow(model, metrics: dict):
    """
    Log model parameters, metrics, and model artifact to MLflow.

    Args:
        model: Trained model.
        metrics: Evaluation metrics dictionary.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1500)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        model_info = mlflow.sklearn.log_model(model, name="model")
        model_uri = model_info.model_uri

        # Register model
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        print(f"Run ID: {run.info.run_id}")
        print("Model logged and registered successfully.")


if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    log_to_mlflow(model, metrics)
