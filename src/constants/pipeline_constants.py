"""
Centralized configuration constants for the ML training pipeline.

This module defines directory names, file names, environment variables,
and pipeline-wide constants used across different stages.
"""

from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Load environment variables once at module import
load_dotenv()


# -------------------------------------------------------------------------
# Global Pipeline Constants
# -------------------------------------------------------------------------

TARGET_COLUMN: str = "Churn"
LOGS_DIR: Path = Path("logs")
ARTIFACT_DIR: Path = Path("artifacts")
S3_TRAINING_BUCKET_NAME: str = "saas-customer-churn-ml-02"
S3_ARTIFACT_DIR_NAME: str = "artifacts"
S3_MODEL_REGISTRY_DIR_NAME: str = "model_registry"
RANDOM_STATE = 42
REFERENCE_SCHEMA: Path = Path("data_schema") / "schema.yaml"
PRODUCTION_MODEL_DIR: Path = Path("production_model")
PRODUCTION_MODEL_FILE_PATH: Path = Path("production_model") / "model.pkl"
MODEL_REGISTRY_DIR: Path = Path("model_registry")
MODEL_REGISTRY_METADATA_PATH: Path = Path("model_registry") / "registry_metadata.json"
MONITORING_BASELINE_PATH = Path("online") / "monitoring_baseline.json"
LOCK_FILE_PATH = "/tmp/churn_orchestrator.lock"

# -------------------------------------------------------------------------
# ETL Constants
# -------------------------------------------------------------------------

ETL_DIR_NAME: str = "01_etl"
ETL_METADATA_FILE_NAME: str = "metadata.json"
ETL_RAW_DATA_DIR_NAME: str = "raw_data"


# -------------------------------------------------------------------------
# Data Ingestion Constants
# -------------------------------------------------------------------------

DATA_INGESTION_DIR_NAME: str = "02_data_ingestion"

DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"

DATA_INGESTION_SCHEMA_FILE_NAME: str = "ingestion_schema.json"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.30
DATA_INGESTION_TEST_VAL_SPLIT_RATIO: float = 0.50


# Environment variables (validated at runtime, not import time)
DATA_INGESTION_DATABASE_NAME: str | None = os.getenv("MONGODB_DATABASE")
DATA_INGESTION_COLLECTION_NAME: str | None = os.getenv("MONGODB_COLLECTION")
DATA_INGESTION_MONGODB_URL: str | None = os.getenv("MONGODB_URL")


# -------------------------------------------------------------------------
# Data Validation Constants
# -------------------------------------------------------------------------

DATA_VALIDATION_DIR_NAME: str = "03_data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.json"


# -------------------------------------------------------------------------
# Data Transformation Constants
# -------------------------------------------------------------------------

DATA_TRANSFORMATION_DIR_NAME: str = "04_data_transformation"
DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME: str = "lr_preprocessor.pkl"
DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME: str = "tree_preprocessor.pkl"
DATA_TRANSFORMATION_METADATA_FILE_NAME: str = "metadata.json"
DATA_TRANSFORMATION_MONITORING_BASELINE_FILE_NAME = "monitoring_baseline.json"


# -------------------------------------------------------------------------
# Model Training Constants
# -------------------------------------------------------------------------

MODEL_TRAINING_DIR_NAME: str = "05_model_training"
MODEL_TRAINING_TRAINED_MODELS_DIR_NAME: str = "trained_models"
MODEL_TRAINING_METADATA_FILE_NAME: str = "metadata.json"
MODEL_TRAINING_MODELS_REGISTERY: dict = {
    "logistic_regression": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }
MODEL_TRAINING_MODELS_HYPERPARAMETERS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "max_iter": [100, 300, 500],
        "solver": ["liblinear", "lbfgs", "saga"],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    "gradient_boosting": {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    },   
}
MODEL_TRAINING_PRIMARY_METRIC = "recall"
MODEL_TRAINING_DECISION_THRESHOLD = 0.5
MODEL_TRAINING_N_ITER = 1


# -------------------------------------------------------------------------
# Model Evaluation Constants
# -------------------------------------------------------------------------

MODEL_EVALUATION_DIR_NAME: str = "06_model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.json"
MODEL_EVALUATION_METADATA_FILE_NAME: str = "evaluation_metadata.json"
MODEL_EVALUATION_DECISION_THRESHOLD: float = 0.50
MODEL_EVALUATION_RECALL_TOLERANCE: float = 0.005
MODEL_EVALUATION_MIN_IMPROVEMENT: float = 0.01
