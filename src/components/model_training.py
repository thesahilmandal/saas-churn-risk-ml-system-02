"""
Model Training Pipeline.

Responsibilities:
- Train candidate models using pre-built preprocessors
- Perform bounded hyperparameter optimization
- Generate cross-validation and validation metrics
- Persist candidate model artifacts
- Produce experiment-comparable training metadata

IMPORTANT:
- This pipeline DOES NOT decide deployment.
- Model approval is handled by Model Evaluation pipeline.
"""

import os
import sys
import platform
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)

from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.constants.pipeline_constants import TARGET_COLUMN, RANDOM_STATE
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    save_object,
    load_object,
    write_json_file,
    read_json_file,
)


class ModelTrainer:
    """
    Production-grade model training pipeline.

    Guarantees:
    - Deterministic training
    - No data leakage
    - Candidate model generation only
    - Full lineage to upstream pipelines
    """

    PIPELINE_VERSION = "1.1.0"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        model_trainer_config: ModelTrainingConfig,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            logging.info("[MODEL TRAINER INIT] Initializing")

            self.config = model_trainer_config
            self.ingestion_artifact = ingestion_artifact
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.trained_models_dir, exist_ok=True)

            logging.info(
                "[MODEL TRAINER INIT] Initialized | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def _build_pipeline(preprocessor, model) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    # ============================================================
    # VALIDATION METRICS (DIAGNOSTIC ONLY)
    # ============================================================

    def _compute_validation_metrics(
        self,
        model_pipeline: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:

        y_prob = model_pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= self.config.decision_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        return {
            "n_samples": int(len(y_val)),
            "threshold": self.config.decision_threshold,
            "metrics": {
                "accuracy": round(accuracy_score(y_val, y_pred), 6),
                "precision": round(precision_score(y_val, y_pred, zero_division=0), 6),
                "recall": round(recall_score(y_val, y_pred, zero_division=0), 6),
                "f1": round(f1_score(y_val, y_pred, zero_division=0), 6),
                "roc_auc": round(roc_auc_score(y_val, y_prob), 6),
                "pr_auc": round(average_precision_score(y_val, y_prob), 6),
                "log_loss": round(log_loss(y_val, y_prob), 6),
            },
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    # ============================================================
    # TRAIN SINGLE MODEL
    # ============================================================

    def _train_model(
        self,
        model_name: str,
        model,
        param_grid: Dict,
        preprocessor,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:

        logging.info(f"[MODEL TRAINING] Started | model={model_name}")

        pipeline = self._build_pipeline(preprocessor, model)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                f"model__{k}": v for k, v in param_grid.items()
            },
            n_iter=self.config.n_iter,
            scoring=self.config.primary_metric,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_

        # ---------- Artifact Directory ----------
        model_dir = os.path.join(
            self.config.trained_models_dir,
            model_name,
        )
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pkl")
        metrics_path = os.path.join(model_dir, "validation_metrics.json")

        save_object(model_path, best_pipeline)

        validation_metrics = self._compute_validation_metrics(
            best_pipeline, X_val, y_val
        )

        write_json_file(metrics_path, validation_metrics)

        logging.info(
            "[MODEL TRAINING] Completed | "
            f"model={model_name}"
        )

        return {
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "artifact_path": model_path,
            "validation_metrics_path": metrics_path,
            "best_hyperparameters": search.best_params_,
            "cv": {
                "strategy": "StratifiedKFold",
                "n_splits": cv.get_n_splits(),
                "scoring": self.config.primary_metric,
                "mean_score": round(float(search.best_score_), 6),
                "std_score": round(
                    float(
                        search.cv_results_["std_test_score"][
                            search.best_index_
                        ]
                    ),
                    6,
                ),
            },
        }

    # ============================================================
    # METADATA
    # ============================================================

    def _generate_training_metadata(
        self,
        models_summary: Dict[str, Dict],
        started_at_utc: str,
        completed_at_utc: str,
    ) -> None:

        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )

        transformation_metadata = read_json_file(
            self.transformation_artifact.metadata_file_path
        )

        metadata = {
            "pipeline": {
                "name": "model_training",
                "version": self.PIPELINE_VERSION,
            },
            "timing": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "input": {
                "dataset_checksum":
                    ingestion_metadata["split"]["checksums"]["train"],
                "rows":
                    ingestion_metadata["split"]["counts"]["train"],
            },
            "preprocessing": {
                "transformation_fingerprint":
                    transformation_metadata.get(
                        "transformation_fingerprint"
                    ),
            },
            "training": {
                "primary_metric": self.config.primary_metric,
                "models": models_summary,
                "search_strategy": "RandomizedSearchCV",
                "n_iter": self.config.n_iter,
                "random_state": RANDOM_STATE,
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            },
            "created_at_utc":
                datetime.now(timezone.utc).isoformat(),
        }

        write_json_file(
            self.config.metadata_file_path,
            metadata,
        )

    # ============================================================
    # ENTRY POINT
    # ============================================================

    def initiate_model_training(self) -> ModelTrainerArtifact:

        try:
            logging.info("[MODEL TRAINING PIPELINE] Started")

            started_at_utc = datetime.now(timezone.utc).isoformat()

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )
            val_df = self._read_csv(
                self.ingestion_artifact.val_file_path
            )

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_val = val_df.drop(columns=[TARGET_COLUMN])
            y_val = val_df[TARGET_COLUMN]

            models_summary: Dict[str, Dict] = {}

            for model_name, model in self.config.models.items():

                param_grid = self.config.models_hyperparameters.get(
                    model_name, {}
                )

                preprocessor_path = (
                    self.transformation_artifact.linear_preprocessor_file_path
                    if model_name == "logistic_regression"
                    else self.transformation_artifact.tree_preprocessor_file_path
                )

                preprocessor = load_object(preprocessor_path)

                model_metadata = self._train_model(
                    model_name=model_name,
                    model=model,
                    param_grid=param_grid,
                    preprocessor=preprocessor,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )

                models_summary[model_name] = model_metadata

            completed_at_utc = datetime.now(timezone.utc).isoformat()

            self._generate_training_metadata(
                models_summary,
                started_at_utc,
                completed_at_utc,
            )

            artifact = ModelTrainerArtifact(
                trained_models_dir=self.config.trained_models_dir,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("[MODEL TRAINING PIPELINE] Completed successfully")
            logging.info(artifact)
            
            return artifact

        except Exception as e:
            logging.exception("[MODEL TRAINING PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
