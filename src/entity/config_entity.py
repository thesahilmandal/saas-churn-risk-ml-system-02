import os
import sys
from datetime import datetime, timezone
from typing import Optional

from src.constants import pipeline_constants
from src.exception import CustomerChurnException

class TrainingPipelineConfig:
    """
    Root configuration for training pipeline.
    Responsible for creating base artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.artifact_name: str = (
                pipeline_constants.ARTIFACT_DIR / "training_pipeline_runs"
            )

            self.artifact_dir: str = os.path.join(
                self.artifact_name,
                self.run_id,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# ETL Config
# ============================================================

class ETLconfig:
    """Configuration for ETL stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.etl_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.ETL_DIR_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_METADATA_FILE_NAME,
            )

            self.raw_data_dir: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_RAW_DATA_DIR_NAME,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Ingestion Config
# ============================================================

class DataIngestionConfig:
    """Configuration for data ingestion stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_INGESTION_DIR_NAME,
            )

            self.train_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_TRAIN_FILE_NAME,
            )

            self.test_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_TEST_FILE_NAME,
            )

            self.val_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_VAL_FILE_NAME,
            )

            self.schema_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_SCHEMA_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.data_ingestion_dir,
                pipeline_constants.DATA_INGESTION_METADATA_FILE_NAME,
            )

            self.train_temp_split_ratio: float = (
                pipeline_constants.DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO
            )

            self.test_val_split_ratio: float = (
                pipeline_constants.DATA_INGESTION_TEST_VAL_SPLIT_RATIO
            )

            self.random_state: int = pipeline_constants.RANDOM_STATE
            self.database_name: str = (
                pipeline_constants.DATA_INGESTION_DATABASE_NAME
            )
            self.collection_name: str = (
                pipeline_constants.DATA_INGESTION_COLLECTION_NAME
            )
            self.database_url: str = (
                pipeline_constants.DATA_INGESTION_MONGODB_URL
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Validation Config
# ============================================================

class DataValidationConfig:
    """Configuration for data validation stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.data_validation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_VALIDATION_DIR_NAME,
            )

            self.validation_report_file_path: str = os.path.join(
                self.data_validation_dir,
                pipeline_constants.DATA_VALIDATION_REPORT_FILE_NAME,
            )

            self.reference_schema_file_path: str = (
                pipeline_constants.REFERENCE_SCHEMA
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Data Transformation Config
# ============================================================

class DataTransformationConfig:
    """Configuration for data transformation stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.DATA_TRANSFORMATION_DIR_NAME,
            )

            self.lr_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants.DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME,
            )

            self.tree_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants.DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants.DATA_TRANSFORMATION_METADATA_FILE_NAME,
            )

            self.monitoring_baseline_file_path: str = os.path.join(
                self.data_transformation_dir,
                pipeline_constants.DATA_TRANSFORMATION_MONITORING_BASELINE_FILE_NAME,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Training Config
# ============================================================

class ModelTrainingConfig:
    """Configuration for model training stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.model_trainer_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.MODEL_TRAINING_DIR_NAME,
            )

            self.trained_models_dir: str = os.path.join(
                self.model_trainer_dir,
                pipeline_constants.MODEL_TRAINING_TRAINED_MODELS_DIR_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.model_trainer_dir,
                pipeline_constants.MODEL_TRAINING_METADATA_FILE_NAME,
            )

            self.models = pipeline_constants.MODEL_TRAINING_MODELS_REGISTERY
            self.models_hyperparameters = (
                pipeline_constants.MODEL_TRAINING_MODELS_HYPERPARAMETERS
            )

            self.primary_metric: str = (
                pipeline_constants.MODEL_TRAINING_PRIMARY_METRIC
            )

            self.decision_threshold: float = (
                pipeline_constants.MODEL_TRAINING_DECISION_THRESHOLD
            )

            self.n_iter: int = pipeline_constants.MODEL_TRAINING_N_ITER

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Evaluation Config
# ============================================================

class ModelEvaluationConfig:
    """Configuration for model evaluation stage."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.evaluation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                pipeline_constants.MODEL_EVALUATION_DIR_NAME,
            )

            self.evaluation_report_file_path: str = os.path.join(
                self.evaluation_dir,
                pipeline_constants.MODEL_EVALUATION_REPORT_FILE_NAME,
            )

            self.metadata_file_path: str = os.path.join(
                self.evaluation_dir,
                pipeline_constants.MODEL_EVALUATION_METADATA_FILE_NAME,
            )

            self.production_model_file_path: str = (
                pipeline_constants.PRODUCTION_MODEL_FILE_PATH
            )

            self.decision_threshold: float = (
                pipeline_constants.MODEL_EVALUATION_DECISION_THRESHOLD
            )

            self.recall_tolerance: float = (
                pipeline_constants.MODEL_EVALUATION_RECALL_TOLERANCE
            )

            self.min_recall_improvement: float = (
                pipeline_constants.MODEL_EVALUATION_MIN_IMPROVEMENT
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Registry Config
# ============================================================

class ModelRegistryConfig:
    """Configuration for model registry."""

    def __init__(self) -> None:
        try:
            self.registry_dir: str = pipeline_constants.MODEL_REGISTRY_DIR
            self.production_model_file_path: str = (
                pipeline_constants.PRODUCTION_MODEL_FILE_PATH
            )

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Model Monitoring Config
# ============================================================

class ModelMonitoringConfig:
    """Configuration for monitoring pipeline."""

    def __init__(self) -> None:
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.artifact_dir = pipeline_constants.ARTIFACT_DIR / "model_monitoring_runs" / self.run_id
            self.psi_threshold: float = 0.2
            self.drifted_feature_ratio_threshold: float = 0.3
            self.monitoring_sample_size: int = 1000
            self.strict_mode: bool = True

        except Exception as e:
            raise CustomerChurnException(e, sys) from e


# ============================================================
# Orchestrator Config
# ============================================================

class OrchestratorConfig:
    def __init__(self) -> None:
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.artifact_dir = pipeline_constants.ARTIFACT_DIR / "orchestrator_runs" / self.run_id
            self.metadata_path = self.artifact_dir / "run_metadata.json"
            self.monitoring_snapshot_path = self.artifact_dir / "monitoring_snapshot.json"
        except Exception as e:
            raise CustomerChurnException(e, sys)