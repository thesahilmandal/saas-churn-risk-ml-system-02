import sys
from typing import Optional

from src.components.etl import CustomerChurnETL
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_registry import ModelRegistry

from src.entity.config_entity import (
    TrainingPipelineConfig,
    ETLconfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig,
)

from src.entity.artifact_entity import (
    ETLArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

from src.logging import logging
from src.exception import CustomerChurnException


class TrainingPipeline:
    """
    Orchestrates the complete ML training lifecycle including
    ETL, data preparation, model training, evaluation, and registry update.

    Responsibilities
    ----------------
    1. Sequential execution of pipeline stages.
    2. Managing artifact flow between stages.
    3. Providing structured logging and failure handling.
    4. Maintaining deterministic execution behavior.
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self) -> None:
        try:
            self.pipeline_config = TrainingPipelineConfig()

            logging.info("==================================================")
            logging.info("TRAINING PIPELINE INITIALIZED")
            logging.info("==================================================\n")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Logging Helpers
    # ============================================================

    @staticmethod
    def _log_stage_start(stage_name: str) -> None:
        logging.info(f">>>>>> {stage_name} STARTED <<<<<<")

    @staticmethod
    def _log_stage_end(stage_name: str) -> None:
        logging.info(f">>>>>> {stage_name} COMPLETED <<<<<<\n")

    # ============================================================
    # Pipeline Stages
    # ============================================================

    def start_etl(self) -> ETLArtifact:
        try:
            self._log_stage_start("Stage 1: ETL")

            etl_config = ETLconfig(self.pipeline_config)
            etl = CustomerChurnETL(etl_config)
            artifact = etl.initiate_etl()

            self._log_stage_end("Stage 1: ETL")
            return artifact

        except Exception as e:
            logging.exception("ETL stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self._log_stage_start("Stage 2: Data Ingestion")

            ingestion_config = DataIngestionConfig(self.pipeline_config)
            ingestion = DataIngestion(ingestion_config)
            artifact = ingestion.initiate_data_ingestion()

            self._log_stage_end("Stage 2: Data Ingestion")
            return artifact

        except Exception as e:
            logging.exception("Data ingestion stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_validation(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            self._log_stage_start("Stage 3: Data Validation")

            validation_config = DataValidationConfig(self.pipeline_config)
            validation = DataValidation(
                validation_config,
                ingestion_artifact,
            )

            artifact = validation.initiate_data_validation()

            self._log_stage_end("Stage 3: Data Validation")
            return artifact

        except Exception as e:
            logging.exception("Data validation stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            self._log_stage_start("Stage 4: Data Transformation")

            transformation_config = DataTransformationConfig(
                self.pipeline_config
            )

            transformation = DataTransformation(
                transformation_config,
                ingestion_artifact,
                validation_artifact,
            )

            artifact = transformation.initiate_data_transformation()

            self._log_stage_end("Stage 4: Data Transformation")
            return artifact

        except Exception as e:
            logging.exception("Data transformation stage failed")
            raise CustomerChurnException(e, sys)

    def start_model_training(
        self,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            self._log_stage_start("Stage 5: Model Training")

            trainer_config = ModelTrainingConfig(self.pipeline_config)

            trainer = ModelTrainer(
                trainer_config,
                ingestion_artifact,
                transformation_artifact,
            )

            artifact = trainer.initiate_model_training()

            self._log_stage_end("Stage 5: Model Training")
            return artifact

        except Exception as e:
            logging.exception("Model training stage failed")
            raise CustomerChurnException(e, sys)

    def start_model_evaluation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            self._log_stage_start("Stage 6: Model Evaluation")

            evaluation_config = ModelEvaluationConfig(
                self.pipeline_config
            )

            evaluation = ModelEvaluation(
                evaluation_config,
                ingestion_artifact,
                trainer_artifact,
            )

            artifact = evaluation.initiate_model_evaluation()

            self._log_stage_end("Stage 6: Model Evaluation")
            return artifact

        except Exception as e:
            logging.exception("Model evaluation stage failed")
            raise CustomerChurnException(e, sys)

    def start_model_registry(
        self,
        evaluation_artifact: ModelEvaluationArtifact,
    ):
        try:
            self._log_stage_start("Stage 7: Model Registry")

            registry_config = ModelRegistryConfig()

            registry = ModelRegistry(
                registry_config,
                evaluation_artifact,
            )

            artifact = registry.initiate_model_registry()

            self._log_stage_end("Stage 7: Model Registry")
            return artifact

        except Exception as e:
            logging.exception("Model registry stage failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Pipeline Execution
    # ============================================================

    def run_pipeline(self):
        try:
            logging.info("==================================================")
            logging.info("TRAINING PIPELINE EXECUTION STARTED")
            logging.info("==================================================\n")

            etl_artifact = self.start_etl()

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact,
                validation_artifact,
            )

            trainer_artifact = self.start_model_training(
                ingestion_artifact,
                transformation_artifact,
            )

            evaluation_artifact = self.start_model_evaluation(
                ingestion_artifact,
                trainer_artifact,
            )

            registry_artifact = self.start_model_registry(
                evaluation_artifact
            )

            logging.info("==================================================")
            logging.info("TRAINING PIPELINE EXECUTION COMPLETED")
            logging.info("==================================================\n")

            return registry_artifact

        except Exception as e:
            logging.exception("Training pipeline execution failed")
            raise CustomerChurnException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception("Unhandled exception in main execution")
