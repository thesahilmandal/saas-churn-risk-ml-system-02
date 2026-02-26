from dataclasses import dataclass


@dataclass(frozen=True)
class ETLArtifact:
    raw_data_dir_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nETLArtifact(\n"
            f"  raw_data_file_path = {self.raw_data_dir_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact generated after the Data Ingestion stage.

    Attributes:
        train_file_path (str): Path to the training dataset
        test_file_path (str): Path to the test dataset
        val_file_path (str): Path to the validation dataset
        schema_file_path (str): Path to the schema file
        metadata_file_path (str): Path to metadata generated during ingestion
    """
    train_file_path: str
    test_file_path: str
    val_file_path: str
    schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataIngestionArtifact(\n"
            f"  train_file_path    = {self.train_file_path}\n"
            f"  test_file_path     = {self.test_file_path}\n"
            f"  val_file_path      = {self.val_file_path}\n"
            f"  schema_file_path   = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact generated after the Data Validation stage.

    Attributes:
        validation_status (bool): Whether validation passed or failed
        validation_report (str): Path to validation report
    """
    validation_status: bool
    validation_report: str

    def __str__(self) -> str:
        return (
            "\nDataValidationArtifact(\n"
            f"  validation_status = {self.validation_status}\n"
            f"  validation_report = {self.validation_report}\n"
            ")"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact generated after the Data Transformation stage.

    Responsibilities:
    -----------------
    Encapsulates all outputs produced by the transformation pipeline,
    including preprocessing artifacts, transformation metadata,
    and monitoring baseline reference.

    Attributes:
        tree_preprocessor_file_path (str):
            Path to fitted tree-based preprocessor.

        linear_preprocessor_file_path (str):
            Path to fitted linear-model preprocessor.

        metadata_file_path (str):
            Path to transformation metadata artifact.

        monitoring_baseline_file_path (str):
            Path to monitoring baseline artifact used for
            drift detection and observability pipelines.
    """

    tree_preprocessor_file_path: str
    linear_preprocessor_file_path: str
    metadata_file_path: str
    monitoring_baseline_file_path: str

    def __str__(self) -> str:
        """
        Human-readable artifact representation for structured logging.
        """
        return (
            "\nDataTransformationArtifact(\n"
            f"  tree_preprocessor_file_path        = {self.tree_preprocessor_file_path}\n"
            f"  linear_preprocessor_file_path      = {self.linear_preprocessor_file_path}\n"
            f"  metadata_file_path                 = {self.metadata_file_path}\n"
            f"  monitoring_baseline_file_path      = {self.monitoring_baseline_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_models_dir: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nModelTrainingArtifact(\n"
            f"  trained_models_dir = {self.trained_models_dir}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Artifact produced by Model Evaluation Pipeline.

    Contains:
    - Selected best candidate model
    - Evaluation outputs
    - Approval decision for model registration
    """

    best_model_name: str
    best_model_path: str
    evaluation_report_path: str
    metadata_path: str
    approval_status: bool

    def __str__(self) -> str:
        return (
            "\nModelEvaluationArtifact(\n"
            f"  best_model_name        = {self.best_model_name}\n"
            f"  best_model_path        = {self.best_model_path}\n"
            f"  evaluation_report_path = {self.evaluation_report_path}\n"
            f"  metadata_path          = {self.metadata_path}\n"
            f"  approval_status        = {self.approval_status}\n"
            ")"
        )


@dataclass(frozen=True)
class ModelMonitoringArtifact:
    """
    Artifact generated after Model Monitoring stage.

    Responsibilities:
    -----------------
    - Provide structured reference to monitoring outputs
    - Encapsulate drift results and retraining decision
    - Maintain immutability for deterministic lineage

    Attributes:
        artifact_dir (str):
            Timestamped monitoring directory.

        report_file_path (str):
            Path to drift report JSON.

        metadata_file_path (str):
            Path to monitoring metadata JSON.

        retraining_flag_file_path (str):
            Path to retraining decision JSON.

        retraining_required (bool):
            Boolean flag indicating whether retraining is required.
    """

    artifact_dir: str
    report_file_path: str
    metadata_file_path: str
    retraining_flag_file_path: str
    retraining_required: bool

    def __str__(self) -> str:
        return (
            "\nModelMonitoringArtifact(\n"
            f"  artifact_dir               = {self.artifact_dir}\n"
            f"  report_file_path           = {self.report_file_path}\n"
            f"  metadata_file_path         = {self.metadata_file_path}\n"
            f"  retraining_flag_file_path  = {self.retraining_flag_file_path}\n"
            f"  retraining_required        = {self.retraining_required}\n"
            ")"
        )