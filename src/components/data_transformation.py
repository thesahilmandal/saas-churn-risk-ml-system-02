"""
Data Transformation Pipeline.

Responsibilities:
- Build model-aware preprocessing pipelines
- Fit preprocessors on training data only (no leakage)
- Persist fitted preprocessors
- Generate transformation metadata
- Generate monitoring baseline artifact for drift detection

Design Guarantees:
- Validation-gated execution
- Deterministic preprocessing
- No data leakage
- Reproducible experiment metadata
- Monitoring baseline aligned with training distribution
"""

import os
import sys
import json
import hashlib
import platform
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file, save_object, read_json_file
from src.constants.pipeline_constants import TARGET_COLUMN, MONITORING_BASELINE_PATH


class DataTransformation:
    """
    Production-grade data transformation pipeline.

    This component:
    - Builds preprocessing pipelines for different model types
    - Fits transformations on training data only
    - Persists preprocessors
    - Generates experiment-comparable metadata
    - Produces monitoring baseline artifact for drift detection
    """

    PIPELINE_VERSION = "1.1.0"

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        try:
            logging.info("[DATA TRANSFORMATION INIT] Initializing")

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed. Transformation aborted."
                )

            self.config = transformation_config
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact

            os.makedirs(
                self.config.data_transformation_dir,
                exist_ok=True,
            )

            logging.info(
                "[DATA TRANSFORMATION INIT] Initialized | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            logging.exception("[DATA TRANSFORMATION INIT] Failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Utility Methods
    # ============================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        """Read CSV safely with existence check."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def _compute_hash(obj: Any) -> str:
        """Compute stable SHA-256 hash for JSON-serializable object."""
        payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _get_feature_groups(
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """
        Split features into numerical and categorical groups
        with strict validation.
        """
        numeric = X.select_dtypes(include=["int", "float"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]

        if not numeric:
            raise ValueError("No numeric features detected.")

        if not categorical:
            raise ValueError("No categorical features detected.")

        return numeric, categorical

    # ============================================================
    # Preprocessor Builders
    # ============================================================

    def _build_linear_preprocessor(
        self, num_features: List[str], cat_features: List[str]
    ) -> ColumnTransformer:
        """
        Preprocessor optimized for linear models.
        """

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    def _build_tree_preprocessor(
        self, num_features: List[str], cat_features: List[str]
    ) -> ColumnTransformer:
        """
        Preprocessor optimized for tree-based models.
        """

        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    # ============================================================
    # Encoder Metadata Extraction
    # ============================================================

    def _extract_encoder_metadata(
        self, preprocessor: ColumnTransformer, cat_features: List[str]
    ) -> Dict[str, Any]:
        """
        Extract encoder metadata including:
        - Categorical cardinality
        - Encoded feature names
        """

        metadata: Dict[str, Any] = {
            "categorical_cardinality": {},
            "output_feature_names": [],
        }

        for name, transformer, features in preprocessor.transformers_:
            if name == "cat":
                encoder: OneHotEncoder = transformer.named_steps["encoder"]
                for feature, categories in zip(features, encoder.categories_):
                    metadata["categorical_cardinality"][feature] = len(categories)

                metadata["output_feature_names"].extend(
                    encoder.get_feature_names_out(features).tolist()
                )

            if name == "num":
                metadata["output_feature_names"].extend(features)

        metadata["output_feature_count"] = len(
            metadata["output_feature_names"]
        )

        return metadata

    # ============================================================
    # Monitoring Baseline Generation
    # ============================================================

    def _compute_numerical_baseline(
        self, X_train: pd.DataFrame, num_features: List[str]
    ) -> Dict[str, Any]:
        """Compute baseline statistics for numeric features."""
        baseline = {}

        for col in num_features:
            series = X_train[col]

            baseline[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "quantiles": {
                    "p10": float(series.quantile(0.10)),
                    "p25": float(series.quantile(0.25)),
                    "p50": float(series.quantile(0.50)),
                    "p75": float(series.quantile(0.75)),
                    "p90": float(series.quantile(0.90)),
                },
                "missing_ratio": float(series.isna().mean()),
            }

        return baseline

    def _compute_categorical_baseline(
        self, X_train: pd.DataFrame, cat_features: List[str]
    ) -> Dict[str, Any]:
        """Compute baseline distributions for categorical features."""
        baseline = {}

        for col in cat_features:
            series = X_train[col]
            value_counts = (
                series.value_counts(normalize=True, dropna=False).to_dict()
            )

            baseline[col] = {
                "cardinality": int(series.nunique(dropna=True)),
                "distribution": {
                    str(k): float(v) for k, v in value_counts.items()
                },
                "missing_ratio": float(series.isna().mean()),
            }

        return baseline

    def _generate_monitoring_baseline(
        self,
        X_train: pd.DataFrame,
        num_features: List[str],
        cat_features: List[str],
        linear_preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:
        """
        Generate monitoring baseline artifact aligned with training distribution.
        """

        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )

        numerical_stats = self._compute_numerical_baseline(
            X_train, num_features
        )

        categorical_stats = self._compute_categorical_baseline(
            X_train, cat_features
        )

        encoder_metadata = self._extract_encoder_metadata(
            linear_preprocessor, cat_features
        )

        return {
            "pipeline": {
                "name": "monitoring_baseline",
                "version": self.PIPELINE_VERSION,
            },
            "dataset_reference": {
                "train_checksum": ingestion_metadata["split"]["checksums"]["train"],
                "train_rows": len(X_train),
                "feature_count": len(X_train.columns),
            },
            "feature_groups": {
                "numerical": num_features,
                "categorical": cat_features,
            },
            "numerical_statistics": numerical_stats,
            "categorical_statistics": categorical_stats,
            "encoded_feature_mapping": encoder_metadata,
            "schema_snapshot": {
                "feature_names": list(X_train.columns),
                "dtypes": {
                    col: str(dtype)
                    for col, dtype in X_train.dtypes.items()
                },
            },
            "drift_thresholds": {
                "psi_threshold": 0.25,
                "mean_shift_threshold": 0.15,
                "prediction_mean_shift_threshold": 0.10,
                "min_records_required": 200,
            },
            "governance": {
                "validation_passed": self.validation_artifact.validation_status,
                "transformation_version": self.PIPELINE_VERSION,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Metadata Generation (Existing)
    # ============================================================

    def _generate_metadata(
        self,
        X_train: pd.DataFrame,
        num_features: List[str],
        cat_features: List[str],
        linear_preprocessor: ColumnTransformer,
        tree_preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:

        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )

        linear_meta = self._extract_encoder_metadata(
            linear_preprocessor, cat_features
        )
        tree_meta = self._extract_encoder_metadata(
            tree_preprocessor, cat_features
        )

        transformation_config = {
            "numerical_features": num_features,
            "categorical_features": cat_features,
        }

        return {
            "pipeline": {
                "name": "data_transformation",
                "version": self.PIPELINE_VERSION,
            },
            "input": {
                "dataset_checksum": ingestion_metadata["split"]["checksums"]["train"],
                "rows": len(X_train),
                "features": len(X_train.columns),
            },
            "feature_groups": {
                "numerical": num_features,
                "categorical": cat_features,
            },
            "output_schema": {
                "linear": linear_meta,
                "tree": tree_meta,
            },
            "transformation_fingerprint": {
                "config_hash": self._compute_hash(transformation_config),
            },
            "environment": {
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:
            logging.info("[DATA TRANSFORMATION PIPELINE] Started")

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in training data"
                )

            X_train = train_df.drop(columns=[TARGET_COLUMN])

            num_features, cat_features = self._get_feature_groups(X_train)

            # Fit preprocessors
            linear_preprocessor = self._build_linear_preprocessor(
                num_features, cat_features
            )
            linear_preprocessor.fit(X_train)

            tree_preprocessor = self._build_tree_preprocessor(
                num_features, cat_features
            )
            tree_preprocessor.fit(X_train)

            save_object(
                self.config.lr_preprocessor_file_path,
                linear_preprocessor,
            )

            save_object(
                self.config.tree_preprocessor_file_path,
                tree_preprocessor,
            )

            # Metadata
            metadata = self._generate_metadata(
                X_train,
                num_features,
                cat_features,
                linear_preprocessor,
                tree_preprocessor,
            )

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            # Monitoring Baseline
            monitoring_baseline = self._generate_monitoring_baseline(
                X_train,
                num_features,
                cat_features,
                linear_preprocessor,
            )

            monitoring_baseline_path = self.config.monitoring_baseline_file_path            
            write_json_file(
                monitoring_baseline_path,
                monitoring_baseline,
            )
            write_json_file(MONITORING_BASELINE_PATH, monitoring_baseline)
            
            artifact = DataTransformationArtifact(
                tree_preprocessor_file_path=self.config.tree_preprocessor_file_path,
                linear_preprocessor_file_path=self.config.lr_preprocessor_file_path,
                metadata_file_path=self.config.metadata_file_path,
                monitoring_baseline_file_path=self.config.monitoring_baseline_file_path
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Completed successfully"
            )
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[DATA TRANSFORMATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
