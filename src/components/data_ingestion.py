"""
Data Ingestion Pipeline for Customer Churn Project.

Responsibilities:
- Load cleaned customer data from MongoDB
- Perform stratified train / validation / test split
- Persist datasets, schema, and experiment lineage metadata
"""

import os
import sys
import hashlib
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.exception import CustomerChurnException
from src.logging import logging
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.pipeline_constants import TARGET_COLUMN
from src.utils.main_utils import write_json_file

load_dotenv()


class DataIngestion:
    """
    Production-grade data ingestion pipeline with full dataset lineage,
    reproducibility guarantees, and experiment-comparable metadata.
    """

    PIPELINE_VERSION = "1.0.0"

    def __init__(self, config: DataIngestionConfig) -> None:
        try:
            self.config = config
            self.target_column = TARGET_COLUMN

            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            logging.info(
                "[DATA INGESTION INIT] Initialized | "
                f"dataset=customer_churn | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_checksum(df: pd.DataFrame) -> str:
        """
        Compute SHA-256 checksum for a DataFrame.
        Ensures dataset identity and reproducibility.
        """
        hash_bytes = pd.util.hash_pandas_object(
            df, index=True
        ).values.tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def _load_from_mongodb(self) -> pd.DataFrame:
        """
        Load records from MongoDB in a deterministic manner.
        """
        try:
            logging.info(
                "[DATA INGESTION] Connecting to MongoDB | "
                f"db={self.config.database_name}, "
                f"collection={self.config.collection_name}"
            )

            with pymongo.MongoClient(self.config.database_url) as client:
                collection = client[
                    self.config.database_name
                ][self.config.collection_name]

                records = list(collection.find({}, {"_id": 0}))

            if not records:
                raise ValueError("MongoDB collection returned zero records")

            df = pd.DataFrame(records)

            logging.info(
                "[DATA INGESTION] MongoDB read successful | "
                f"rows={len(df)}, columns={len(df.columns)}"
            )

            return df

        except Exception as e:
            logging.exception("[DATA INGESTION] MongoDB read failed")
            raise CustomerChurnException(e, sys)

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deterministic, logged cleaning logic.
        """
        try:
            initial_rows = len(df)

            drop_columns = [
                "data_source",
                "ingested_at_utc",
                "customerID",
                "_id"
            ]

            df = df.drop(
                columns=[c for c in drop_columns if c in df.columns]
            )

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            )
            
            df['SeniorCitizen'] = df['SeniorCitizen'].map({1: "Yes", 0: "No"})
            df[self.target_column] = df[self.target_column].map({"Yes": 1, "No": 0})

            df = df.replace({"na": np.nan})
            df = df.drop_duplicates()

            if df[self.target_column].isna().any():
                raise ValueError(
                    "Target column contains null values after cleaning"
                )

            logging.info(
                "[DATA INGESTION] Cleaning completed | "
                f"rows_before={initial_rows}, rows_after={len(df)}"
            )

            return df

        except Exception as e:
            logging.exception("[DATA INGESTION] Data cleaning failed")
            raise CustomerChurnException(e, sys)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform reproducible stratified split.
        """
        try:
            if df[self.target_column].nunique() < 2:
                raise ValueError(
                    "Stratified split requires at least two target classes"
                )

            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.train_temp_split_ratio,
                random_state=self.config.random_state,
                stratify=df[self.target_column],
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.test_val_split_ratio,
                random_state=self.config.random_state,
                stratify=temp_df[self.target_column],
            )

            logging.info(
                "[DATA INGESTION] Split completed | "
                f"train={len(train_df)}, "
                f"val={len(val_df)}, "
                f"test={len(test_df)}"
            )

            return train_df, val_df, test_df

        except Exception as e:
            logging.exception("[DATA INGESTION] Data split failed")
            raise CustomerChurnException(e, sys)

    # ------------------------------------------------------------------
    # Schema & Metadata
    # ------------------------------------------------------------------
    def _generate_schema(self, train_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate schema strictly from training data.
        """
        schema: Dict[str, Dict] = {}

        for column in train_df.columns:
            col = train_df[column]

            schema[column] = {
                "dtype": str(col.dtype),
                "nullable": bool(col.isna().any()),
                "unique_values": int(col.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(col):
                schema[column].update(
                    {
                        "min": float(col.min()),
                        "max": float(col.max()),
                        "mean": float(col.mean()),
                        "median": float(col.median())
                    }
                )

        return schema

    def _target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        counts = df[self.target_column].value_counts(normalize=True)
        return {str(k): round(v, 4) for k, v in counts.items()}

    def _generate_metadata(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """
        Generate experiment-comparable lineage metadata.
        """
        total = len(train_df) + len(val_df) + len(test_df)

        return {
            "dataset": {
                "id": "customer_churn",
                "pipeline_version": self.PIPELINE_VERSION,
                "target_column": self.target_column,
                "feature_count": cleaned_df.shape[1] - 1,
            },
            "source": {
                "type": "mongodb",
                "database": self.config.database_name,
                "collection": self.config.collection_name,
                "raw_rows": len(raw_df),
                "cleaned_rows": len(cleaned_df),
                "raw_checksum": self._compute_checksum(raw_df),
                "cleaned_checksum": self._compute_checksum(cleaned_df),
            },
            "split": {
                "strategy": "stratified",
                "random_state": self.config.random_state,
                "counts": {
                    "train": len(train_df),
                    "validation": len(val_df),
                    "test": len(test_df),
                },
                "ratios": {
                    "train": round(len(train_df) / total, 4),
                    "validation": round(len(val_df) / total, 4),
                    "test": round(len(test_df) / total, 4),
                },
                "target_distribution": {
                    "train": self._target_distribution(train_df),
                    "validation": self._target_distribution(val_df),
                    "test": self._target_distribution(test_df),
                },
                "checksums": {
                    "train": self._compute_checksum(train_df),
                    "validation": self._compute_checksum(val_df),
                    "test": self._compute_checksum(test_df),
                },
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Pipeline Entry Point
    # ------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the data ingestion pipeline end-to-end.
        """
        try:
            logging.info("[DATA INGESTION PIPELINE] Started")

            raw_df = self._load_from_mongodb()
            cleaned_df = self._clean_dataframe(raw_df)

            train_df, val_df, test_df = self._split_data(cleaned_df)

            for path in [
                self.config.train_file_path,
                self.config.val_file_path,
                self.config.test_file_path,
                self.config.schema_file_path,
                self.config.metadata_file_path,
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            train_df.to_csv(self.config.train_file_path, index=False)
            val_df.to_csv(self.config.val_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            schema = self._generate_schema(train_df)
            write_json_file(self.config.schema_file_path, schema)

            metadata = self._generate_metadata(
                raw_df, cleaned_df, train_df, val_df, test_df
            )
            write_json_file(self.config.metadata_file_path, metadata)

            artifact = DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                val_file_path=self.config.val_file_path,
                test_file_path=self.config.test_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("[DATA INGESTION PIPELINE] Completed successfully")
            logging.info(artifact)
            return artifact

        except Exception as e:
            logging.exception("[DATA INGESTION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)