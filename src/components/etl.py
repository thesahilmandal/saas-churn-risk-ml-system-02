"""
ETL pipeline for Customer Churn data ingestion.

Responsibilities:
- Extract raw data from Kaggle
- Apply lightweight, non-destructive transformations
- Load cleaned data into MongoDB
- Generate metadata for auditability and observability

NOTE:
This ETL layer intentionally avoids feature engineering,
label processing, and modeling logic.
"""

import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file
from src.entity.config_entity import ETLconfig, TrainingPipelineConfig
from src.entity.artifact_entity import ETLArtifact


load_dotenv()


class CustomerChurnETL:
    """
    Production-grade ETL orchestrator.

    Pipeline:
        Extract → Transform → Load → Metadata

    Design goals:
    - Deterministic
    - Observable
    - Fail-fast
    - Maintainable
    """

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self, etl_config: ETLconfig, delete_old_data: bool = True) -> None:
        try:
            self.config = etl_config
            self.delete_old_data = delete_old_data

            self.mongodb_url: str | None = os.getenv("MONGODB_URL")
            self.database_name: str | None = os.getenv("MONGODB_DATABASE")
            self.collection_name: str | None = os.getenv("MONGODB_COLLECTION")
            self.data_source: str | None = os.getenv("DATA_SOURCE")

            self._validate_env_variables()

            self.ca_file = certifi.where()

            self.raw_data_dir = Path(self.config.raw_data_dir)
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)

            # Single reusable Mongo client (connection pooling)
            self.client = pymongo.MongoClient(
                self.mongodb_url,
                tlsCAFile=self.ca_file,
                serverSelectionTimeoutMS=5000,
            )

            logging.info("[ETL INIT] CustomerChurnETL initialized successfully.")

        except Exception as e:
            logging.exception("[ETL INIT] Initialization failed.")
            raise CustomerChurnException(e, sys)

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_env_variables(self) -> None:
        required = {
            "MONGODB_URL": self.mongodb_url,
            "MONGODB_DATABASE": self.database_name,
            "MONGODB_COLLECTION": self.collection_name,
            "DATA_SOURCE": self.data_source,
        }

        missing = [k for k, v in required.items() if not v]

        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")

    # =========================================================
    # EXTRACT
    # =========================================================

    def extract_data(self) -> pd.DataFrame:
        """
        Download dataset from Kaggle and load the first CSV file found.
        """
        try:
            logging.info(
                "[ETL EXTRACT] Starting Kaggle download | source=%s",
                self.data_source,
            )

            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.data_source,
                    "-p",
                    str(self.raw_data_dir),
                    "--unzip",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            logging.debug("[ETL EXTRACT] Kaggle CLI output captured.")

            csv_files = sorted(self.raw_data_dir.glob("*.csv"))

            if not csv_files:
                raise FileNotFoundError("No CSV files found after download.")

            df = pd.read_csv(csv_files[0])

            if df.empty:
                raise ValueError("Extracted CSV is empty.")

            logging.info(
                "[ETL EXTRACT] Completed | rows=%d cols=%d",
                df.shape[0],
                df.shape[1],
            )

            return df

        except Exception as e:
            logging.exception("[ETL EXTRACT] Extraction failed.")
            raise CustomerChurnException(e, sys)

    # =========================================================
    # TRANSFORM
    # =========================================================

    def transform_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform lightweight, non-destructive cleaning.
        """
        try:
            logging.info("[ETL TRANSFORM] Starting transformation.")

            df_clean = df.copy()
            initial_rows = len(df_clean)

            for col in df_clean.select_dtypes(include="object"):
                df_clean[col] = df_clean[col].str.strip()

            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

            df_clean["data_source"] = self.data_source
            df_clean["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

            records = df_clean.to_dict(orient="records")

            logging.info(
                "[ETL TRANSFORM] Completed | before=%d after=%d",
                initial_rows,
                len(records),
            )

            return records

        except Exception as e:
            logging.exception("[ETL TRANSFORM] Transformation failed.")
            raise CustomerChurnException(e, sys)

    # =========================================================
    # LOAD
    # =========================================================

    def load_data(self, records: List[Dict[str, Any]]) -> int:
        """
        Insert transformed records into MongoDB.
        """
        if not records:
            logging.warning("[ETL LOAD] No records to insert. Skipping load.")
            return 0

        try:
            logging.info(
                "[ETL LOAD] Loading into MongoDB | db=%s collection=%s",
                self.database_name,
                self.collection_name,
            )

            collection = self.client[self.database_name][self.collection_name]

            if self.delete_old_data:
                collection.delete_many({})

            result = collection.insert_many(records, ordered=True)

            inserted = len(result.inserted_ids)

            logging.info("[ETL LOAD] Completed | inserted=%d", inserted)

            return inserted

        except Exception as e:
            logging.exception("[ETL LOAD] MongoDB insertion failed.")
            raise CustomerChurnException(e, sys)

    # =========================================================
    # METADATA
    # =========================================================

    def generate_metadata(self, raw_df: pd.DataFrame, records_inserted: int) -> None:
        """
        Generate ETL metadata for observability and auditing.
        """
        try:
            logging.info("[ETL METADATA] Generating metadata.")

            metadata = {
                "data_source": self.data_source,
                "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": {
                    "rows_raw": raw_df.shape[0],
                    "columns": raw_df.shape[1],
                    "column_names": list(raw_df.columns),
                    "dtypes": raw_df.dtypes.astype(str).to_dict(),
                },
                "data_quality": {
                    "duplicate_rows_removed": int(raw_df.duplicated().sum()),
                    "missing_values_per_column": raw_df.isnull().sum().to_dict(),
                },
                "load_target": {
                    "database": self.database_name,
                    "collection": self.collection_name,
                    "records_inserted": records_inserted,
                },
            }

            write_json_file(self.config.metadata_file_path, metadata)

            logging.info(
                "[ETL METADATA] Written successfully | path=%s",
                self.config.metadata_file_path,
            )

        except Exception as e:
            logging.exception("[ETL METADATA] Metadata generation failed.")
            raise CustomerChurnException(e, sys)

    # =========================================================
    # ORCHESTRATION
    # =========================================================

    def initiate_etl(self) -> ETLArtifact:
        """
        Execute the complete ETL pipeline.
        """
        try:
            logging.info("[ETL PIPELINE] Execution started.")

            raw_df = self.extract_data()
            records = self.transform_data(raw_df)
            inserted = self.load_data(records)

            self.generate_metadata(raw_df, inserted)

            artifact = ETLArtifact(
                raw_data_dir_path=self.config.raw_data_dir,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("[ETL PIPELINE] Execution completed successfully.")
            logging.info("[ETL PIPELINE] Artifact created.")
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[ETL PIPELINE] Pipeline failed.")
            raise CustomerChurnException(e, sys)