"""
Model Monitoring Pipeline.

Responsibilities:
- Load monitoring baseline from transformation stage.
- Extract recent inference logs from MongoDB.
- Detect feature drift using PSI.
- Generate structured monitoring artifacts:
    - report.json
    - metadata.json
    - retraining_flag.json
- Return a ModelMonitoringArtifact object.

Design Principles:
- Deterministic drift detection
- Config-governed thresholds
- Strict artifact isolation
- Clean integration with existing pipeline
- Idempotent execution
- Minimal runtime overhead
"""

import os
import sys
import math
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

from src.entity.config_entity import ModelMonitoringConfig
from src.entity.artifact_entity import ModelMonitoringArtifact
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file
from src.constants.pipeline_constants import MONITORING_BASELINE_PATH

load_dotenv()


class ModelMonitoring:
    """
    Production-grade feature drift monitoring component.

    Evaluates feature distribution shift between:
        - Training baseline (monitoring_baseline.json)
        - Live inference data (MongoDB logs)

    Only feature drift is evaluated (no performance drift).
    """

    PIPELINE_VERSION = "1.0.0"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self, config: ModelMonitoringConfig) -> None:
        try:
            logging.info("[MODEL MONITORING INIT] Initializing")

            self.config = config

            if not os.path.exists(MONITORING_BASELINE_PATH):
                raise FileNotFoundError(
                    f"Monitoring baseline not found at {MONITORING_BASELINE_PATH}"
                )

            os.makedirs(
                self.config.artifact_dir,
                exist_ok=True
            )

            logging.info(
                "[MODEL MONITORING INIT] Initialized | "
                f"psi_threshold={self.config.psi_threshold}, "
                f"drift_ratio_threshold={self.config.drifted_feature_ratio_threshold}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # BASELINE LOADING
    # ============================================================

    def _load_baseline(self) -> Dict[str, Any]:
        logging.info("[MODEL MONITORING] Loading monitoring baseline")
        return read_json_file(MONITORING_BASELINE_PATH)

    # ============================================================
    # INFERENCE DATA EXTRACTION
    # ============================================================

    def _load_inference_data(self) -> pd.DataFrame:
        """
        Extract recent inference logs from MongoDB.
        """

        logging.info("[MODEL MONITORING] Fetching inference logs")

        db_url = os.getenv("MONGODB_URL")
        database_name = os.getenv("ONLINE_DATABASE", "churn_monitoring_db")
        collection_name = os.getenv("ONLINE_COLLECTION", "predictions")

        if not db_url:
            raise EnvironmentError("MONGODB_URL not set. Monitoring cannot proceed.")

        with pymongo.MongoClient(
            db_url,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
        ) as client:

            collection = client[database_name][collection_name]

            cursor = (
                collection.find({}, {"_id": 0})
                .sort("timestamp_utc", pymongo.DESCENDING)
                .limit(self.config.monitoring_sample_size)
            )

            records = list(cursor)

        if not records:
            if self.config.strict_mode:
                raise ValueError("No inference logs available for monitoring")

            logging.warning("[MODEL MONITORING] No inference logs found")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        drop_cols = [
            "churn_probability",
            "timestamp_utc",
            "model_version",
            "_id",
            "customerID",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        logging.info(
            "[MODEL MONITORING] Inference data loaded | "
            f"rows={len(df)}"
        )

        return df

    # ============================================================
    # PSI CALCULATION
    # ============================================================

    @staticmethod
    def _calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) for numeric features.
        """

        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        if np.all(expected == expected[0]):
            return 0.0

        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) <= 2:
            return 0.0

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_ratio = expected_counts / len(expected)
        actual_ratio = actual_counts / len(actual)

        psi_values = []

        for e, a in zip(expected_ratio, actual_ratio):
            e = max(e, 1e-6)
            a = max(a, 1e-6)
            psi_values.append((a - e) * math.log(a / e))

        return round(float(sum(psi_values)), 6)

    # ============================================================
    # DRIFT DETECTION
    # ============================================================

    def _detect_feature_drift(
        self,
        baseline: Dict[str, Any],
        live_df: pd.DataFrame,
    ) -> Dict[str, Any]:

        logging.info("[MODEL MONITORING] Detecting feature drift")

        drift_report: Dict[str, Any] = {}
        drifted_features = []

        if live_df.empty:
            logging.warning(
                "[MODEL MONITORING] Live dataframe empty — skipping drift detection"
            )

        for feature, meta in baseline.get("features", {}).items():

            if feature not in live_df.columns:
                continue

            live_values = live_df[feature].dropna()
            if live_values.empty:
                continue

            baseline_dist = np.array(meta["distribution"])

            if meta["type"] == "numeric":
                psi = self._calculate_psi(
                    baseline_dist,
                    live_values.to_numpy(),
                )
            else:
                baseline_probs = np.array(baseline_dist)

                categories = meta.get("categories", [])
                live_probs = (
                    live_values.value_counts(normalize=True)
                    .reindex(categories)
                    .fillna(0)
                    .values
                )

                if len(baseline_probs) != len(live_probs):
                    logging.warning(
                        f"[MODEL MONITORING] Category length mismatch for feature: {feature}"
                    )
                    continue

                psi = sum(
                    (a - e) * math.log((a + 1e-6) / (e + 1e-6))
                    for e, a in zip(baseline_probs, live_probs)
                )

                psi = round(float(psi), 6)

            is_drifted = psi >= self.config.psi_threshold

            drift_report[feature] = {
                "psi": psi,
                "drifted": is_drifted,
            }

            if is_drifted:
                drifted_features.append(feature)

        total_features = len(drift_report)
        drift_ratio = (
            len(drifted_features) / total_features
            if total_features > 0
            else 0.0
        )

        return {
            "features": drift_report,
            "drifted_features": drifted_features,
            "drift_ratio": round(float(drift_ratio), 4),
        }


    # ============================================================
    # ENTRY POINT
    # ============================================================

    def initiate_model_monitoring(self) -> ModelMonitoringArtifact:
        try:
            logging.info("[MODEL MONITORING PIPELINE] Started")

            baseline = self._load_baseline()
            live_df = self._load_inference_data()

            drift_results = self._detect_feature_drift(
                baseline,
                live_df,
            )

            retraining_required = (
                drift_results["drift_ratio"]
                >= self.config.drifted_feature_ratio_threshold
            )

            artifact_dir = self.config.artifact_dir

            report_path = os.path.join(artifact_dir, "report.json")
            metadata_path = os.path.join(artifact_dir, "metadata.json")
            flag_path = os.path.join(artifact_dir, "retraining_flag.json")

            write_json_file(report_path, drift_results)

            metadata = {
                "pipeline_version": self.PIPELINE_VERSION,
                "psi_threshold": self.config.psi_threshold,
                "drift_ratio_threshold": self.config.drifted_feature_ratio_threshold,
                "monitoring_sample_size": self.config.monitoring_sample_size,
                "retraining_required": retraining_required,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(metadata_path, metadata)
            write_json_file(flag_path, {"retraining_required": retraining_required})

            artifact = ModelMonitoringArtifact(
                artifact_dir=str(artifact_dir),
                report_file_path=report_path,
                metadata_file_path=metadata_path,
                retraining_flag_file_path=flag_path,
                retraining_required=retraining_required,
            )

            logging.info(
                "[MODEL MONITORING PIPELINE] Completed | "
                f"retraining_required={artifact.retraining_required}"
            )
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[MODEL MONITORING PIPELINE] Failed")
            raise CustomerChurnException(e, sys)


if __name__ == "__main__":
    try:
        config = ModelMonitoringConfig()
        monitor = ModelMonitoring(config)
        artifact = monitor.initiate_model_monitoring()
        print(artifact)
    except Exception as e:
        raise CustomerChurnException(e, sys)