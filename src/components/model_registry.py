"""
Model Registry Pipeline (Production-Grade with Rollback Support)

Responsibilities:
- Promote approved models from evaluation pipeline
- Maintain immutable versioned model storage
- Update production model pointer
- Maintain rich registry metadata for lineage, audit, and rollback
- Support safe rollback to previous versions

Design Guarantees:
- Only approved models are promoted
- Registered versions are immutable
- Production model always points to a valid registered version
- Full lineage preserved
- Audit-safe metadata tracking
- Rollback operations are fully recorded
"""

import os
import sys
import shutil
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.entity.config_entity import ModelRegistryConfig
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file


class ModelRegistry:
    """
    Production-oriented Model Registry.

    This component is responsible for safely promoting evaluated models
    into a versioned registry, updating production pointers, and maintaining
    metadata for lineage, auditing, and rollback.
    """

    PIPELINE_VERSION = "1.0.0"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        config: ModelRegistryConfig,
        evaluation_artifact: Optional[ModelEvaluationArtifact] = None,
    ) -> None:
        """
        Initialize ModelRegistry.

        Args:
            config: Model registry configuration object.
            evaluation_artifact: Evaluation artifact (required for promotion,
                                 optional for rollback).
        """
        try:
            logging.info("[MODEL REGISTRY INIT] Initializing")

            self.config = config
            self.evaluation_artifact = evaluation_artifact

            os.makedirs(self.config.registry_dir, exist_ok=True)
            os.makedirs(
                os.path.dirname(self.config.production_model_file_path),
                exist_ok=True,
            )

            self.registry_metadata_path = os.path.join(
                self.config.registry_dir,
                "registry_metadata.json",
            )

            logging.info(
                "[MODEL REGISTRY INIT] Completed | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # METADATA HELPERS
    # ============================================================

    def _load_registry_metadata(self) -> Dict[str, Any]:
        """
        Load existing registry metadata or initialize new structure.
        """
        if not os.path.exists(self.registry_metadata_path):
            return {
                "registry_version": "1.0",
                "current_production_version": None,
                "registered_versions": [],
                "versions_metadata": {},
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "rollback_history": [],
            }

        return read_json_file(self.registry_metadata_path)

    def _save_registry_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Persist registry metadata to disk.
        """
        write_json_file(self.registry_metadata_path, metadata)

    def _get_next_version(self, metadata: Dict[str, Any]) -> str:
        """
        Determine next semantic version (v1, v2, v3...).
        """
        versions = metadata.get("registered_versions", [])

        if not versions:
            return "v1"

        latest = max(int(v[1:]) for v in versions)
        return f"v{latest + 1}"

    # ============================================================
    # CHECKSUM
    # ============================================================

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """
        Compute SHA256 checksum of model file.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ============================================================
    # MODEL PROMOTION HELPERS
    # ============================================================

    def _register_model(self, version: str) -> str:
        """
        Copy approved model into versioned registry directory.

        Args:
            version: Version string (e.g., v2)

        Returns:
            Path to registered model file.
        """
        version_dir = os.path.join(self.config.registry_dir, version)

        if os.path.exists(version_dir):
            raise ValueError(f"Version directory already exists: {version}")

        os.makedirs(version_dir, exist_ok=False)

        destination_model_path = os.path.join(version_dir, "model.pkl")

        shutil.copy2(
            self.evaluation_artifact.best_model_path,
            destination_model_path,
        )

        logging.info(f"[MODEL REGISTRY] Model registered | version={version}")

        return destination_model_path

    def _update_production_model(self, version_model_path: str) -> None:
        """
        Update production model pointer.
        """
        shutil.copy2(
            version_model_path,
            self.config.production_model_file_path,
        )

        logging.info("[MODEL REGISTRY] Production model updated")

    def _update_registry_metadata(
        self,
        metadata: Dict[str, Any],
        version: str,
        version_model_path: str,
    ) -> None:
        """
        Update registry metadata with version-level lineage.
        """

        model_size_bytes = os.path.getsize(version_model_path)
        model_size_mb = round(model_size_bytes / (1024 * 1024), 4)
        checksum = self._compute_checksum(version_model_path)

        version_metadata = {
            "model_name": self.evaluation_artifact.best_model_name,
            "model_path": version_model_path,
            "evaluation_report_path":
                self.evaluation_artifact.evaluation_report_path,
            "evaluation_metadata_path":
                self.evaluation_artifact.metadata_path,
            "approval_status":
                self.evaluation_artifact.approval_status,
            "model_size_mb": model_size_mb,
            "checksum_sha256": checksum,
            "registered_at_utc":
                datetime.now(timezone.utc).isoformat(),
        }

        metadata.setdefault("registered_versions", []).append(version)
        metadata.setdefault("versions_metadata", {})[version] = version_metadata
        metadata["current_production_version"] = version
        metadata["last_updated_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        self._save_registry_metadata(metadata)

        logging.info("[MODEL REGISTRY] Registry metadata updated")

    # ============================================================
    # MODEL PROMOTION ENTRY POINT
    # ============================================================

    def initiate_model_registry(self) -> Optional[str]:
        """
        Execute model promotion pipeline.

        Returns:
            Version string if promotion successful, otherwise None.
        """
        try:
            logging.info("[MODEL REGISTRY PIPELINE] Started")

            if not self.evaluation_artifact:
                raise ValueError(
                    "Evaluation artifact required for model promotion."
                )

            # Approval Gate
            if not self.evaluation_artifact.approval_status:
                logging.info(
                    "[MODEL REGISTRY] Model not approved. Skipping."
                )
                return None

            metadata = self._load_registry_metadata()
            version = self._get_next_version(metadata)
            version_model_path = self._register_model(version)
            self._update_production_model(version_model_path)
            self._update_registry_metadata(
                metadata,
                version,
                version_model_path,
            )

            logging.info(
                "[MODEL REGISTRY PIPELINE] Completed | "
                f"production_version={version}"
            )

            return version

        except Exception as e:
            logging.exception("[MODEL REGISTRY PIPELINE] Failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # ROLLBACK FUNCTIONALITY
    # ============================================================

    def rollback_to_version(self, version: str) -> str:
        """
        Rollback production model to a specific registered version.

        Args:
            version: Version string (e.g., "v2")

        Returns:
            Rolled-back version string.
        """
        try:
            logging.info(
                f"[MODEL REGISTRY] Rollback requested | version={version}"
            )

            metadata = self._load_registry_metadata()

            if version not in metadata.get("registered_versions", []):
                raise ValueError(
                    f"Version '{version}' not found in registry."
                )

            version_dir = os.path.join(self.config.registry_dir, version)
            version_model_path = os.path.join(version_dir, "model.pkl")

            if not os.path.exists(version_model_path):
                raise FileNotFoundError(
                    f"Model file missing for version: {version}"
                )

            shutil.copy2(
                version_model_path,
                self.config.production_model_file_path,
            )

            metadata["current_production_version"] = version
            metadata["last_rollback_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )

            metadata.setdefault("rollback_history", []).append(
                {
                    "rolled_back_to": version,
                    "timestamp_utc":
                        datetime.now(timezone.utc).isoformat(),
                }
            )

            self._save_registry_metadata(metadata)

            logging.info(
                "[MODEL REGISTRY] Rollback successful | "
                f"production_version={version}"
            )

            return version

        except Exception as e:
            logging.exception("[MODEL REGISTRY] Rollback failed")
            raise CustomerChurnException(e, sys)