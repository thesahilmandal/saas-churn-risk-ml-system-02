"""
Data Validation Pipeline.

Responsibilities:
- Validate generated dataset schema against a reference schema
- Apply severity-aware validation rules
- Persist validation report and status artifact
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List

import yaml

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file


class DataValidation:
    """
    Validates generated dataset schema against a reference schema.

    Acts as a strict gatekeeper before downstream ML processing.
    """

    def __init__(
        self,
        config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        try:
            logging.info("[DATA VALIDATION INIT] Initializing")

            self.config = config
            self.ingestion_artifact = ingestion_artifact

            if not os.path.exists(self.config.reference_schema_file_path):
                raise FileNotFoundError(
                    f"Reference schema not found at: "
                    f"{self.config.reference_schema_file_path}"
                )

            logging.info("[DATA VALIDATION INIT] Initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _read_yaml(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.exception(
                f"[DATA VALIDATION] Failed to read YAML: {file_path}"
            )
            raise CustomerChurnException(e, sys)

    @staticmethod
    def _escalate_status(current: str, severity: str) -> str:
        if severity == "error":
            return "error"
        if current != "error":
            return "warning"
        return current

    # ============================================================
    # Dataset-Level Validation
    # ============================================================
    def _validate_dataset_constraints(
        self,
        generated_schema: Dict[str, Any],
        reference_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        results = {"status": "pass", "details": []}
        constraints = reference_schema.get("dataset_constraints", {})

        feature_count = len(generated_schema) - 1  # exclude target

        if "expected_feature_count" in constraints:
            if feature_count != constraints["expected_feature_count"]:
                results["status"] = "warning"
                results["details"].append(
                    f"Expected {constraints['expected_feature_count']} features, "
                    f"found {feature_count}"
                )

        return results

    # ============================================================
    # Column-Level Validation
    # ============================================================
    def _validate_column(
        self,
        column: str,
        rules: Dict[str, Any],
        gen_col: Dict[str, Any],
        dtype_mapping: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        result = {"status": "pass", "details": []}
        severity = rules.get("severity", "error")

        # ---------- Required ----------
        if rules.get("required", False) and gen_col is None:
            result["status"] = "error"
            result["details"].append("Missing required column")
            return result

        # ---------- Dtype ----------
        expected_dtype = rules.get("expected_dtype")
        if expected_dtype:
            allowed_raw = dtype_mapping.get(expected_dtype, [])
            if gen_col["dtype"] not in allowed_raw:
                result["status"] = self._escalate_status(
                    result["status"], severity
                )
                result["details"].append(
                    f"Invalid dtype '{gen_col['dtype']}', "
                    f"expected semantic type '{expected_dtype}'"
                )

        # ---------- Nullability ----------
        if rules.get("nullable") is False and gen_col.get("nullable") is True:
            result["status"] = self._escalate_status(
                result["status"], severity
            )
            result["details"].append(
                "Null values present in non-nullable column"
            )

        # ---------- Numeric Range ----------
        if "min" in rules and "min" in gen_col:
            if gen_col["min"] < rules["min"]:
                result["status"] = "warning"
                result["details"].append(
                    f"Observed min {gen_col['min']} < expected {rules['min']}"
                )

        if "max" in rules and "max" in gen_col:
            if gen_col["max"] > rules["max"]:
                result["status"] = "warning"
                result["details"].append(
                    f"Observed max {gen_col['max']} > expected {rules['max']}"
                )

        # ---------- Allowed Values ----------
        if "allowed_values" in rules:
            result["details"].append(
                "Allowed values declared; enforcement deferred to data-level validation"
            )

        return result

    # ============================================================
    # Schema Validation
    # ============================================================
    def _validate_schema(
        self,
        generated_schema: Dict[str, Any],
        reference_schema: Dict[str, Any],
    ) -> Dict[str, Any]:

        logging.info("[DATA VALIDATION] Schema validation started")

        ref_columns = reference_schema["columns"]
        dtype_mapping = reference_schema.get("dtype_mapping", {})

        results: Dict[str, Any] = {}
        error_count = 0
        warning_count = 0

        # ---------- Dataset-level ----------
        dataset_result = self._validate_dataset_constraints(
            generated_schema, reference_schema
        )
        results["_dataset"] = dataset_result

        # ---------- Column-level ----------
        for column, rules in ref_columns.items():
            gen_col = generated_schema.get(column)
            col_result = self._validate_column(
                column, rules, gen_col, dtype_mapping
            )

            if col_result["status"] == "error":
                error_count += 1
            elif col_result["status"] == "warning":
                warning_count += 1

            results[column] = col_result

        # ---------- Unexpected columns ----------
        for column in generated_schema:
            if column not in ref_columns:
                results[column] = {
                    "status": "warning",
                    "details": ["Unexpected column not defined in reference schema"],
                }
                warning_count += 1

        logging.info(
            "[DATA VALIDATION] Completed | "
            f"errors={error_count}, warnings={warning_count}"
        )

        return {
            "column_checks": results,
            "summary": {
                "errors": error_count,
                "warnings": warning_count,
                "passed": error_count == 0,
            },
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("[DATA VALIDATION PIPELINE] Started")

            generated_schema = read_json_file(
                self.ingestion_artifact.schema_file_path
            )

            reference_schema = self._read_yaml(
                self.config.reference_schema_file_path
            )

            validation = self._validate_schema(
                generated_schema, reference_schema
            )

            report = {
                "validation_type": "schema_validation",
                "summary": validation["summary"],
                "column_results": validation["column_checks"],
                "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(
                self.config.validation_report_file_path, report
            )

            artifact = DataValidationArtifact(
                validation_status=validation["summary"]["passed"],
                validation_report=self.config.validation_report_file_path,
            )

            logging.info(
                "[DATA VALIDATION PIPELINE] Completed | "
                f"passed={artifact.validation_status}"
            )
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[DATA VALIDATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)