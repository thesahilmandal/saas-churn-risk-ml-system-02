"""
Model Evaluation Pipeline.

Responsibilities:
- Evaluate candidate models on unseen test data
- Select best candidate using deterministic metric rules
- Evaluate currently deployed (champion) model on SAME test dataset
- Perform fair champion–challenger comparison
- Produce approval decision for Model Registry pipeline
- Persist evaluation report and metadata for lineage and auditability

Design Goals:
- Deterministic evaluation
- Fair comparison using identical evaluation data
- Clear separation from training and deployment
- Reproducible decision logic
- Minimal but production-aligned structure

IMPORTANT:
This pipeline DOES NOT perform model training or model registration.
It only evaluates and produces an approval decision.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import pandas as pd

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.constants.pipeline_constants import TARGET_COLUMN
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    load_object,
    write_json_file,
    read_json_file,
)


class ModelEvaluation:
    """
    Production-grade model evaluation pipeline.

    Pipeline Responsibilities:
        1. Load test dataset
        2. Evaluate all candidate models on test data
        3. Select best candidate using deterministic logic
        4. Evaluate champion model on SAME test dataset
        5. Perform champion–challenger comparison
        6. Produce approval decision
        7. Persist evaluation artifacts

    Guarantees:
        - No data leakage
        - Deterministic decisions
        - Fair model comparison
        - Reproducible evaluation
        - Clear audit trail
    """

    PIPELINE_VERSION = "1.0.0"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        config: ModelEvaluationConfig,
        ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        try:
            logging.info("[MODEL EVALUATION INIT] Initializing")

            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

            os.makedirs(self.config.evaluation_dir, exist_ok=True)

            logging.info(
                "[MODEL EVALUATION INIT] Initialized | "
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

    def _load_candidate_models(self) -> Dict[str, str]:
        """
        Discover trained candidate models.

        Expected structure:
            trained_models_dir/
                model_name/
                    model.pkl
        """
        models = {}

        for model_name in os.listdir(self.model_trainer_artifact.trained_models_dir):
            model_dir = os.path.join(
                self.model_trainer_artifact.trained_models_dir,
                model_name,
            )

            model_path = os.path.join(model_dir, "model.pkl")

            if os.path.exists(model_path):
                models[model_name] = model_path

        if not models:
            raise ValueError("No candidate models found for evaluation")

        logging.info(
            f"[MODEL EVALUATION] Discovered {len(models)} candidate models"
        )

        return models

    # ============================================================
    # METRIC COMPUTATION
    # ============================================================

    def _compute_metrics(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a model.
        """

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.config.decision_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "recall": round(recall_score(y_test, y_pred), 6),
            "precision": round(
                precision_score(y_test, y_pred, zero_division=0), 6
            ),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 6),
            "pr_auc": round(average_precision_score(y_test, y_prob), 6),
            "log_loss": round(log_loss(y_test, y_prob), 6),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        return metrics

    # ============================================================
    # CANDIDATE SELECTION
    # ============================================================

    def _select_best_candidate(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select best model based on:
            1. Highest recall
            2. Precision as tie-breaker within tolerance
        """

        tolerance = self.config.recall_tolerance

        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"]["recall"],
            reverse=True,
        )

        best_name, best_result = sorted_models[0]

        for name, result in sorted_models[1:]:
            recall_diff = (
                best_result["metrics"]["recall"]
                - result["metrics"]["recall"]
            )

            if abs(recall_diff) <= tolerance:
                if (
                    result["metrics"]["precision"]
                    > best_result["metrics"]["precision"]
                ):
                    best_name, best_result = name, result

        logging.info(
            f"[MODEL EVALUATION] Best candidate selected: {best_name}"
        )

        return best_name, best_result

    # ============================================================
    # CHAMPION–CHALLENGER COMPARISON
    # ============================================================

    def _evaluate_champion_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate currently deployed model on the SAME test dataset.
        """

        champion_model_path = self.config.production_model_file_path

        if not os.path.exists(champion_model_path):
            logging.info(
                "[MODEL EVALUATION] No production model found."
            )
            return None

        logging.info("[MODEL EVALUATION] Evaluating champion model")

        champion_model = load_object(champion_model_path)

        champion_metrics = self._compute_metrics(
            champion_model, X_test, y_test
        )

        return champion_metrics

    def _compare_with_champion(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Compare challenger against champion using recall improvement rule.
        """

        if champion_metrics is None:
            return True, "No production model available"

        candidate_recall = candidate_metrics["recall"]
        champion_recall = champion_metrics["recall"]

        improvement = candidate_recall - champion_recall

        if improvement >= self.config.min_recall_improvement:
            return True, f"Recall improved by {round(improvement, 6)}"

        return False, "Recall improvement below threshold"

    # ============================================================
    # ENTRY POINT
    # ============================================================

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")

            started_at_utc = datetime.now(timezone.utc).isoformat()

            # ---------- Load Test Data ----------
            test_df = self._read_csv(
                self.ingestion_artifact.test_file_path
            )

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # ---------- Load Candidate Models ----------
            candidate_models = self._load_candidate_models()

            evaluation_results: Dict[str, Dict[str, Any]] = {}

            # ---------- Evaluate Candidates ----------
            for model_name, model_path in candidate_models.items():
                logging.info(
                    f"[MODEL EVALUATION] Evaluating model={model_name}"
                )

                model = load_object(model_path)

                metrics = self._compute_metrics(model, X_test, y_test)

                evaluation_results[model_name] = {
                    "model_path": model_path,
                    "metrics": metrics,
                }

            # ---------- Select Best Candidate ----------
            best_model_name, best_result = self._select_best_candidate(
                evaluation_results
            )

            # ---------- Evaluate Champion ----------
            champion_metrics = self._evaluate_champion_model(
                X_test, y_test
            )

            # ---------- Champion–Challenger Comparison ----------
            approved, reason = self._compare_with_champion(
                best_result["metrics"], champion_metrics
            )

            completed_at_utc = datetime.now(timezone.utc).isoformat()

            # ---------- Report ----------
            report = {
                "pipeline": {
                    "name": "model_evaluation",
                    "version": self.PIPELINE_VERSION,
                },
                "timing": {
                    "started_at_utc": started_at_utc,
                    "completed_at_utc": completed_at_utc,
                },
                "best_model": best_model_name,
                "approved": approved,
                "reason": reason,
                "candidate_results": evaluation_results,
                "champion_metrics": champion_metrics,
            }

            write_json_file(self.config.evaluation_report_file_path, report)

            # ---------- Metadata ----------
            best_model_size_bytes = os.path.getsize(best_result["model_path"])
            best_model_size_mb = round(best_model_size_bytes / (1024 * 1024), 4)

            metadata = {
                "pipeline_version": self.PIPELINE_VERSION,
                "decision_threshold": self.config.decision_threshold,
                "recall_tolerance": self.config.recall_tolerance,
                "min_recall_improvement": self.config.min_recall_improvement,
                "best_model": best_model_name,
                "approved": approved,
                "model_size_mb": best_model_size_mb,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(self.config.metadata_file_path, metadata)

            artifact = ModelEvaluationArtifact(
                best_model_name=best_model_name,
                best_model_path=best_result["model_path"],
                evaluation_report_path=self.config.evaluation_report_file_path,
                metadata_path=self.config.metadata_file_path,
                approval_status=approved,
            )

            logging.info(
                "[MODEL EVALUATION PIPELINE] Completed | "
                f"approved={approved}"
            )
            logging.info(artifact)

            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
