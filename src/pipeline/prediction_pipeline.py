import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    load_object,
    read_json_file,
    read_yaml_file,
)
from src.constants.pipeline_constants import (
    PRODUCTION_MODEL_FILE_PATH,
    MODEL_REGISTRY_METADATA_PATH,
    REFERENCE_SCHEMA,
)

load_dotenv()


class CustomerChurnPredictor:
    """
    Production-oriented prediction service for Customer Churn.

    Responsibilities:
        1. Load the latest approved production model from the registry.
        2. Validate incoming inference data against the reference schema.
        3. Generate churn probabilities using the loaded model.
        4. Log input features, predictions, and metadata for monitoring.

    Attributes:
        model (sklearn.pipeline.Pipeline): The loaded production model pipeline.
        registry_metadata (dict): Metadata about the currently deployed model version.
        mongo_client (pymongo.MongoClient): Connection to the observability database.
    """

    # Columns that identify a record but are not used for prediction
    IDENTIFIER_COLUMNS = ["customerID"]

    def __init__(self) -> None:
        """
        Initialize the predictor by loading the model and connecting to the database.

        Raises:
            CustomerChurnException: If model loading or DB connection fails.
        """
        try:
            self.model_path = PRODUCTION_MODEL_FILE_PATH
            self.registry_path = MODEL_REGISTRY_METADATA_PATH
            self.reference_schema = read_yaml_file(REFERENCE_SCHEMA)

            self._load_production_model()
            self._connect_to_database()

        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def _load_production_model(self) -> None:
        """Load the model artifact and registry metadata."""
        logging.info("[PREDICTOR INIT] Loading production model...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Production model not found at {self.model_path}")

        self.model = load_object(self.model_path)
        
        # Load registry metadata if available, else use defaults
        if os.path.exists(self.registry_path):
            self.registry_metadata = read_json_file(self.registry_path)
        else:
            logging.warning("[PREDICTOR INIT] Registry metadata not found. Using default.")
            self.registry_metadata = {"current_production_version": "unknown"}

        logging.info(
            f"[PREDICTOR INIT] Model loaded successfully | "
            f"version={self.registry_metadata.get('current_production_version')}"
        )

    def _connect_to_database(self) -> None:
        """Establish connection to MongoDB for prediction logging."""
        self.db_url = os.getenv("MONGODB_URL")
        self.database_name = os.getenv("ONLINE_DATABASE", "churn_monitoring_db")
        self.collection_name = os.getenv("ONLINE_COLLECTION", "predictions")

        if not self.db_url:
            logging.warning("[PREDICTOR INIT] MONGODB_URL not set. Logging will be disabled.")
            self.mongo_client = None
            return

        self.mongo_client = pymongo.MongoClient(
            self.db_url,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
        )

    # ------------------------------------------------------------------
    # Validation Logic
    # ------------------------------------------------------------------

    def _validate_input_schema(self, input_df: pd.DataFrame) -> None:
        """
        Validate inference input against the reference schema.
        
        Checks:
        - Required columns presence.
        - Forbidden columns (e.g., target).
        - Data types (numeric vs object).
        - Null values in non-nullable columns.

        Args:
            input_df (pd.DataFrame): The inference payload.

        Raises:
            ValueError: If schema validation fails.
            TypeError: If input is not a DataFrame.
        """
        try:
            if not isinstance(input_df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            if input_df.empty:
                raise ValueError("Input DataFrame is empty")

            # Load schema definition
            schema = self.reference_schema
            columns_schema = schema.get("columns", {})
            target_column = schema.get("dataset", {}).get("target_column")

            # 1. Check for Forbidden Columns
            if target_column and target_column in input_df.columns:
                raise ValueError(
                    f"Target column '{target_column}' is not allowed during inference."
                )

            # 2. Check for Missing Required Columns
            input_cols = set(input_df.columns)
            identifier_cols = set(self.IDENTIFIER_COLUMNS)
            feature_cols = input_cols - identifier_cols

            required_cols = {
                col for col, meta in columns_schema.items()
                if meta.get("required", False)
            }
            
            missing = required_cols - feature_cols
            if missing:
                raise ValueError(f"Missing required columns: {list(missing)}")

            # 3. Column-Level Checks
            for col_name, meta in columns_schema.items():
                if col_name not in input_df.columns:
                    continue

                series = input_df[col_name]

                # Null check
                if not meta.get("nullable", True) and series.isna().any():
                    raise ValueError(f"Column '{col_name}' contains null values.")

                # Dtype check (Robust)
                expected_type = meta.get("expected_dtype")
                if expected_type == "integer" or expected_type == "floating":
                    if not pd.api.types.is_numeric_dtype(series):
                         # Attempt coercion for robust handling
                        try:
                            pd.to_numeric(series)
                        except (ValueError, TypeError):
                            raise TypeError(f"Column '{col_name}' must be numeric.")
                
                elif expected_type == "string":
                     if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                          raise TypeError(f"Column '{col_name}' must be string/object.")

                # Allowed Values check
                if "allowed_values" in meta:
                    valid_values = set(meta["allowed_values"])
                    unique_inputs = set(series.dropna().unique())
                    invalid = unique_inputs - valid_values
                    if invalid:
                        raise ValueError(
                            f"Invalid values in '{col_name}': {list(invalid)}. "
                            f"Allowed: {list(valid_values)}"
                        )

        except Exception as e:
            # Re-raise as ValueError for clean API error handling
            raise ValueError(f"Schema Validation Failed: {str(e)}") from e

    # ------------------------------------------------------------------
    # Prediction Logic
    # ------------------------------------------------------------------

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn predictions for the input data.

        Args:
            input_df (pd.DataFrame): The raw customer data.

        Returns:
            pd.DataFrame: The input data enriched with 'churn_probability', 
                          'timestamp_utc', and 'model_version'.
        """
        try:
            logging.info("[PREDICTION] Received prediction request")

            # 1. Validate Input
            self._validate_input_schema(input_df)

            # 2. Pre-process (Drop non-features)
            # We explicitly drop identifiers so the model receives exactly what it expects.
            features_df = input_df.drop(
                columns=self.IDENTIFIER_COLUMNS, 
                errors="ignore"
            )

            # 3. Generate Prediction
            # Note: The loaded pipeline handles scaling/encoding internally.
            churn_probs = self.model.predict_proba(features_df)[:, 1]

            # 4. Format Output
            output_df = input_df.copy()
            output_df["churn_probability"] = churn_probs.round(4)
            output_df["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            output_df["model_version"] = self.registry_metadata.get(
                "current_production_version", "unknown"
            )

            # 5. Async/Non-blocking Logging
            self._log_prediction_event(output_df)

            logging.info("[PREDICTION] Success")
            return output_df

        except Exception as e:
            logging.exception("[PREDICTION] Inference failed")
            raise CustomerChurnException(e, sys) from e

    def _log_prediction_event(self, output_df: pd.DataFrame) -> None:
        """
        Log the prediction result to MongoDB for monitoring.
        This method is fail-safe; database errors will NOT crash the application.
        """
        if self.mongo_client is None:
            return

        try:
            records = output_df.to_dict(orient="records")
            collection = self.mongo_client[self.database_name][self.collection_name]
            collection.insert_many(records, ordered=False)
        except Exception as e:
            # We log the error but do NOT raise it. 
            # Observability failure should not break the user experience.
            logging.error(f"[PREDICTION LOGGING] Failed to log to MongoDB: {str(e)}")


if __name__ == "__main__":
    # Smoke Test
    try:
        # Create a dummy dataframe matching the schema structure
        data = {
            "customerID": ["Test-1"],
            "gender": ["Male"],
            "SeniorCitizen": ["No"],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.5],
            "TotalCharges": [840.5]
        }
        test_df = pd.DataFrame(data)
        
        predictor = CustomerChurnPredictor()
        result = predictor.predict(test_df)
        print("Prediction Result:")
        print(result[["customerID", "churn_probability", "model_version"]])
        
    except Exception as e:
        print(f"Smoke Test Failed: {e}")