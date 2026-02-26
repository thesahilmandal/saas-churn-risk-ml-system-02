import sys
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi import UploadFile, File, Query
from fastapi.responses import StreamingResponse
import io

from app.schemas import CustomerData, PredictionResponse
from src.pipeline.prediction_pipeline import CustomerChurnPredictor
from src.exception import CustomerChurnException
from src.logging import logging

# ============================================================
# Lifecycle Management (The "Singleton" Pattern)
# ============================================================

ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy ML artifacts ONCE at startup, not per request.
    This prevents memory leaks and reduces latency.
    """
    try:
        logging.info("[API LIFECYCLE] Starting up...")
        ml_components["predictor"] = CustomerChurnPredictor()
        logging.info("[API LIFECYCLE] Model loaded successfully.")
        yield
    except Exception as e:
        logging.exception("[API LIFECYCLE] Failed to load model.")
        raise RuntimeError("Model failed to load") from e
    finally:
        logging.info("[API LIFECYCLE] Shutting down...")
        ml_components.clear()

# ============================================================
# API Initialization
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time inference API for Telco Customer Churn.",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers (e.g., AWS ELB, K8s).
    """
    if "predictor" not in ml_components:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "service": "churn-prediction"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: CustomerData):
    """
    Main inference endpoint.
    
    Flow:
    1. Receive JSON payload (validated by Pydantic).
    2. Convert to DataFrame.
    3. Run inference via PredictionPipeline.
    4. Return structured response.
    """
    try:
        # 1. Access the pre-loaded predictor
        predictor = ml_components["predictor"]

        # 2. Convert Pydantic object to DataFrame
        # We wrap it in a list to create a single-row DataFrame
        input_df = pd.DataFrame([request.model_dump()])

        # 3. Generate Prediction
        result_df = predictor.predict(input_df)

        # 4. Extract Results
        prediction_record = result_df.iloc[0]
        
        churn_prob = float(prediction_record["churn_probability"])
        
        # Simple business logic for risk level (can be moved to config)
        risk_level = "High" if churn_prob > 0.5 else "Low"

        return PredictionResponse(
            customerID=prediction_record["customerID"],
            churn_probability=churn_prob,
            risk_level=risk_level,
            model_version=prediction_record["model_version"],
            timestamp_utc=prediction_record["timestamp_utc"]
        )

    except ValueError as ve:
        # Schema validation errors from the pipeline
        logging.warning(f"[API ERROR] Validation failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        # Unexpected server errors
        logging.exception("[API ERROR] Inference failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    download: bool = Query(False, description="Set true to download results as CSV")
):
    """
    Batch inference endpoint for CSV file uploads.

    Flow:
    1. Accept CSV file.
    2. Load into pandas DataFrame.
    3. Run vectorized inference.
    4. Return JSON response or downloadable CSV.
    """

    try:
        predictor = ml_components["predictor"]

        # --------------------------------------------------
        # 1. Validate File Type
        # --------------------------------------------------
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # --------------------------------------------------
        # 2. Read CSV into DataFrame (In-Memory)
        # --------------------------------------------------
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if input_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

        # --------------------------------------------------
        # 3. Run Vectorized Prediction
        # --------------------------------------------------
        result_df = predictor.predict(input_df)

        # Add risk level column (vectorized)
        result_df["risk_level"] = result_df["churn_probability"].apply(
            lambda x: "High" if x > 0.5 else "Low"
        )

        # --------------------------------------------------
        # 4A. Return Downloadable CSV
        # --------------------------------------------------
        if download:
            output_buffer = io.StringIO()
            result_df.to_csv(output_buffer, index=False)
            output_buffer.seek(0)

            return StreamingResponse(
                iter([output_buffer.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=churn_predictions.csv"
                },
            )

        # --------------------------------------------------
        # 4B. Return JSON Response
        # --------------------------------------------------
        return JSONResponse(
            content={
                "total_records": len(result_df),
                "model_version": result_df.iloc[0]["model_version"],
                "predictions": result_df.to_dict(orient="records")
            }
        )

    except ValueError as ve:
        logging.warning(f"[BATCH API ERROR] Validation failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logging.exception("[BATCH API ERROR] Inference failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    # For local development
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)