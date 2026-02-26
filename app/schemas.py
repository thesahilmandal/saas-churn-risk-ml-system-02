from typing import Literal
from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    """
    Pydantic model representing the input data contract for the Churn Prediction API.

    Design Note:
    - Fields use `Literal` to enforce exact allowed values (e.g., "Yes", "No") at the API gateway level.
    - This ensures invalid categorical values are rejected immediately with a 422 error
      before reaching the model pipeline.
    - Matches the `REFERENCE_SCHEMA` strictly to prevent downstream type errors.
    """

    gender: Literal["Male", "Female"] = Field(
        ..., 
        description="Customer gender"
    )

    SeniorCitizen: Literal["Yes", "No"] = Field(
        ..., 
        description="Is the customer a senior citizen? (Matches training data format)"
    )

    Partner: Literal["Yes", "No"] = Field(
        ..., 
        description="Does the customer have a partner?"
    )

    Dependents: Literal["Yes", "No"] = Field(
        ..., 
        description="Does the customer have dependents?"
    )

    tenure: int = Field(
        ..., 
        ge=0, 
        description="Number of months the customer has stayed with the company"
    )

    PhoneService: Literal["Yes", "No"] = Field(
        ..., 
        description="Does the customer have phone service?"
    )

    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., 
        description="Does the customer have multiple lines?"
    )

    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., 
        description="Customer's internet service provider type"
    )

    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer have online security?"
    )

    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer have online backup?"
    )

    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer have device protection?"
    )

    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer have tech support?"
    )

    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer stream TV?"
    )

    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., 
        description="Does the customer stream movies?"
    )

    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., 
        description="The contract term of the customer"
    )

    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., 
        description="Does the customer use paperless billing?"
    )

    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ] = Field(
        ..., 
        description=" The customer's payment method"
    )

    MonthlyCharges: float = Field(
        ..., 
        ge=0, 
        description="The amount charged to the customer monthly"
    )

    TotalCharges: float = Field(
        ..., 
        ge=0, 
        description="The total amount charged to the customer"
    )

    customerID: str = Field(
        ..., 
        description="Unique identifier for the customer (used for tracking, not prediction)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 840.5,
                "customerID": "12345-TEST"
            }
        }


class PredictionResponse(BaseModel):
    """
    Standardized response format for the Churn Prediction API.
    """
    customerID: str
    churn_probability: float
    risk_level: str
    model_version: str
    timestamp_utc: str