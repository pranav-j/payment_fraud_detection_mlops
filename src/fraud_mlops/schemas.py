"""Pydantic schemas for the inference API.

These define the JSON contract between clients and the API. FastAPI uses
them for automatic request validation, response serialization, and OpenAPI
documentation.

Keep these intentionally separate from `inference.Prediction`:
  - inference.Prediction is the internal Python result type
  - PredictionResponse is the external HTTP representation

That separation means we can change one without breaking the other. For
example, we could later add an internal `model_features_used` field to
Prediction without changing the public API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """One transaction submitted for fraud scoring.

    Field constraints:
      - type: one of the five PaySim transaction types
      - amount: positive
      - balances: non-negative
      - drains_origin: 0 or 1 (we accept int rather than bool for compatibility
        with the model's training-time dtype, which was int8)
    """

    model_config = ConfigDict(
        # Reject unexpected fields rather than silently dropping them. If a
        # client sends a typo'd field, fail loudly so the bug is caught early.
        extra="forbid",
        populate_by_name=True,
        # Example for the auto-generated OpenAPI docs at /docs
        json_schema_extra={
            "example": {
                "step": 100,
                "type": "TRANSFER",
                "amount": 5000.0,
                "oldbalanceOrg": 10000.0,
                "newbalanceOrig": 5000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 5000.0,
                "sender_txn_count_1h": 1,
                "sender_txn_count_24h": 5,
                "sender_amount_sum_24h": 20000.0,
                "sender_amount_mean_historical": 4000.0,
                "sender_time_since_last_txn": 2.5,
                "amount_to_oldbalance_ratio": 0.5,
                "drains_origin": 0,
            }
        },
    )

    step: int = Field(..., ge=0, description="Hours since simulation start")
    type: Literal["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"]
    amount: float = Field(..., gt=0)
    old_balance_org: float = Field(..., ge=0, alias="oldbalanceOrg")
    new_balance_orig: float = Field(..., ge=0, alias="newbalanceOrig")
    old_balance_dest: float = Field(..., ge=0, alias="oldbalanceDest")
    new_balance_dest: float = Field(..., ge=0, alias="newbalanceDest")

    # Feature-store-derived features. In week 6, Feast will fill these in.
    # For week 3, the caller supplies them.
    sender_txn_count_1h: int = Field(..., ge=0)
    sender_txn_count_24h: int = Field(..., ge=0)
    sender_amount_sum_24h: float = Field(..., ge=0)
    sender_amount_mean_historical: float = Field(..., ge=0)
    sender_time_since_last_txn: float = Field(
        ..., description="-1 sentinel if first-ever transaction"
    )
    amount_to_oldbalance_ratio: float = Field(..., description="-1 sentinel if oldbalanceOrg was 0")
    drains_origin: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """The fraud-check decision for one transaction."""

    is_fraud: bool
    probability: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    model_version: str
    feature_set: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    feature_set: str
