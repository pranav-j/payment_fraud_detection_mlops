"""FastAPI app for fraud detection inference.

Lifecycle:
  - At startup: load the production model from MLflow (singleton).
  - Per request: validate JSON, call inference, serialize result.

The model is loaded once at startup and held in app.state. This is the
standard pattern for stateful FastAPI services: heavy resources go in
state, request handlers are stateless.

To run locally:
    uv run uvicorn fraud_mlops.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from fraud_mlops.inference import FraudDetector, load_production_detector
from fraud_mlops.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks.

    On startup: load the production model and cache it on app.state.
    On shutdown: nothing to clean up (model is just Python objects).

    `asynccontextmanager` is FastAPI's modern lifespan pattern, replacing
    the older @app.on_event decorators.
    """
    logger.info("Loading production fraud detector from MLflow registry...")
    app.state.detector = load_production_detector()
    logger.info(
        "Loaded fraud-detector v%s (feature_set=%s)",
        app.state.detector.version,
        app.state.detector.feature_set,
    )

    yield  # Server is ready to accept requests

    logger.info("Shutting down.")


app = FastAPI(
    title="Fraud Detector API",
    description="Real-time fraud scoring service.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_detector(request: Request) -> FraudDetector:
    """Pull the detector singleton off app.state.

    Defined as a small helper rather than a global so tests can override
    it via FastAPI's dependency_overrides mechanism.
    """
    detector: FraudDetector | None = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting or in a bad state.",
        )
    return detector


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Always returns OK if the process is alive.

    Note: this is intentionally cheap and doesn't check the model. A
    Kubernetes liveness probe should fail only when the process is dead,
    not when downstream deps are flaky.
    """
    return HealthResponse()


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """What model is the service currently serving?

    Useful for debugging ("am I getting predictions from v2 or v3?") and
    for downstream systems that want to log the model version alongside
    decisions.
    """
    detector = get_detector(request)
    return ModelInfoResponse(
        model_version=detector.version,
        threshold=detector.threshold,
        feature_set=detector.feature_set,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, request: Request) -> PredictionResponse:
    """Score one transaction for fraud.

    Returns a PredictionResponse with the decision, the probability,
    and the model metadata. The metadata travels with every response
    so callers can log it for audit purposes.
    """
    detector = get_detector(request)
    result = detector.predict_one(transaction.model_dump(by_alias=True))
    return PredictionResponse(**result.to_dict())
