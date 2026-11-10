"""FastAPI churn-scoring service with two endpoints: GET /health and POST /predict.

Run from the repository root:
    uvicorn serving.app:app --port 8000

Environment variables:
  CHURN_MODEL_PATH            artifact to load (default: artifacts/churn_model.joblib)
  CHURN_API_SIMULATE_FAILURE  TEST-ONLY fault injection, unset (off) by default.
                              "model": the model layer fails -> heuristic fallback.
                              "all":   model and heuristic fail -> prior-rate fallback.

Status codes:
  200  scored; `source` says by which layer (model / fallback_heuristic / fallback_prior_rate)
  422  malformed payload (schema violations, unseen categories, impossible values)
  503  model artifact not loaded (missing file or failed integrity check)
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.schemas import HealthResponse, PredictRequest, PredictResponse
from serving.scoring import (SIMULATE_FAILURE_ENV, SIMULATE_FAILURE_VALUES, ModelLoadError, load_model,
                             prepare_features, score_frame)
from src.data_pipeline import FEATURES, REPO_ROOT

DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "churn_model.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the artifact a single time and keep it in memory."""
    path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        app.state.bundle, app.state.load_error = load_model(path), None
        logger.info("Model loaded from %s (threshold %.4f)", path, app.state.bundle["threshold"])
    except ModelLoadError as exc:
        app.state.bundle, app.state.load_error = None, str(exc)
        logger.error("Model NOT loaded from %s: %s", path, exc)

    switch = os.environ.get(SIMULATE_FAILURE_ENV) or None
    if switch is not None and switch not in SIMULATE_FAILURE_VALUES:
        logger.warning("Ignoring %s=%r (allowed: %s)", SIMULATE_FAILURE_ENV, switch, sorted(SIMULATE_FAILURE_VALUES))
        switch = None
    if switch is not None:
        logger.warning("TEST-ONLY %s=%s is active: scoring failures are being simulated", SIMULATE_FAILURE_ENV, switch)
    app.state.simulate_failure = switch
    yield


app = FastAPI(title="Customer churn scoring API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log every rejected payload, then return FastAPI's standard 422 body."""
    problems = [(".".join(str(p) for p in e["loc"]), e["msg"]) for e in exc.errors()]
    logger.warning("422 on %s: %s", request.url.path, problems)
    return await request_validation_exception_handler(request, exc)


@app.get("/health", response_model=HealthResponse)
def health():
    bundle = app.state.bundle
    if bundle is None:
        body = HealthResponse(status="unavailable", model_loaded=False,
                              simulated_failure=app.state.simulate_failure, detail=app.state.load_error)
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse(status="ok", model_loaded=True, threshold=bundle["threshold"],
                          fallback_probability=bundle["train_churn_rate"],
                          simulated_failure=app.state.simulate_failure)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    request_id = req.request_id or str(uuid.uuid4())
    bundle = app.state.bundle
    if bundle is None:
        logger.error("request_id=%s customer_id=%s rejected with 503: model not loaded", request_id, req.customer_id)
        raise HTTPException(status_code=503, detail=f"Model not loaded: {app.state.load_error}")

    frame, row_warnings = prepare_features([req.model_dump(include=set(FEATURES))], bundle)
    warnings = row_warnings[0]
    for message in warnings:
        logger.warning("request_id=%s customer_id=%s %s", request_id, req.customer_id, message)

    result = score_frame(frame, bundle, app.state.simulate_failure)
    if result.fallback_reason:
        logger.error("request_id=%s customer_id=%s FALLBACK (%s): %s",
                     request_id, req.customer_id, result.source, result.fallback_reason)

    response = PredictResponse(
        request_id=request_id, customer_id=req.customer_id,
        churn_probability=float(result.probability[0]), churn_prediction=bool(result.prediction[0]),
        threshold=result.threshold, decision_rule=result.decision_rule,
        source=result.source, fallback_reason=result.fallback_reason, warnings=warnings,
    )
    logger.info("request_id=%s customer_id=%s source=%s probability=%.4f prediction=%s warnings=%d latency_ms=%.2f",
                request_id, req.customer_id, response.source, response.churn_probability,
                response.churn_prediction, len(warnings), 1000 * (time.perf_counter() - start))
    return response
