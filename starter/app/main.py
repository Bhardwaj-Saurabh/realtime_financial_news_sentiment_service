"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from utils import load_classifier

MAX_TEXT_CHARS = 2000

# Each record is already a JSON string, so the format is just the message.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    """Emit one log entry as a JSON object."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(entry, default=str))


PREDICTION_REQUESTS = Counter(
    "prediction_requests", "Total prediction requests", ["sentiment"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)
PREDICTION_ERRORS = Counter("prediction_errors", "Total prediction errors")

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        long_indices = [i for i, t in enumerate(texts) if len(t) > MAX_TEXT_CHARS]
        if long_indices:
            log(
                "WARNING",
                "Input text exceeds max length and will be truncated",
                max_chars=MAX_TEXT_CHARS,
                indices=long_indices,
            )

        start = time.perf_counter()
        outputs = classifiers["sentiment"](texts, truncation=True)
        latency_ms = (time.perf_counter() - start) * 1000

        results = [
            PredictionResult(
                text=text,
                sentiment=output["label"],
                confidence=output["score"],
                latency_ms=latency_ms,
            )
            for text, output in zip(texts, outputs)
        ]

        for result in results:
            PREDICTION_LATENCY.observe(result.latency_ms)
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            log(
                "INFO",
                "Prediction",
                sentiment=result.sentiment,
                confidence=round(result.confidence, 4),
                latency_ms=round(result.latency_ms, 2),
            )
        return results
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(e), num_texts=len(texts))
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def require_model() -> None:
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health")
def health():
    require_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    require_model()
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return run_predictions([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    require_model()
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    if any(not t.strip() for t in request.texts):
        raise HTTPException(status_code=422, detail="texts must not contain empty strings")
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
