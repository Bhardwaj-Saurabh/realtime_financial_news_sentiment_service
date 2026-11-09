"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)
SENTIMENTS = ("positive", "negative", "neutral")


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    counts = Counter(r["sentiment"] for r in window)
    metrics = {f"pct_{s}": 100 * counts.get(s, 0) / len(window) for s in SENTIMENTS}
    metrics["avg_confidence"] = mean(r["confidence"] for r in window)
    metrics["avg_latency_ms"] = mean(r["latency_ms"] for r in window)
    metrics["window_size"] = len(window)
    mlflow.log_metrics(metrics, step=window_idx)

    print(
        f"Window {window_idx}: {len(window)} predictions | "
        + " ".join(f"{s}={metrics[f'pct_{s}']:.1f}%" for s in SENTIMENTS)
        + f" | confidence={metrics['avg_confidence']:.3f}"
        + f" | latency={metrics['avg_latency_ms']:.1f}ms"
    )


def main():
    headlines = pd.read_csv("data/stream.csv")["text"].dropna().astype(str).tolist()
    print(f"Loaded {len(headlines)} headlines; sending to {API_URL}/predict")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.set_tag("run_type", "monitoring")
        mlflow.log_params({
            "api_url": API_URL,
            "window_size": WINDOW_SIZE,
            "sleep_ms": SLEEP_MS,
            "num_headlines": len(headlines),
        })

        window: list[dict] = []
        window_idx = 0
        errors = 0
        for text in headlines:
            try:
                window.append(predict(text))
            except requests.RequestException as e:
                errors += 1
                print(f"Request failed: {e}")
                continue

            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window_idx += 1
                window = []
            time.sleep(SLEEP_MS / 1000)

        # Log the final partial window so no predictions are dropped.
        if window:
            log_window(window, window_idx)

        mlflow.log_metric("request_errors", errors)
        print(f"Done: {len(headlines) - errors} predictions, {errors} errors")


if __name__ == "__main__":
    main()
