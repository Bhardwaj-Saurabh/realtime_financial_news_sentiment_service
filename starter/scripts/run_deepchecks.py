"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from app.utils import load_classifier
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

load_dotenv()

LABELS = ["negative", "neutral", "positive"]


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    test_dataset = TextData(test_texts, task_type="text_classification", name="test")
    stream_dataset = TextData(stream_texts, task_type="text_classification", name="stream")
    # Only Sentiment is checked; the model-based default properties
    # (Toxicity, Fluency, Formality) would add large downloads to CI.
    for dataset in (test_dataset, stream_dataset):
        dataset.calculate_builtin_properties(include_properties=["Sentiment"])

    # Property drift: test set is the reference, stream is the current data.
    property_result = PropertyDrift().run(
        train_dataset=test_dataset, test_dataset=stream_dataset, with_display=False
    )
    sentiment_drift = property_result.value["Sentiment"]["Drift score"]
    print(f"Sentiment property drift: {sentiment_drift:.4f} (threshold {property_drift_threshold})")
    if sentiment_drift > property_drift_threshold:
        print(
            f"ERROR: Sentiment property drift {sentiment_drift:.4f} exceeds "
            f"threshold {property_drift_threshold}."
        )
        sys.exit(1)

    # Prediction drift between model outputs on the two datasets.
    print("Running predictions...")
    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)

    prediction_check = PredictionDrift().add_condition_drift_score_less_than(
        max_allowed_drift_score=prediction_drift_threshold
    )
    prediction_result = prediction_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_predictions,
        test_predictions=stream_predictions,
        model_classes=LABELS,
        with_display=False,
    )
    prediction_drift = prediction_result.value["Drift score"]
    print(f"Prediction drift: {prediction_drift:.4f} (threshold {prediction_drift_threshold})")
    if not prediction_result.passed_conditions():
        print(
            f"ERROR: Prediction drift {prediction_drift:.4f} exceeds "
            f"threshold {prediction_drift_threshold}."
        )
        sys.exit(1)

    print("All drift checks passed.")


if __name__ == "__main__":
    main()
