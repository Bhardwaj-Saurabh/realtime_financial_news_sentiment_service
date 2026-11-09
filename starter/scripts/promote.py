"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

PRODUCTION_ALIAS = "production"


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Run evaluate.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No finished runs in experiment '{experiment_name}'.")

    run = runs[0]
    if "f1_weighted" not in run.data.metrics:
        raise ValueError(f"Run {run.info.run_id} has no 'f1_weighted' metric.")
    return run.info.run_id, run.data.metrics["f1_weighted"]


def promote(client: MlflowClient, model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions of model '{model_name}'.")

    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, latest.version)
    print(f"Promoted {model_name} v{latest.version} to '{PRODUCTION_ALIAS}'.")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "finbert-evaluation"
    model_name = os.getenv("MODEL_NAME") or "finbert"
    threshold = load_params()["f1_threshold"]

    client = MlflowClient()
    run_id, f1 = get_latest_f1(client, experiment_name)
    print(f"Latest run {run_id}: f1_weighted={f1:.4f} (threshold {threshold})")

    if f1 >= threshold:
        promote(client, model_name)
    else:
        print(
            f"Model not promoted: f1_weighted {f1:.4f} is below the "
            f"threshold {threshold}."
        )


if __name__ == "__main__":
    main()
