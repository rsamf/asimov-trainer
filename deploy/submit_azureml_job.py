"""Submit a dancer/ training job to Azure Machine Learning.

Counterpart of the SageMaker submission script in the old AWS template.
Authenticates via DefaultAzureCredential (picks up the OIDC token GitHub
Actions injects via azure/login), then submits a serverless command job that
runs `python -m dancer.train <hydra-overrides>` inside the ACR image.

Run locally for testing:
    az login
    python deploy/submit_azureml_job.py \\
        --job-name smoke \\
        --image-uri myacr.azurecr.io/asimov-trainer:abc1234 \\
        --subscription-id ... --resource-group ... --workspace-name ... \\
        --config deploy/azureml-job-config.yaml \\
        --hydra-config deploy/train-config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, JobResourceConfiguration
from azure.identity import DefaultAzureCredential


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-name", required=True,
                   help="Display name suffix (e.g. short SHA + branch hint).")
    p.add_argument("--image-uri", required=True,
                   help="Full ACR image URI, e.g. myacr.azurecr.io/asimov-trainer:<sha>.")
    p.add_argument("--subscription-id", required=True)
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--config", required=True, type=Path,
                   help="Path to azureml-job-config.yaml.")
    p.add_argument("--hydra-config", required=True, type=Path,
                   help="Path to YAML with `overrides: [\"key=value\", ...]`.")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    hydra_cfg = load_yaml(args.hydra_config)
    overrides = list(hydra_cfg.get("overrides", []))

    # Persist everything under the AML-mounted output:
    #   - hydra run dir + dancer log_dir (checkpoints + console log)
    #   - nebo .nebo/ (events file written by nb.init() / nb.log_line)
    # ${{outputs.run_dir}} is templated by AML at job start to the mount path.
    train_cmd = ["python", "-m", "dancer.train", *overrides,
                 "hydra.run.dir=${{outputs.run_dir}}",
                 "log_dir=${{outputs.run_dir}}"]
    cmd_str = (
        "mkdir -p ${{outputs.run_dir}}/.nebo && "
        "NEBO_URI=${{outputs.run_dir}}/.nebo "
        + " ".join(train_cmd)
    )

    display_name = f"{cfg.get('display_name_prefix', 'dancer')}-{args.job_name}"

    job = command(
        display_name=display_name,
        experiment_name=cfg["experiment_name"],
        environment=Environment(image=args.image_uri),
        command=cmd_str,
        outputs={"run_dir": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
        resources=JobResourceConfiguration(
            instance_type=cfg["instance_type"],
            instance_count=int(cfg.get("instance_count", 1)),
        ),
        environment_variables=cfg.get("environment_variables", {}) or {},
    )

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    submitted = ml_client.jobs.create_or_update(job)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
