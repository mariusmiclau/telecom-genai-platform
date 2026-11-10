"""Validated single-host deployment, publication and rollback using pinned images."""

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values

from src.deployment import preflight


def deploy(action: str, env_file: Path, project: str):
    values = {**dotenv_values(env_file), **os.environ}
    directory = Path(values.get("SECRETS_DIR", ".secrets"))
    preflight(values.get("APP_IMAGE", ""), values.get("DOMAIN", ""), directory)
    if not values.get("KNOWLEDGE_VERSION"):
        raise ValueError("KNOWLEDGE_VERSION is required")
    command = ["docker", "compose", "--env-file", str(env_file), "-p", project,
               "-f", "compose.production.yaml"]
    subprocess.run(command + ["config", "--quiet"], check=True)
    if action == "publish":
        subprocess.run(command + ["up", "-d", "--wait", "postgres"], check=True)
        subprocess.run(command + ["run", "--rm", "publisher"], check=True)
    else:
        # Applying a previous deployment file is rollback; no database rows are deleted.
        subprocess.run(command + ["pull", "api", "gateway", "postgres"], check=True)
        subprocess.run(command + ["up", "-d", "--wait", "api", "gateway"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["publish", "apply", "rollback"])
    parser.add_argument("--env-file", type=Path, default=Path("deploy.env"))
    parser.add_argument("--project", default="telecom-prod")
    args = parser.parse_args()
    deploy(args.action, args.env_file, args.project)
