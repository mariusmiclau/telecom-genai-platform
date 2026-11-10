"""Custom-format backup and restore to a NEW database; never overwrite existing data."""

import argparse
import os
import re
import subprocess
from pathlib import Path


def manage(action, backup, env_file, project, restore_database):
    command = ["docker", "compose", "--env-file", str(env_file), "-p", project,
               "-f", "compose.production.yaml", "exec", "-T", "postgres"]
    if action == "backup":
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            subprocess.run(command + ["pg_dump", "-U", "genai", "-d", "telecom_genai", "-Fc"], stdout=stream, check=True)
    else:
        if not re.fullmatch(r"restore_[a-z0-9_]+", restore_database):
            raise ValueError("Restore target must be a new database named restore_<suffix>")
        # createdb fails if the target exists; no --clean, drop or overwrite is used.
        subprocess.run(command + ["createdb", "-U", "genai", restore_database], check=True)
        with backup.open("rb") as stream:
            subprocess.run(command + ["pg_restore", "-U", "genai", "-d", restore_database], stdin=stream, check=True)
    print("Database operation completed; verify the restored data before changing DATABASE_URL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["backup", "restore"])
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("deploy.env"))
    parser.add_argument("--project", default="telecom-prod")
    parser.add_argument("--restore-database", default="")
    args = parser.parse_args()
    manage(args.action, args.file, args.env_file, args.project, args.restore_database)
