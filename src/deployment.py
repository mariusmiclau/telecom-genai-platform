"""Prepare protected secret files and validate an immutable deployment configuration."""

import argparse
import os
import re
import secrets
from pathlib import Path


def initialize_secrets(directory: Path, model_key: str):
    if not model_key:
        raise ValueError("Set OPENAI_API_KEY in the environment before initializing secrets")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    filenames = ("api_token", "admin_token", "database_password", "database_url", "openai_api_key")
    if any((directory / name).exists() for name in filenames):
        raise ValueError("Secret files already exist; initialization never overwrites credentials")
    password = secrets.token_urlsafe(32)
    values = {
        "api_token": secrets.token_urlsafe(40), "admin_token": secrets.token_urlsafe(40),
        "database_password": password,
        "database_url": f"postgresql://genai:{password}@postgres:5432/telecom_genai",
        "openai_api_key": model_key,
    }
    for name, value in values.items():
        with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        # Directory traversal is restricted on the host; individual file mounts
        # must remain readable by the non-root application and database users.
        (directory / name).chmod(0o444)


def preflight(image: str, domain: str, directory: Path):
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("APP_IMAGE must be a registry image pinned by sha256 digest")
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", domain) or "." not in domain:
        raise ValueError("DOMAIN must be a DNS hostname")
    from src.settings import AppSettings
    AppSettings(_env_file=None, _secrets_dir=directory, app_mode="live")
    if not (directory / "database_password").is_file():
        raise ValueError("database_password secret is required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["init", "preflight"])
    parser.add_argument("--secrets-dir", type=Path, default=Path(".secrets"))
    args = parser.parse_args()
    if args.operation == "init":
        initialize_secrets(args.secrets_dir, os.getenv("OPENAI_API_KEY", ""))
        print("Secret files created; credential values were not printed")
    else:
        preflight(os.getenv("APP_IMAGE", ""), os.getenv("DOMAIN", ""), args.secrets_dir)
        print("Deployment configuration passed preflight")


if __name__ == "__main__":
    main()
