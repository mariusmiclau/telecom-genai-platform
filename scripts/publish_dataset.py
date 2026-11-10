"""Compatibility entry point; use python -m src.operations publish in containers."""

import argparse
import asyncio
import json
from pathlib import Path

from src.operations import publish

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(publish(args.source, args.version))))
