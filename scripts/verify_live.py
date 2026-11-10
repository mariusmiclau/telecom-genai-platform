"""Acceptance smoke query for a credential-backed deployed HTTPS endpoint."""

import argparse
import json
import ssl
import urllib.request
from pathlib import Path


def verify(url: str, token_file: Path, question: str, expected_ticket: str, ca_file=None):
    if not url.startswith("https://"):
        raise ValueError("Live verification requires HTTPS")
    context = ssl.create_default_context(cafile=ca_file)
    headers = {"Authorization": "Bearer " + token_file.read_text().strip(), "Content-Type": "application/json"}
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/chat",
        data=json.dumps({"message": question}).encode(), headers=headers)
    with urllib.request.urlopen(req, context=context, timeout=60) as response:
        result = json.load(response)
    if result.get("outcome") != "answered" or not any(expected_ticket in source["document"] or expected_ticket in (source.get("section") or "") for source in result.get("sources", [])):
        raise RuntimeError("The endpoint did not return an accepted answer with the expected ticket source")
    print(json.dumps({"live_smoke": "passed", "ticket": expected_ticket, "mode": result.get("mode")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path, default=Path(".secrets/api_token"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--expect-ticket", required=True)
    parser.add_argument("--ca-file")
    args = parser.parse_args()
    verify(args.url, args.token_file, args.question, args.expect_ticket, args.ca_file)
