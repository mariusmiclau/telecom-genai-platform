"""Exercise the actual production stack over TLS using an explicit provider fixture.

Creates its own named Docker project and removes only that project's synthetic
test containers and volumes. Never connects to a real model account.
"""

import argparse
import json
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from src.deployment import initialize_secrets
from src.monitoring.tracing import REQUIRED_TRACE_STAGES


def verify_trace_export(logs: str, trace_id: str):
    """Check real exported JSON spans for one API response, not span-name declarations."""
    decoder = json.JSONDecoder()
    spans = []
    position = 0
    while (position := logs.find("{", position)) != -1:
        try:
            value, end = decoder.raw_decode(logs, position)
            position = end
        except ValueError:
            position += 1
            continue
        context = value.get("context") if isinstance(value, dict) else None
        if isinstance(context, dict) and context.get("trace_id") == "0x" + trace_id:
            spans.append(value)
    names = {span.get("name") for span in spans}
    missing = set(REQUIRED_TRACE_STAGES) - names
    if missing:
        raise AssertionError("Missing exported trace stages: " + ", ".join(sorted(missing)))
    for span in spans:
        attributes = span.get("attributes") or {}
        assert "genai.query" not in attributes, "Raw query exported with capture disabled"
        assert "telecom.request.context" not in attributes, "Raw context exported with capture disabled"
        assert "telecom.entities" not in attributes, "Raw entities exported with capture disabled"
    return sorted(set(REQUIRED_TRACE_STAGES) & names)


def verify_metrics_export(metrics: str):
    samples = [sample for family in text_string_to_metric_families(metrics) for sample in family.samples]
    expected = {"answered": 2, "abstained": 1, "dependency_error": 1}
    for outcome, minimum in expected.items():
        count = sum(sample.value for sample in samples if sample.name == "genai_answer_outcomes_total"
                    and sample.labels.get("outcome") == outcome)
        assert count >= minimum, f"Missing monitoring outcome: {outcome}"
    failures = sum(sample.value for sample in samples if sample.name == "genai_auth_failures_total")
    assert failures >= 1, "Rejected authentication was not counted"


def accept(image: str, output: Path):
    project = "telco-acceptance-" + uuid.uuid4().hex[:10]
    directory = Path(".release-check") / project
    initialize_secrets(directory / "secrets", "acceptance-only-not-a-real-model-key")
    environment = {
        **os.environ, "APP_IMAGE": image, "DOMAIN": "localhost",
        "HTTP_BIND": "127.0.0.1:18080", "HTTPS_BIND": "127.0.0.1:18443",
        "SECRETS_DIR": (directory / "secrets").resolve().as_posix(),
        "KNOWLEDGE_VERSION": project,
    }
    command = ["docker", "compose", "-p", project, "-f", "compose.production.yaml", "-f", "infrastructure/docker/compose.acceptance.yaml"]

    def run(*args, capture=False, check=True):
        return subprocess.run(command + list(args), env=environment, check=check,
                              text=True, capture_output=capture)

    evidence = {"project": project, "provider": "local-wire-fixture", "real_model_quality_measured": False}
    try:
        run("up", "-d", "--wait", "postgres", "provider")
        run("run", "--rm", "publisher")
        run("up", "-d", "--wait", "api", "gateway")
        certificate = directory / "root.crt"
        run("cp", "gateway:/data/caddy/pki/authorities/local/root.crt", str(certificate))
        context = ssl.create_default_context(cafile=str(certificate))
        token = (directory / "secrets" / "api_token").read_text()

        def request(path, payload=None, authenticated=True, method=None):
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["Authorization"] = "Bearer " + token
            req = urllib.request.Request("https://localhost:18443" + path,
                data=json.dumps(payload).encode() if payload is not None else None,
                headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, context=context, timeout=60) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as error:
                return error.code, error.read().decode()

        assert request("/ready", authenticated=False)[0] == 200
        assert request("/metrics")[0] == 404
        query = {"message": "Which incident fixed DIAMETER_TOO_BUSY on PCRF-LON-01?",
                 "context": {"operator_note": "acceptance-private-context"}}
        assert request("/api/v1/chat", query, authenticated=False)[0] == 401
        status, body = request("/api/v1/chat", query)
        result = json.loads(body)
        assert status == 200, body
        assert result["outcome"] == "answered", body
        assert "INC20260001" in result["response"], body
        assert result["sources"][0]["document"].endswith("INC20260001.json"), body
        assert result["trace_id"] and len(result["trace_id"]) == 32, body
        deadline = time.monotonic() + 10
        while True:
            try:
                stages = verify_trace_export(run("logs", "--no-log-prefix", "--no-color", "api", capture=True).stdout, result["trace_id"])
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        assert request("/api/v1/chat", {**query, "session_id": result["session_id"]})[0] == 200
        unsupported_status, unsupported_body = request("/api/v1/chat", {"message": "PCRF-LON-01 UNSUPPORTED_VALUE"})
        assert unsupported_status == 200
        unsupported = json.loads(unsupported_body)
        assert unsupported["outcome"] == "abstained", unsupported_body
        assert "99.1234" not in unsupported["response"]
        assert request("/api/v1/sessions/" + unsupported["session_id"], method="DELETE")[0] == 204
        assert request("/api/v1/chat", {"message": "PROVIDER_UNAVAILABLE"})[0] == 503
        assert request("/api/v1/sessions/" + result["session_id"], method="DELETE")[0] == 204
        metrics = run("exec", "-T", "api", "python", "-c",
            "from pathlib import Path; from urllib.request import Request,urlopen; "
            "request=Request('http://localhost:8000/metrics', headers={'Authorization': 'Bearer '+Path('/run/secrets/api_token').read_text()}); "
            "print(urlopen(request).read().decode())", capture=True).stdout
        verify_metrics_export(metrics)

        def database(*args):
            return run("exec", "-T", "postgres", *args, capture=True).stdout.strip()
        assert database("psql", "-U", "genai", "-d", "telecom_genai", "-Atc",
                        "SELECT count(*) FROM platform_messages") == "0"
        database("pg_dump", "-U", "genai", "-d", "telecom_genai", "-Fc", "-f", "/tmp/acceptance.dump")
        database("createdb", "-U", "genai", "acceptance_restore")
        database("pg_restore", "-U", "genai", "-d", "acceptance_restore", "/tmp/acceptance.dump")
        assert database("psql", "-U", "genai", "-d", "acceptance_restore", "-Atc",
                        "SELECT count(*) FROM knowledge_chunks") == "150"
        evidence.update(tls_verified=True, unauthorized_rejected=True, live_rag_answer=True,
                        session_followup=True, provider_failure_sanitized=True, session_deletion=True,
                        unsupported_answer_blocked=True, backup_restore=True, restored_chunks=150,
                        exported_trace_stages=stages, trace_content_disabled=True, monitoring_verified=True, status="passed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence))
    except Exception:
        run("logs", "--tail", "40", check=False)
        raise
    finally:
        run("down", "--volumes", check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="genaitelco-demo:latest")
    parser.add_argument("--output", type=Path, default=Path(".release-check/deployment-acceptance.json"))
    args = parser.parse_args()
    accept(args.image, args.output)
