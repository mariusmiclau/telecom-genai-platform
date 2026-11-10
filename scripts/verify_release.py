"""Smoke test a running demo process or container; exits nonzero on failure."""

import argparse
import json
import time
import urllib.error
import urllib.request


def verify(base_url: str):
    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()

    deadline = time.monotonic() + 45
    while True:
        try:
            assert request("/ready")[0] == 200
            break
        except (urllib.error.URLError, AssertionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    assert "Ask the evidence" in request("/demo")[1]
    assert "/api/v1/demo/chat" in request("/static/demo.js")[1]
    response = json.loads(request("/api/v1/demo/chat", {
        "message": "Which incident fixed DIAMETER_TOO_BUSY on PCRF-LON-01?",
    })[1])
    assert response["outcome"] == "answered"
    assert "INC20260001" in response["response"]
    assert "stale route" in response["response"]
    assert response["sources"][0]["section"] == "INC20260001"
    assert response["evidence"][0]["citation_id"] == "source-1"
    assert response["claims"] and response["answer_kind"] == "historical_resolution"
    assert any(warning["code"] == "historical_resolution" for warning in response["warnings"])
    no_evidence = json.loads(request("/api/v1/demo/chat", {"message": "purple macaroni unicorn astronomy"})[1])
    assert no_evidence["outcome"] == "abstained"
    print(json.dumps({"demo_smoke": "passed", "base_url": base_url, "ticket": "INC20260001"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    verify(parser.parse_args().url)
