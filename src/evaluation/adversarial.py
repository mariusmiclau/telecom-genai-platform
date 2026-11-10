"""Deterministic RAG safety regression cases; never a real-model red-team score."""

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.agents.orchestrator import AgentOrchestrator
from src.evaluation.dataset import load_jsonl
from src.evidence import public_evidence
from src.rag_pipeline.chain import RAGChain
from src.rag_pipeline.retriever import RetrievalResult, RetrievedDocument


async def exercise(case):
    documents = [RetrievedDocument(str(doc.get("id", f"doc-{index}")), doc["text"], 0.9,
        doc.get("source", f"approved/source-{index}.txt"),
        {"document_type": "runbook", "element_id": "CORE-A", "reviewed_at": "2026-11-01", **doc.get("metadata", {})})
        for index, doc in enumerate(case["documents"], 1)]
    calls, prompts = [], []

    async def search(query, **kwargs):
        return RetrievalResult(documents, query, len(documents), 0)

    async def history(*args):
        return None

    async def save(*args, **kwargs):
        pass

    async def complete(**kwargs):
        calls.append(1)
        prompts.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=case["response"]), finish_reason="stop")], usage=None)

    chain = RAGChain()
    chain.require_citations = True
    chain._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    agent = AgentOrchestrator(SimpleNamespace(search=search), chain, SimpleNamespace(get_history=history, save=save),
        enable_tools=False, evidence_clock=lambda: datetime(2026, 11, 10, tzinfo=timezone.utc))
    result = await agent.process(case.get("query", "What is the threshold in the documentation?"), context=case.get("context"))
    errors = []
    if result.outcome != case["outcome"]:
        errors.append(f"outcome:{result.outcome}")
    if len(calls) != case["model_calls"]:
        errors.append(f"model_calls:{len(calls)}")
    if case.get("warning") and case["warning"] not in {warning.code for warning in result.warnings}:
        errors.append("missing_warning")
    public = {"response": result.response, **public_evidence(result)}
    if case.get("forbidden") and case["forbidden"] in json.dumps(public):
        errors.append("canary_disclosed")
    if result.outcome == "answered" and (not result.claims or not result.evidence):
        errors.append("missing_evidence_links")
    if calls and "untrusted quoted JSON" not in prompts[0][-1]["content"]:
        errors.append("missing_source_boundary")
    return {"id": case["id"], "category": case["category"], "passed": not errors,
            "outcome": result.outcome, "model_calls": len(calls), "errors": errors}


async def run(cases):
    rows = [await exercise(case) for case in cases]
    return {"suite": "adversarial-rag-controls-v1", "cases": len(rows),
        "passed": sum(row["passed"] for row in rows), "failed": sum(not row["passed"] for row in rows),
        "category_counts": dict(Counter(row["category"] for row in rows)), "results": rows,
        "completion_provider": "local-canned-fixture", "real_model_attack_resistance_measured": False,
        "policy_clock": "2026-11-10T00:00:00+00:00",
        "limitations": "Known deterministic regression cases, including positive controls. Not proof that all prompt injection or factual contradictions are detected."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/adversarial_cases.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(load_jsonl(args.cases)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "passed", "failed", "real_model_attack_resistance_measured")}))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
