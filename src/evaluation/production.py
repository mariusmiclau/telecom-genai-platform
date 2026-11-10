"""Evaluate the actual runtime retriever/SDK/PostgreSQL path, not a substitute index."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from src.agents.orchestrator import AgentOrchestrator
from src.agents.store import PostgresConversationStore
from src.evaluation.dataset import load_jsonl
from src.evaluation.retrieval import SearchHit
from src.evaluation.runner import EvaluationRunner
from src.rag_pipeline.chain import RAGChain
from src.rag_pipeline.embeddings import EmbeddingService, EmbeddingSettings
from src.rag_pipeline.postgres import PostgresTicketRetriever
from src.security import Principal, principal_context
from src.settings import AppSettings


async def evaluate(retrievers, tickets, questions):
    scorer = EvaluationRunner(tickets, questions, retrievers=[])
    configurations = {}
    for name, retriever in retrievers.items():
        await retriever.ready()
        rows, latencies = [], []
        for question in questions:
            started = time.monotonic()
            result = await retriever.search(question["question"], top_k=10)
            latencies.append((time.monotonic() - started) * 1000)
            hits = [SearchHit(doc.document_id, doc.source_path, doc.relevance_score, doc.content) for doc in result.documents]
            rows.append({"question_id": question["question_id"], **scorer._evaluate_question(question, hits)})
        configurations[name] = {"version": retriever.version, "metrics": scorer._aggregate(rows, latencies), "questions": rows}
    return {"backend": "runtime-postgresql", "golden_questions": len(questions), "configurations": configurations}


async def run(settings, tickets, questions, prefix, publish=False, answers=False, fixture=False):
    if fixture and (not settings.allow_http_model_endpoint or not settings.embedding_model_name.startswith("acceptance-")):
        raise ValueError("Fixture classification requires explicit test provider settings")
    store = PostgresConversationStore(settings.database_url.get_secret_value())
    embeddings = EmbeddingService(EmbeddingSettings(OPENAI_API_KEY=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url, model_name=settings.embedding_model_name,
        dimensions=settings.embedding_dimensions, cache_enabled=False))
    chain = RAGChain(model=settings.openai_model)
    chain.api_key, chain.base_url, chain.require_citations = settings.openai_api_key.get_secret_value(), settings.openai_base_url, True
    try:
        await store.start()
        retrievers = {name: PostgresTicketRetriever(store, embeddings, prefix + "-" + name,
            settings.embedding_model_name, settings.embedding_dimensions, configuration=name) for name in ("A", "B", "C")}
        if publish:
            source = json.dumps(tickets, sort_keys=True).encode()
            for retriever in retrievers.values():
                await retriever.publish(tickets, source)
        report = await evaluate(retrievers, tickets, questions)
        report.update(provider="fixture" if fixture else "configured-provider", real_provider_used=not fixture,
                      real_model_quality_measured=False, embedding_model=settings.embedding_model_name,
                      embedding_dimensions=settings.embedding_dimensions)
        if answers:
            agent = AgentOrchestrator(retrievers["C"], chain, store, enable_tools=False,
                source_review_max_age_days=settings.source_review_max_age_days)
            review = []
            for question in questions:
                session = "evaluation-" + prefix + "-" + question["question_id"]
                try:
                    result = await agent.process(question["question"], session)
                    expected = question["expected_sources"]
                    review.append({"question_id": question["question_id"], "outcome": result.outcome,
                        "expected_source_cited": any(source.document in expected for source in result.sources),
                        "reference_answer": question["reference_answer"], "generated_answer": result.response,
                        "human_correctness_review": None})
                finally:
                    await store.forget(session)
            report["generated_answer_review"] = review
            report["automated_expected_source_rate"] = statistics.fmean(row["expected_source_cited"] for row in review)
            report["human_review_required"] = True
        return report
    finally:
        await chain.close()
        await embeddings.close()
        await store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--publish", action="store_true", help="Explicitly create immutable A/B/C snapshots; may incur model cost")
    parser.add_argument("--answers", action="store_true", help="Generate answers for human review; may incur model cost")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--tenant", help="Evaluate with this tenant's enforced document ACLs")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if str(args.output) != "-" and args.output.exists():
        parser.error("Output already exists; choose a new evaluation report path")
    scope = principal_context.set(Principal("evaluation-operator", args.tenant, frozenset({"reader"}), frozenset(args.group), True)) if args.tenant else None
    try:
        report = asyncio.run(run(AppSettings.load(), load_jsonl(args.source), load_jsonl(args.questions),
            args.prefix, args.publish, args.answers, args.fixture))
    finally:
        if scope is not None:
            principal_context.reset(scope)
    if str(args.output) == "-":
        print(json.dumps(report))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"backend": report["backend"], "golden_questions": report["golden_questions"],
                      "recall_at_5": {key: value["metrics"]["recall_at_5"] for key, value in report["configurations"].items()},
                      "real_model_quality_measured": False}))


if __name__ == "__main__":
    main()
