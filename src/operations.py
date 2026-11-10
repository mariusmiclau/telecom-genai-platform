"""Container-compatible operator commands for immutable knowledge publication."""

import argparse
import asyncio
import json
from pathlib import Path

from src.agents.store import PostgresConversationStore
from src.evaluation.dataset import REQUIRED_TICKET_FIELDS, load_jsonl
from src.rag_pipeline.embeddings import EmbeddingService, EmbeddingSettings
from src.rag_pipeline.postgres import PostgresTicketRetriever
from src.settings import AppSettings


async def publish(source: Path, version: str):
    settings = AppSettings.load()
    if not settings.database_url.get_secret_value() or not settings.openai_api_key.get_secret_value():
        raise ValueError("Configure DATABASE_URL and OPENAI_API_KEY before publishing")
    tickets = load_jsonl(source)
    for row in tickets:
        if not REQUIRED_TICKET_FIELDS.issubset(row):
            raise ValueError("Snapshot is missing required ticket fields")
    store = PostgresConversationStore(settings.database_url.get_secret_value())
    embedding_settings = EmbeddingSettings(
        OPENAI_API_KEY=settings.openai_api_key.get_secret_value(),
        model_name=settings.embedding_model_name, dimensions=settings.embedding_dimensions,
        base_url=settings.openai_base_url, cache_enabled=False,
    )
    embeddings = EmbeddingService(embedding_settings)
    retriever = PostgresTicketRetriever(store, embeddings, version, embedding_settings.model_name, embedding_settings.dimensions,
        configuration=settings.retrieval_configuration)
    try:
        await store.start()
        await retriever.start(require_data=False)
        return await retriever.publish(tickets, source.read_bytes())
    finally:
        await retriever.close()
        await store.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    command = subparsers.add_parser("publish")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(publish(args.source, args.version))))


if __name__ == "__main__":
    main()
