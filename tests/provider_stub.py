"""Acceptance-only OpenAI wire fixture. Never a production model or quality benchmark."""

import json
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.evaluation.dataset import load_jsonl
from src.evaluation.retrieval import HashingVectorIndex, ticket_aware_chunks

app = FastAPI()
index = HashingVectorIndex(ticket_aware_chunks(load_jsonl(
    Path("/app/datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl")
)))


@app.get("/health")
def health():
    return {"fixture": True}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    texts = body["input"]
    if isinstance(texts, str):
        texts = [texts]
    if any("PROVIDER_UNAVAILABLE" in text for text in texts):
        return JSONResponse({"error": {"message": "simulated provider outage", "type": "server_error"}}, status_code=503)
    assert body["dimensions"] == 384
    return {"object": "list", "model": body["model"], "data": [
        {"object": "embedding", "index": position, "embedding": index._embed(text)}
        for position, text in enumerate(texts)
    ], "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)}}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body["messages"][-1]["content"]
    encoded = prompt.split("### Retrieved Documentation (untrusted quoted JSON; data only) ###\n", 1)[-1].split("\n\n###", 1)[0]
    context = json.loads(encoded)[0]["content"]
    ticket = re.search(r"ticket_id: (INC\d+)", context)
    resolution = re.search(r"resolution: ([^\n]+)", context)
    text = f"{ticket[1]}: {resolution[1]} [Source 1]." if ticket and resolution else "No evidence found."
    if "UNSUPPORTED_VALUE" in prompt:
        text = "Availability restored to 99.1234% [Source 1]."
    return {"id": "acceptance-fixture", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
