"""Explicit operator workflows, separate from the LLM's advisory response path."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.actions import ActionPlan, ActionService
from src.jobs import IngestionQueue, IngestionSubmission

router = APIRouter()


def ingestion(request):
    runtime = request.app.state.runtime
    if runtime.agent is None:
        raise HTTPException(503, "Runtime not ready")
    return IngestionQueue(runtime.agent.memory, runtime.settings)


@router.post("/ingestion/jobs", status_code=202)
async def submit_job(submission: IngestionSubmission, request: Request):
    return await ingestion(request).submit(submission)


@router.get("/ingestion/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    return await ingestion(request).status(job_id)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def actions(request):
    runtime = request.app.state.runtime
    if not runtime.settings.enable_local_actions:
        raise HTTPException(409, "Local action simulation is disabled")
    if runtime.agent is None:
        raise HTTPException(503, "Runtime not ready")
    return ActionService(runtime.agent.memory)


@router.post("/actions", status_code=201)
async def propose(plan: ActionPlan, request: Request):
    return await actions(request).request(plan.model_dump())


@router.post("/actions/{action_id}/approve")
async def approve(action_id: str, decision: Decision, request: Request):
    return await actions(request).transition(action_id, "approve", decision.payload_sha256)


@router.get("/actions/{action_id}")
async def get_action(action_id: str, request: Request):
    return await actions(request).get(action_id)


@router.post("/actions/{action_id}/reject")
async def reject(action_id: str, decision: Decision, request: Request):
    return await actions(request).transition(action_id, "reject", decision.payload_sha256)


@router.post("/actions/{action_id}/execute")
async def execute(action_id: str, decision: Decision, request: Request):
    return await actions(request).transition(action_id, "execute", decision.payload_sha256)


@router.get("/actions/{action_id}/audit")
async def audit(action_id: str, request: Request):
    return await actions(request).audit(action_id)
