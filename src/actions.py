"""Two-person, tenant-scoped approval and durable audit for LOCAL simulation only.

No carrier adapter, shell, HTTP request or arbitrary tool execution exists here.
The simulated effect, state transition and audit event commit together.
"""

import hashlib
import json
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.security import current_principal


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    change_summary: str = Field(min_length=1, max_length=1000)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ActionService:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _audit(cursor, action_id, principal, event, payload_hash):
        cursor.execute("SELECT event_hash FROM action_audit WHERE action_id=%s ORDER BY id DESC LIMIT 1", (action_id,))
        previous = cursor.fetchone()
        previous_hash = previous[0] if previous else ""
        value = {"action_id": action_id, "actor": principal.subject, "event": event,
                 "payload_sha256": payload_hash, "previous_hash": previous_hash}
        cursor.execute("INSERT INTO action_audit(action_id,tenant_id,actor,event,payload_sha256,previous_hash,event_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                       (action_id, principal.tenant, principal.subject, event, payload_hash, previous_hash, digest(value)))

    async def request(self, parameters):
        principal = current_principal()
        principal.require("action_request")
        if not principal.enforce_acl:
            raise HTTPException(409, "Actions require verified individual identities, not a shared operator token")
        plan = ActionPlan.model_validate(parameters).model_dump()
        action_id = str(uuid.uuid4())
        payload_hash = digest({"operation": "simulate_config_change", "parameters": plan})

        def operation(cursor):
            cursor.execute("INSERT INTO operational_actions(id,tenant_id,requester,operation,parameters,payload_sha256,expires_at) VALUES (%s,%s,%s,'simulate_config_change',%s,%s,now()+interval '15 minutes')",
                           (action_id, principal.tenant, principal.subject, json.dumps(plan), payload_hash))
            self._audit(cursor, action_id, principal, "requested", payload_hash)
            return {"id": action_id, "state": "pending", "payload_sha256": payload_hash, "mode": "local-simulation"}
        return await self.store.run(operation)

    async def transition(self, action_id, event, expected_hash):
        if event not in ("approve", "reject", "execute"):
            raise ValueError("Unsupported action transition")
        principal = current_principal()
        principal.require("action_execute" if event == "execute" else "action_approve")
        if not principal.enforce_acl:
            raise HTTPException(409, "Action decisions require verified individual identities")

        def operation(cursor):
            cursor.execute("SELECT requester,state,payload_sha256,parameters,operation,expires_at>now(),approver FROM operational_actions WHERE id=%s AND tenant_id=%s FOR UPDATE", (action_id, principal.tenant))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(404, "Action not found")
            requester, state, payload_hash, parameters, action, active, approver = row
            if not active:
                raise HTTPException(409, "Action expired")
            if expected_hash != payload_hash or digest({"operation": action, "parameters": parameters}) != payload_hash:
                raise HTTPException(409, "Action payload changed or digest does not match")
            cursor.execute("SELECT payload_sha256 FROM action_audit WHERE action_id=%s AND event='requested'", (action_id,))
            if cursor.fetchone() != (payload_hash,):
                raise HTTPException(409, "Action audit does not match the plan")
            if event in ("approve", "reject"):
                if state != "pending" or requester == principal.subject:
                    raise HTTPException(409, "An independent approver and pending action are required")
                new_state = "approved" if event == "approve" else "rejected"
                cursor.execute("UPDATE operational_actions SET state=%s,approver=%s WHERE id=%s", (new_state, principal.subject, action_id))
                result = {"id": action_id, "state": new_state}
            else:
                if state != "approved" or approver == requester or action != "simulate_config_change":
                    raise HTTPException(409, "Approved, unexecuted local action required")
                # A local ledger result only. No external system is contacted.
                result = {"id": action_id, "state": "succeeded", "mode": "local-simulation", "node_id": parameters["node_id"]}
                cursor.execute("UPDATE operational_actions SET state='succeeded',executor=%s,result=%s WHERE id=%s", (principal.subject, json.dumps(result), action_id))
            self._audit(cursor, action_id, principal, event, payload_hash)
            return result
        return await self.store.run(operation)

    async def get(self, action_id):
        principal = current_principal()
        def operation(cursor):
            cursor.execute("SELECT requester,operation,parameters,payload_sha256,state,expires_at,approver,executor FROM operational_actions WHERE id=%s AND tenant_id=%s", (action_id, principal.tenant))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(404, "Action not found")
            if principal.subject != row[0] and not principal.roles.intersection({"admin", "action_approve", "action_execute", "auditor"}):
                raise HTTPException(403, "Permission denied")
            return {"id": action_id, **dict(zip(("requester", "operation", "parameters", "payload_sha256", "state", "expires_at", "approver", "executor"), row))}
        return await self.store.run(operation)

    async def audit(self, action_id):
        principal = current_principal()
        principal.require("auditor")

        def operation(cursor):
            cursor.execute("SELECT actor,event,payload_sha256,previous_hash,event_hash FROM action_audit WHERE action_id=%s AND tenant_id=%s ORDER BY id", (action_id, principal.tenant))
            rows = cursor.fetchall()
            if not rows:
                raise HTTPException(404, "Action not found")
            previous, events = "", []
            for actor, event, payload_hash, previous_hash, event_hash in rows:
                value = {"action_id": action_id, "actor": actor, "event": event,
                         "payload_sha256": payload_hash, "previous_hash": previous_hash}
                if previous != previous_hash or digest(value) != event_hash:
                    raise HTTPException(409, "Audit chain verification failed")
                previous = event_hash
                events.append({**value, "event_hash": event_hash})
            return events
        return await self.store.run(operation)
