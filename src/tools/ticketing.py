"""Ticketing tool for incident and change management.

Integrates with ITSM systems to create and manage tickets.
In production, this would call ServiceNow, Jira, or similar APIs.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.tools.registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class Ticket:
    """Represents a ticket in the ITSM system."""

    ticket_id: str
    title: str
    description: str
    priority: str  # critical, high, medium, low
    status: str  # new, in_progress, resolved, closed
    category: str  # incident, change, problem, request
    assignee: Optional[str] = None
    created_at: str = ""
    region: Optional[str] = None
    affected_elements: list[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if self.affected_elements is None:
            self.affected_elements = []


class TicketCreatorTool(BaseTool):
    """Create incident tickets in the ITSM system.

    Used when the agent needs to:
    - Create incident tickets for detected issues
    - Log change requests for configuration modifications
    - Track problems for root cause analysis
    """

    name = "ticket_creator"
    description = (
        "Create a new ticket in the IT Service Management system. "
        "Use this to log incidents, change requests, or problems. "
        "Returns the ticket ID and confirmation details."
    )
    parameters = {
        "title": {
            "type": "string",
            "description": "Brief title describing the issue or request",
        },
        "description": {
            "type": "string",
            "description": "Detailed description of the issue or request",
        },
        "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low"],
            "description": "Ticket priority level",
        },
        "category": {
            "type": "string",
            "enum": ["incident", "change", "problem", "request"],
            "description": "Type of ticket to create",
        },
        "region": {
            "type": "string",
            "description": "Affected network region (optional)",
            "required": False,
        },
        "affected_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of affected network element IDs (optional)",
            "required": False,
        },
    }

    # Simulated ticket storage
    _tickets: dict[str, Ticket] = {}

    async def execute(self, params: dict) -> ToolResult:
        """Create a new ticket."""
        start = time.time()

        title = params.get("title", "Untitled Ticket")
        description = params.get("description", "")
        priority = params.get("priority", "medium")
        category = params.get("category", "incident")
        region = params.get("region")
        affected_elements = params.get("affected_elements", [])

        # Generate ticket ID based on category
        prefix = {
            "incident": "INC",
            "change": "CHG",
            "problem": "PRB",
            "request": "REQ",
        }.get(category, "TKT")

        ticket_id = f"{prefix}{uuid.uuid4().hex[:8].upper()}"

        # Determine assignee based on priority and category
        assignee = self._assign_ticket(priority, category, region)

        # Create ticket
        ticket = Ticket(
            ticket_id=ticket_id,
            title=title[:200],  # Truncate long titles
            description=description[:2000],
            priority=priority,
            status="new",
            category=category,
            assignee=assignee,
            region=region,
            affected_elements=affected_elements,
        )

        # Store ticket (in production, this would call ITSM API)
        self._tickets[ticket_id] = ticket

        latency = (time.time() - start) * 1000

        logger.info(
            f"Ticket created: {ticket_id}",
            extra={
                "ticket_id": ticket_id,
                "priority": priority,
                "category": category,
                "assignee": assignee,
            },
        )

        return ToolResult(
            data={
                "ticket_id": ticket_id,
                "title": ticket.title,
                "priority": priority,
                "status": "new",
                "category": category,
                "assignee": assignee,
                "created_at": ticket.created_at,
                "url": f"https://itsm.example.com/tickets/{ticket_id}",
            },
            summary=f"Created {category} ticket {ticket_id} (priority: {priority}, assigned to: {assignee})",
            latency_ms=latency,
        )

    def _assign_ticket(
        self,
        priority: str,
        category: str,
        region: Optional[str],
    ) -> str:
        """Determine ticket assignee based on routing rules."""
        # Simulated assignment logic
        if priority == "critical":
            return "noc-critical-team@telecom.net"
        elif category == "change":
            return "change-management@telecom.net"
        elif region:
            return f"noc-{region}@telecom.net"
        else:
            return "noc-general@telecom.net"


class TicketQueryTool(BaseTool):
    """Query existing tickets from the ITSM system."""

    name = "ticket_query"
    description = (
        "Search for existing tickets by ID, status, or affected elements. "
        "Use this to check ticket status or find related incidents."
    )
    parameters = {
        "ticket_id": {
            "type": "string",
            "description": "Specific ticket ID to retrieve",
            "required": False,
        },
        "status": {
            "type": "string",
            "enum": ["new", "in_progress", "resolved", "closed"],
            "description": "Filter by ticket status",
            "required": False,
        },
        "element_id": {
            "type": "string",
            "description": "Find tickets affecting a specific network element",
            "required": False,
        },
    }

    async def execute(self, params: dict) -> ToolResult:
        """Query tickets."""
        start = time.time()

        ticket_id = params.get("ticket_id")
        status_filter = params.get("status")
        element_id = params.get("element_id")

        tickets = TicketCreatorTool._tickets

        if ticket_id:
            # Get specific ticket
            ticket = tickets.get(ticket_id)
            if ticket:
                data = {"ticket": ticket.__dict__}
                summary = f"Found ticket {ticket_id}: {ticket.status}"
            else:
                data = {"error": f"Ticket {ticket_id} not found"}
                summary = f"Ticket {ticket_id} not found"
        else:
            # Search tickets
            results = []
            for t in tickets.values():
                if status_filter and t.status != status_filter:
                    continue
                if element_id and element_id not in t.affected_elements:
                    continue
                results.append(t.__dict__)

            data = {"tickets": results, "count": len(results)}
            summary = f"Found {len(results)} tickets matching criteria"

        latency = (time.time() - start) * 1000
        return ToolResult(data=data, summary=summary, latency_ms=latency)
