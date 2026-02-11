"""Network health check tools for telecom operations.

Provides agent with ability to query live network status,
check element health, and retrieve performance metrics.

In production, these would call real OSS/NMS APIs.
This implementation uses simulated data for demonstration.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

from src.tools.registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# Simulated network topology
NETWORK_TOPOLOGY = {
    "region_a": {
        "nodes": [f"node-{i}" for i in range(1, 13)],
        "core_router": "cr-a-01",
        "edge_routers": ["er-a-01", "er-a-02", "er-a-03"],
    },
    "region_b": {
        "nodes": [f"node-{i}" for i in range(13, 25)],
        "core_router": "cr-b-01",
        "edge_routers": ["er-b-01", "er-b-02"],
    },
}


@dataclass
class NetworkElementStatus:
    """Status of a network element."""

    element_id: str
    element_type: str
    status: str  # operational, degraded, down
    availability: float
    latency_ms: float
    throughput_mbps: float
    error_rate: float
    last_updated: str


class NetworkHealthTool(BaseTool):
    """Check network element health and performance metrics."""

    name = "network_health_check"
    description = (
        "Query the current health status of network elements in a specified region. "
        "Returns availability, latency, throughput, and error rates for each element."
    )
    parameters = {
        "region": {"type": "string", "description": "Network region to query (e.g., region_a, region_b)"},
        "element_id": {"type": "string", "description": "Specific element ID (optional)", "required": False},
    }

    async def execute(self, params: dict) -> ToolResult:
        """Execute network health check."""
        start = time.time()
        region = params.get("region", "region_a").lower().replace(" ", "_")
        element_id = params.get("element_id")

        if region not in NETWORK_TOPOLOGY:
            return ToolResult(
                data={"error": f"Unknown region: {region}"},
                summary=f"Region '{region}' not found in network topology",
                latency_ms=(time.time() - start) * 1000,
            )

        topology = NETWORK_TOPOLOGY[region]

        if element_id:
            # Query specific element
            status = self._get_element_status(element_id, region)
            data = {"element": status.__dict__}
            summary = f"{element_id}: {status.status}, availability: {status.availability}%"
        else:
            # Query all elements in region
            elements = []
            degraded = []
            for node in topology["nodes"]:
                status = self._get_element_status(node, region)
                elements.append(status.__dict__)
                if status.status != "operational":
                    degraded.append(node)

            overall_availability = sum(e["availability"] for e in elements) / len(elements)
            data = {
                "region": region,
                "total_elements": len(elements),
                "overall_availability": round(overall_availability, 2),
                "degraded_elements": degraded,
                "elements": elements,
            }
            summary = (
                f"Region {region}: {len(elements)} elements, "
                f"availability: {overall_availability:.2f}%, "
                f"degraded: {len(degraded)}"
            )

        latency = (time.time() - start) * 1000
        logger.info(f"Network health check completed: {summary}")

        return ToolResult(data=data, summary=summary, latency_ms=latency)

    def _get_element_status(self, element_id: str, region: str) -> NetworkElementStatus:
        """Generate simulated status for a network element."""
        # Simulate realistic network metrics with occasional degradation
        is_degraded = random.random() < 0.15  # 15% chance of degradation
        is_down = random.random() < 0.02  # 2% chance of down

        if is_down:
            status = "down"
            availability = round(random.uniform(0, 50), 2)
            latency = 0
            throughput = 0
            error_rate = round(random.uniform(50, 100), 2)
        elif is_degraded:
            status = "degraded"
            availability = round(random.uniform(95, 99.5), 2)
            latency = round(random.uniform(40, 80), 1)
            throughput = round(random.uniform(500, 800), 1)
            error_rate = round(random.uniform(1, 5), 2)
        else:
            status = "operational"
            availability = round(random.uniform(99.5, 99.99), 2)
            latency = round(random.uniform(5, 25), 1)
            throughput = round(random.uniform(900, 1000), 1)
            error_rate = round(random.uniform(0, 0.5), 3)

        return NetworkElementStatus(
            element_id=element_id,
            element_type="network_node",
            status=status,
            availability=availability,
            latency_ms=latency,
            throughput_mbps=throughput,
            error_rate=error_rate,
            last_updated="2026-02-11T12:00:00Z",
        )


class NetworkConfigTool(BaseTool):
    """Retrieve and validate network configuration."""

    name = "network_config"
    description = (
        "Retrieve current network configuration for a specific element. "
        "Can also validate proposed configuration changes against policies."
    )
    parameters = {
        "element_id": {"type": "string", "description": "Network element ID"},
        "action": {
            "type": "string",
            "description": "Action: 'get' to retrieve config, 'validate' to check proposed changes",
        },
        "proposed_config": {
            "type": "object",
            "description": "Proposed configuration changes (for validate action)",
            "required": False,
        },
    }

    async def execute(self, params: dict) -> ToolResult:
        """Execute network config retrieval or validation."""
        start = time.time()
        element_id = params.get("element_id", "unknown")
        action = params.get("action", "get")

        if action == "get":
            config = {
                "element_id": element_id,
                "interfaces": [
                    {"name": "eth0", "status": "up", "speed": "10Gbps", "mtu": 9000},
                    {"name": "eth1", "status": "up", "speed": "10Gbps", "mtu": 9000},
                ],
                "routing": {"protocol": "OSPF", "area": "0.0.0.0"},
                "qos_policy": "telecom-standard-v2",
                "last_modified": "2026-02-10T08:30:00Z",
                "modified_by": "admin@telecom.net",
            }
            summary = f"Retrieved config for {element_id}: 2 interfaces, OSPF routing"
        else:
            # Validate proposed changes
            config = {
                "validation": "passed",
                "warnings": ["MTU change may affect existing tunnels"],
                "policy_compliance": True,
            }
            summary = f"Config validation for {element_id}: passed with 1 warning"

        latency = (time.time() - start) * 1000
        return ToolResult(data=config, summary=summary, latency_ms=latency)
