"""SLA Monitoring tool for service level agreement tracking.

Monitors and reports on SLA compliance for network services.
Tracks availability, latency, and throughput against contracted thresholds.
"""

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.tools.registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class SLAThreshold:
    """SLA threshold definition."""

    metric: str
    target: float
    unit: str
    measurement_window: str  # hourly, daily, monthly


@dataclass
class SLAStatus:
    """Current SLA compliance status."""

    service_id: str
    service_name: str
    region: str
    # Availability
    current_availability: float
    target_availability: float
    availability_compliant: bool
    # Latency
    current_latency_ms: float
    target_latency_ms: float
    latency_compliant: bool
    # Error rate
    current_error_rate: float
    target_error_rate: float
    error_rate_compliant: bool
    # Overall
    is_compliant: bool
    breach_count_30d: int
    last_breach: Optional[str]
    breach_details: list[str]
    measurement_period: str


# Default SLA thresholds (as specified in requirements)
DEFAULT_SLA_THRESHOLDS = {
    "availability": SLAThreshold("availability", 99.95, "%", "monthly"),
    "latency": SLAThreshold("latency", 30, "ms", "hourly"),
    "error_rate": SLAThreshold("error_rate", 1.0, "%", "daily"),
}

# Tiered SLA definitions for different service levels
SLA_DEFINITIONS = {
    "gold": {
        "availability": SLAThreshold("availability", 99.99, "%", "monthly"),
        "latency": SLAThreshold("latency", 15, "ms", "hourly"),
        "error_rate": SLAThreshold("error_rate", 0.1, "%", "daily"),
    },
    "silver": {
        "availability": SLAThreshold("availability", 99.95, "%", "monthly"),
        "latency": SLAThreshold("latency", 30, "ms", "hourly"),
        "error_rate": SLAThreshold("error_rate", 1.0, "%", "daily"),
    },
    "bronze": {
        "availability": SLAThreshold("availability", 99.5, "%", "monthly"),
        "latency": SLAThreshold("latency", 50, "ms", "hourly"),
        "error_rate": SLAThreshold("error_rate", 2.0, "%", "daily"),
    },
}

# Simulated services
SERVICES = {
    "region_a": [
        {"id": "svc-001", "name": "Enterprise WAN - Region A", "tier": "gold"},
        {"id": "svc-002", "name": "Business Internet - Region A", "tier": "silver"},
        {"id": "svc-003", "name": "Backup Link - Region A", "tier": "bronze"},
    ],
    "region_b": [
        {"id": "svc-004", "name": "Enterprise WAN - Region B", "tier": "gold"},
        {"id": "svc-005", "name": "Business Internet - Region B", "tier": "silver"},
    ],
}


class SLAMonitorTool(BaseTool):
    """Monitor SLA compliance for network services.

    Provides real-time SLA status and historical compliance data.
    Critical for identifying services at risk of SLA breach.
    """

    name = "sla_monitor"
    description = (
        "Check SLA compliance status for network services. "
        "Returns current metrics vs. SLA thresholds, breach history, "
        "and identifies services at risk of SLA violation."
    )
    parameters = {
        "region": {
            "type": "string",
            "description": "Network region to check (e.g., region_a, region_b)",
        },
        "service_id": {
            "type": "string",
            "description": "Specific service ID to check (optional)",
            "required": False,
        },
        "check_type": {
            "type": "string",
            "enum": ["status", "at_risk", "breaches"],
            "description": "Type of check: current status, at-risk services, or breach history",
            "required": False,
        },
    }

    async def execute(self, params: dict) -> ToolResult:
        """Execute SLA monitoring check."""
        start = time.time()

        region = params.get("region", "region_a").lower().replace(" ", "_")
        service_id = params.get("service_id")
        check_type = params.get("check_type", "status")

        if region not in SERVICES:
            return ToolResult(
                data={"error": f"Unknown region: {region}"},
                summary=f"Region '{region}' not found",
                latency_ms=(time.time() - start) * 1000,
            )

        services = SERVICES[region]

        if service_id:
            # Check specific service
            service = next((s for s in services if s["id"] == service_id), None)
            if not service:
                return ToolResult(
                    data={"error": f"Service {service_id} not found in {region}"},
                    summary=f"Service not found",
                    latency_ms=(time.time() - start) * 1000,
                )
            services = [service]

        if check_type == "status":
            result = self._get_status(services, region)
        elif check_type == "at_risk":
            result = self._get_at_risk(services, region)
        elif check_type == "breaches":
            result = self._get_breaches(services, region)
        else:
            result = self._get_status(services, region)

        latency = (time.time() - start) * 1000
        return ToolResult(
            data=result["data"],
            summary=result["summary"],
            latency_ms=latency,
        )

    def _get_status(self, services: list, region: str) -> dict:
        """Get current SLA status for services."""
        statuses = []
        compliant_count = 0
        at_risk_count = 0

        for svc in services:
            status = self._generate_sla_status(svc, region)
            statuses.append(status.__dict__)

            if status.is_compliant:
                compliant_count += 1
            elif status.current_availability < status.target_availability + 0.1:
                at_risk_count += 1

        return {
            "data": {
                "region": region,
                "services": statuses,
                "total_services": len(services),
                "compliant": compliant_count,
                "non_compliant": len(services) - compliant_count,
                "at_risk": at_risk_count,
            },
            "summary": (
                f"Region {region}: {compliant_count}/{len(services)} services compliant, "
                f"{at_risk_count} at risk"
            ),
        }

    def _get_at_risk(self, services: list, region: str) -> dict:
        """Identify services at risk of SLA breach."""
        at_risk = []

        for svc in services:
            status = self._generate_sla_status(svc, region)

            # Check if approaching threshold (within 0.5% of target)
            availability_margin = status.current_availability - status.target_availability
            latency_margin = status.target_latency_ms - status.current_latency_ms

            if availability_margin < 0.5 or latency_margin < 5:
                at_risk.append({
                    "service": status.__dict__,
                    "availability_margin": round(availability_margin, 3),
                    "latency_margin_ms": round(latency_margin, 1),
                    "risk_level": "high" if availability_margin < 0.1 else "medium",
                })

        return {
            "data": {
                "region": region,
                "at_risk_services": at_risk,
                "count": len(at_risk),
            },
            "summary": f"Found {len(at_risk)} services at risk of SLA breach in {region}",
        }

    def _get_breaches(self, services: list, region: str) -> dict:
        """Get SLA breach history."""
        breaches = []

        for svc in services:
            status = self._generate_sla_status(svc, region)

            if status.breach_count_30d > 0:
                breaches.append({
                    "service_id": status.service_id,
                    "service_name": status.service_name,
                    "breach_count": status.breach_count_30d,
                    "last_breach": status.last_breach,
                    "current_status": "compliant" if status.is_compliant else "breaching",
                })

        total_breaches = sum(b["breach_count"] for b in breaches)

        return {
            "data": {
                "region": region,
                "services_with_breaches": breaches,
                "total_breaches_30d": total_breaches,
            },
            "summary": f"{len(breaches)} services had {total_breaches} total breaches in last 30 days",
        }

    def _generate_sla_status(self, service: dict, region: str) -> SLAStatus:
        """Generate simulated SLA status for a service."""
        tier = service["tier"]
        thresholds = SLA_DEFINITIONS[tier]

        target_avail = thresholds["availability"].target
        target_latency = thresholds["latency"].target
        target_error_rate = thresholds["error_rate"].target

        # Simulate realistic metrics with occasional degradation
        is_degraded = random.random() < 0.1
        breach_details = []

        if is_degraded:
            current_avail = round(random.uniform(target_avail - 0.5, target_avail + 0.1), 3)
            current_latency = round(random.uniform(target_latency * 0.8, target_latency * 1.5), 1)
            current_error_rate = round(random.uniform(target_error_rate * 0.5, target_error_rate * 2.0), 2)
            breach_count = random.randint(1, 5)
            last_breach = (datetime.utcnow() - timedelta(days=random.randint(1, 10))).isoformat() + "Z"
        else:
            current_avail = round(random.uniform(target_avail, 99.999), 3)
            current_latency = round(random.uniform(target_latency * 0.3, target_latency * 0.9), 1)
            current_error_rate = round(random.uniform(0.01, target_error_rate * 0.8), 2)
            breach_count = 0
            last_breach = None

        # Check compliance for each metric
        avail_compliant = current_avail >= target_avail
        latency_compliant = current_latency <= target_latency
        error_rate_compliant = current_error_rate <= target_error_rate

        # Build breach details
        if not avail_compliant:
            breach_details.append(
                f"Availability {current_avail}% below target {target_avail}%"
            )
        if not latency_compliant:
            breach_details.append(
                f"Latency {current_latency}ms exceeds target {target_latency}ms"
            )
        if not error_rate_compliant:
            breach_details.append(
                f"Error rate {current_error_rate}% exceeds target {target_error_rate}%"
            )

        is_compliant = avail_compliant and latency_compliant and error_rate_compliant

        return SLAStatus(
            service_id=service["id"],
            service_name=service["name"],
            region=region,
            current_availability=current_avail,
            target_availability=target_avail,
            availability_compliant=avail_compliant,
            current_latency_ms=current_latency,
            target_latency_ms=target_latency,
            latency_compliant=latency_compliant,
            current_error_rate=current_error_rate,
            target_error_rate=target_error_rate,
            error_rate_compliant=error_rate_compliant,
            is_compliant=is_compliant,
            breach_count_30d=breach_count,
            last_breach=last_breach,
            breach_details=breach_details,
            measurement_period="2026-02-01 to 2026-02-11",
        )
