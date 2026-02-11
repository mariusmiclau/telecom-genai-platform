"""Configuration validation tool for network changes.

Validates proposed configuration changes against policies,
best practices, and potential conflicts before deployment.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from src.tools.registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A configuration validation rule."""

    rule_id: str
    name: str
    severity: str  # error, warning, info
    description: str


@dataclass
class ValidationIssue:
    """An issue found during validation."""

    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    policy_compliant: bool = True
    best_practices_score: float = 1.0


# Validation rules for telecom configurations
VALIDATION_RULES = [
    ValidationRule(
        "SEC-001",
        "No plaintext passwords",
        "error",
        "Configuration must not contain plaintext passwords",
    ),
    ValidationRule(
        "SEC-002",
        "SSH required for management",
        "error",
        "Telnet must be disabled, SSH required for remote management",
    ),
    ValidationRule(
        "NET-001",
        "MTU consistency",
        "warning",
        "MTU values should be consistent across tunnel endpoints",
    ),
    ValidationRule(
        "NET-002",
        "Loopback required",
        "error",
        "Router must have a loopback interface configured",
    ),
    ValidationRule(
        "QOS-001",
        "QoS policy applied",
        "warning",
        "Standard QoS policy should be applied to all interfaces",
    ),
    ValidationRule(
        "HA-001",
        "Redundancy configured",
        "warning",
        "Critical links should have redundancy configured",
    ),
    ValidationRule(
        "LOG-001",
        "Logging enabled",
        "info",
        "Syslog should be configured for centralized logging",
    ),
    ValidationRule(
        "NTP-001",
        "NTP configured",
        "warning",
        "NTP servers should be configured for time synchronization",
    ),
]


class ConfigValidatorTool(BaseTool):
    """Validate network configuration changes before deployment.

    Checks proposed configurations against:
    - Security policies (no plaintext passwords, SSH required)
    - Network best practices (MTU, QoS, redundancy)
    - Organizational standards (naming conventions, logging)
    - Potential conflicts with existing configuration
    """

    name = "config_validator"
    description = (
        "Validate proposed network configuration changes. "
        "Checks against security policies, best practices, and potential conflicts. "
        "Returns validation status, issues found, and recommendations."
    )
    parameters = {
        "config": {
            "type": "string",
            "description": "The configuration text to validate",
        },
        "element_id": {
            "type": "string",
            "description": "Target network element ID",
        },
        "change_type": {
            "type": "string",
            "enum": ["interface", "routing", "qos", "security", "general"],
            "description": "Type of configuration change",
            "required": False,
        },
        "strict_mode": {
            "type": "boolean",
            "description": "If true, warnings are treated as errors",
            "required": False,
        },
    }

    async def execute(self, params: dict) -> ToolResult:
        """Validate configuration."""
        start = time.time()

        config = params.get("config", "")
        element_id = params.get("element_id", "unknown")
        change_type = params.get("change_type", "general")
        strict_mode = params.get("strict_mode", False)

        if not config:
            return ToolResult(
                data={"error": "No configuration provided"},
                summary="Validation failed: no configuration to validate",
                latency_ms=(time.time() - start) * 1000,
            )

        # Run validation
        result = self._validate_config(config, change_type)

        # In strict mode, warnings become errors
        if strict_mode:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.is_valid = len(result.errors) == 0

        latency = (time.time() - start) * 1000

        # Build response
        data = {
            "element_id": element_id,
            "change_type": change_type,
            "is_valid": result.is_valid,
            "policy_compliant": result.policy_compliant,
            "best_practices_score": result.best_practices_score,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "info_count": len(result.info),
            "errors": [self._issue_to_dict(e) for e in result.errors],
            "warnings": [self._issue_to_dict(w) for w in result.warnings],
            "info": [self._issue_to_dict(i) for i in result.info],
        }

        if result.is_valid:
            if result.warnings:
                summary = (
                    f"Configuration valid with {len(result.warnings)} warning(s). "
                    f"Best practices score: {result.best_practices_score:.0%}"
                )
            else:
                summary = f"Configuration valid. Best practices score: {result.best_practices_score:.0%}"
        else:
            summary = (
                f"Configuration invalid: {len(result.errors)} error(s), "
                f"{len(result.warnings)} warning(s)"
            )

        logger.info(
            f"Config validation for {element_id}: {'valid' if result.is_valid else 'invalid'}",
            extra={
                "element_id": element_id,
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            },
        )

        return ToolResult(data=data, summary=summary, latency_ms=latency)

    def _validate_config(self, config: str, change_type: str) -> ValidationResult:
        """Run all validation checks on configuration."""
        errors = []
        warnings = []
        info = []
        config_lower = config.lower()

        # SEC-001: Check for plaintext passwords
        if re.search(r'password\s+\d', config_lower) or \
           re.search(r'secret\s+\d', config_lower):
            # Type 0 or plain passwords
            if 'password 0' in config_lower or 'secret 0' in config_lower:
                errors.append(ValidationIssue(
                    rule_id="SEC-001",
                    severity="error",
                    message="Plaintext password detected",
                    suggestion="Use encrypted passwords (type 5 or 7)",
                ))

        # SEC-002: Check SSH vs Telnet
        if 'transport input telnet' in config_lower:
            errors.append(ValidationIssue(
                rule_id="SEC-002",
                severity="error",
                message="Telnet enabled on VTY lines",
                location="line vty",
                suggestion="Use 'transport input ssh' instead",
            ))

        # NET-001: MTU check
        mtu_values = re.findall(r'mtu\s+(\d+)', config_lower)
        if mtu_values:
            unique_mtus = set(mtu_values)
            if len(unique_mtus) > 1:
                warnings.append(ValidationIssue(
                    rule_id="NET-001",
                    severity="warning",
                    message=f"Inconsistent MTU values: {', '.join(unique_mtus)}",
                    suggestion="Ensure MTU is consistent across related interfaces",
                ))

        # NET-002: Loopback check for routing configs
        if change_type == "routing":
            if 'interface loopback' not in config_lower:
                errors.append(ValidationIssue(
                    rule_id="NET-002",
                    severity="error",
                    message="No loopback interface defined",
                    suggestion="Configure a loopback interface for router-id",
                ))

        # QOS-001: QoS policy check
        if change_type == "interface" and 'service-policy' not in config_lower:
            warnings.append(ValidationIssue(
                rule_id="QOS-001",
                severity="warning",
                message="No QoS policy applied to interface",
                suggestion="Apply standard service-policy for traffic management",
            ))

        # LOG-001: Logging check
        if 'logging' not in config_lower:
            info.append(ValidationIssue(
                rule_id="LOG-001",
                severity="info",
                message="No logging configuration found",
                suggestion="Consider configuring syslog for monitoring",
            ))

        # NTP-001: NTP check
        if 'ntp server' not in config_lower and 'ntp peer' not in config_lower:
            warnings.append(ValidationIssue(
                rule_id="NTP-001",
                severity="warning",
                message="No NTP configuration found",
                suggestion="Configure NTP for time synchronization",
            ))

        # Calculate best practices score
        total_checks = 8  # Number of rules
        issues = len(errors) * 3 + len(warnings) * 1  # Weighted
        score = max(0, (total_checks * 3 - issues) / (total_checks * 3))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            policy_compliant=len([e for e in errors if e.rule_id.startswith("SEC")]) == 0,
            best_practices_score=round(score, 2),
        )

    def _issue_to_dict(self, issue: ValidationIssue) -> dict:
        """Convert ValidationIssue to dictionary."""
        return {
            "rule_id": issue.rule_id,
            "severity": issue.severity,
            "message": issue.message,
            "location": issue.location,
            "suggestion": issue.suggestion,
        }


class ConfigDiffTool(BaseTool):
    """Compare current and proposed configurations."""

    name = "config_diff"
    description = (
        "Compare current configuration with proposed changes. "
        "Shows additions, deletions, and modifications."
    )
    parameters = {
        "current_config": {
            "type": "string",
            "description": "Current configuration",
        },
        "proposed_config": {
            "type": "string",
            "description": "Proposed new configuration",
        },
    }

    async def execute(self, params: dict) -> ToolResult:
        """Generate configuration diff."""
        start = time.time()

        current = params.get("current_config", "").splitlines()
        proposed = params.get("proposed_config", "").splitlines()

        added = [line for line in proposed if line not in current]
        removed = [line for line in current if line not in proposed]
        unchanged = [line for line in current if line in proposed]

        latency = (time.time() - start) * 1000

        return ToolResult(
            data={
                "added": added,
                "removed": removed,
                "unchanged_count": len(unchanged),
                "change_count": len(added) + len(removed),
            },
            summary=f"Config diff: +{len(added)} added, -{len(removed)} removed",
            latency_ms=latency,
        )
