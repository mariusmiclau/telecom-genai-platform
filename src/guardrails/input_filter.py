"""Input validation and filtering for user messages.

Protects against prompt injection, excessive content, and blocked patterns.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_safe: bool
    reason: Optional[str] = None
    risk_score: float = 0.0


# Prompt injection patterns to detect
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    # Role manipulation
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"pretend\s+(to\s+be|you\s+are)\s+",
    r"roleplay\s+as\s+",
    # System prompt extraction
    r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)",
    # Jailbreak attempts
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
    r"bypass\s+(the\s+)?(content\s+)?(filter|policy|restriction)",
    # Code execution attempts
    r"execute\s+(this\s+)?(code|command|script)",
    r"run\s+(this\s+)?(code|command|script)",
    r"eval\s*\(",
    r"exec\s*\(",
]

# Content that should be blocked
BLOCKED_CONTENT = [
    # Harmful intent
    r"(how\s+to\s+)?(hack|attack|exploit|compromise)\s+(a\s+)?(network|system|server)",
    r"(create|make|build)\s+(a\s+)?(malware|virus|ransomware|trojan)",
    # Sensitive data requests
    r"(give|show|list)\s+(me\s+)?(all\s+)?(passwords|credentials|secrets|api\s*keys)",
    r"(access|dump|extract)\s+(the\s+)?(database|customer\s+data)",
]

# Maximum allowed message length (characters)
MAX_MESSAGE_LENGTH = 10000

# Maximum allowed message length for certain operations
MAX_QUERY_LENGTH = 2000


class InputFilter:
    """Filter and validate user input for safety.

    Checks for:
    - Prompt injection attempts
    - Blocked content patterns
    - Message length limits
    - Suspicious character patterns
    """

    def __init__(
        self,
        max_length: int = MAX_MESSAGE_LENGTH,
        injection_patterns: Optional[list[str]] = None,
        blocked_patterns: Optional[list[str]] = None,
    ):
        """Initialize the input filter.

        Args:
            max_length: Maximum allowed message length
            injection_patterns: Custom injection patterns (uses defaults if None)
            blocked_patterns: Custom blocked content patterns (uses defaults if None)
        """
        self.max_length = max_length
        self.injection_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (injection_patterns or INJECTION_PATTERNS)
        ]
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (blocked_patterns or BLOCKED_CONTENT)
        ]

    def validate(self, message: str) -> ValidationResult:
        """Validate a user message for safety.

        Args:
            message: The user's input message

        Returns:
            ValidationResult with is_safe flag and reason if blocked
        """
        if not message:
            return ValidationResult(is_safe=True)

        # Check message length
        if len(message) > self.max_length:
            logger.warning(
                f"Message too long: {len(message)} chars (max: {self.max_length})"
            )
            return ValidationResult(
                is_safe=False,
                reason=f"Message exceeds maximum length of {self.max_length} characters",
                risk_score=0.3,
            )

        # Check for prompt injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(message):
                logger.warning(
                    f"Prompt injection detected: {pattern.pattern[:50]}...",
                    extra={"pattern": pattern.pattern},
                )
                return ValidationResult(
                    is_safe=False,
                    reason="Potential prompt injection detected",
                    risk_score=0.9,
                )

        # Check for blocked content
        for pattern in self.blocked_patterns:
            if pattern.search(message):
                logger.warning(
                    f"Blocked content detected: {pattern.pattern[:50]}...",
                    extra={"pattern": pattern.pattern},
                )
                return ValidationResult(
                    is_safe=False,
                    reason="Message contains blocked content",
                    risk_score=0.8,
                )

        # Check for suspicious character patterns
        risk_score = self._calculate_risk_score(message)
        if risk_score > 0.7:
            logger.warning(
                f"High risk score: {risk_score}",
                extra={"message_length": len(message)},
            )
            return ValidationResult(
                is_safe=False,
                reason="Message flagged as potentially harmful",
                risk_score=risk_score,
            )

        return ValidationResult(is_safe=True, risk_score=risk_score)

    def _calculate_risk_score(self, message: str) -> float:
        """Calculate a risk score based on message characteristics.

        Args:
            message: The message to analyze

        Returns:
            Risk score between 0.0 and 1.0
        """
        score = 0.0
        message_lower = message.lower()

        # Excessive special characters
        special_chars = sum(1 for c in message if not c.isalnum() and not c.isspace())
        if special_chars > len(message) * 0.3:
            score += 0.2

        # Multiple newlines (potential formatting attack)
        if message.count("\n") > 20:
            score += 0.15

        # Excessive repetition
        words = message_lower.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score += 0.2

        # Suspicious keywords
        suspicious_keywords = [
            "system:",
            "assistant:",
            "user:",
            "[inst]",
            "[/inst]",
            "<|",
            "|>",
            "```",
        ]
        for keyword in suspicious_keywords:
            if keyword in message_lower:
                score += 0.1

        # Base64-like strings (potential encoded attacks)
        if re.search(r"[A-Za-z0-9+/]{50,}={0,2}", message):
            score += 0.15

        return min(score, 1.0)

    def sanitize(self, message: str) -> str:
        """Sanitize a message by removing potentially harmful content.

        This is a softer approach than blocking - use when you want to
        clean input rather than reject it outright.

        Args:
            message: The message to sanitize

        Returns:
            Sanitized message
        """
        if not message:
            return message

        # Truncate if too long
        if len(message) > self.max_length:
            message = message[: self.max_length]

        # Remove excessive whitespace
        message = re.sub(r"\n{3,}", "\n\n", message)
        message = re.sub(r" {3,}", "  ", message)

        # Remove control characters (except newline, tab)
        message = "".join(
            c for c in message if c.isprintable() or c in "\n\t"
        )

        return message.strip()
