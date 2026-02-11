"""Output filtering for AI responses.

Filters PII and sensitive information from responses before returning to users.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FilteredResponse:
    """Result of output filtering."""

    text: str
    sources: list[str] = field(default_factory=list)
    pii_removed: int = 0
    filtered_types: list[str] = field(default_factory=list)


# PII patterns with named groups for identification
PII_PATTERNS = {
    "phone_us": r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "phone_intl": r"\b\+[1-9][0-9]{7,14}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    "ipv6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    "ssn": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
    "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
    "mac_address": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
}

# Replacement text for each PII type
PII_REPLACEMENTS = {
    "phone_us": "[PHONE REDACTED]",
    "phone_intl": "[PHONE REDACTED]",
    "email": "[EMAIL REDACTED]",
    "ipv4": "[IP REDACTED]",
    "ipv6": "[IP REDACTED]",
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CARD REDACTED]",
    "mac_address": "[MAC REDACTED]",
}

# Sensitive keywords that might indicate leaked credentials or secrets
SENSITIVE_KEYWORDS = [
    r"password\s*[:=]\s*\S+",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"secret[_-]?key\s*[:=]\s*\S+",
    r"access[_-]?token\s*[:=]\s*\S+",
    r"bearer\s+[A-Za-z0-9._-]+",
]


class OutputFilter:
    """Filter sensitive information from AI responses.

    Removes or redacts:
    - Phone numbers (US and international)
    - Email addresses
    - IP addresses (v4 and v6)
    - Social Security Numbers
    - Credit card numbers
    - MAC addresses
    - Credentials/API keys
    """

    def __init__(
        self,
        filter_pii: bool = True,
        filter_credentials: bool = True,
        custom_patterns: Optional[dict[str, str]] = None,
    ):
        """Initialize the output filter.

        Args:
            filter_pii: Whether to filter PII patterns
            filter_credentials: Whether to filter credential patterns
            custom_patterns: Additional patterns to filter (name -> pattern)
        """
        self.filter_pii = filter_pii
        self.filter_credentials = filter_credentials

        # Compile PII patterns
        self.pii_patterns = {
            name: re.compile(pattern)
            for name, pattern in PII_PATTERNS.items()
        }

        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.pii_patterns[name] = re.compile(pattern)
                PII_REPLACEMENTS[name] = f"[{name.upper()} REDACTED]"

        # Compile sensitive keyword patterns
        self.sensitive_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SENSITIVE_KEYWORDS
        ]

    def apply(
        self,
        response: str,
        sources: Optional[list[str]] = None,
    ) -> FilteredResponse:
        """Apply output filtering to a response.

        Args:
            response: The AI response text to filter
            sources: List of source references

        Returns:
            FilteredResponse with filtered text and metadata
        """
        if not response:
            return FilteredResponse(text="", sources=sources or [])

        filtered_text = response
        pii_count = 0
        filtered_types = []

        # Filter PII patterns
        if self.filter_pii:
            for pii_type, pattern in self.pii_patterns.items():
                matches = pattern.findall(filtered_text)
                if matches:
                    pii_count += len(matches)
                    filtered_types.append(pii_type)
                    replacement = PII_REPLACEMENTS.get(
                        pii_type, "[REDACTED]"
                    )
                    filtered_text = pattern.sub(replacement, filtered_text)

                    logger.info(
                        f"Filtered {len(matches)} {pii_type} instances",
                        extra={"pii_type": pii_type, "count": len(matches)},
                    )

        # Filter credentials/secrets
        if self.filter_credentials:
            for pattern in self.sensitive_patterns:
                matches = pattern.findall(filtered_text)
                if matches:
                    pii_count += len(matches)
                    filtered_types.append("credential")
                    filtered_text = pattern.sub("[CREDENTIAL REDACTED]", filtered_text)

                    logger.warning(
                        f"Filtered {len(matches)} potential credentials",
                        extra={"count": len(matches)},
                    )

        # Filter sources too
        filtered_sources = sources or []
        if sources:
            filtered_sources = [
                self._filter_source(source) for source in sources
            ]

        if pii_count > 0:
            logger.info(
                f"Total PII filtered: {pii_count}",
                extra={
                    "pii_count": pii_count,
                    "types": filtered_types,
                },
            )

        return FilteredResponse(
            text=filtered_text,
            sources=filtered_sources,
            pii_removed=pii_count,
            filtered_types=list(set(filtered_types)),
        )

    def _filter_source(self, source: str) -> str:
        """Filter PII from a source reference.

        Args:
            source: Source reference string

        Returns:
            Filtered source string
        """
        filtered = source
        for pii_type, pattern in self.pii_patterns.items():
            replacement = PII_REPLACEMENTS.get(pii_type, "[REDACTED]")
            filtered = pattern.sub(replacement, filtered)
        return filtered

    def check_for_pii(self, text: str) -> dict[str, int]:
        """Check text for PII without filtering (for analysis).

        Args:
            text: Text to analyze

        Returns:
            Dictionary of PII type -> count found
        """
        results = {}
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[pii_type] = len(matches)
        return results
