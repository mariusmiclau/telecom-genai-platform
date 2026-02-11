"""Hallucination detection and response grounding.

Ensures AI responses are grounded in source documents and tool data.
Critical for telecom operations where hallucinated configs could cause outages.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.7"))


@dataclass
class GroundingResult:
    """Result of hallucination/grounding check."""

    is_grounded: bool
    grounding_score: float
    ungrounded_claims: list[str]
    recommendation: str


class HallucinationDetector:
    """Detect hallucinations by checking response grounding against sources.

    Strategy:
    1. Extract factual claims from the response
    2. Check each claim against retrieved source documents
    3. Check claims against tool call results
    4. Flag ungrounded claims
    5. Score overall grounding confidence

    In telecom context, we are especially strict about:
    - Network element identifiers (node IDs, interface names)
    - Numerical values (throughput, latency, error rates)
    - Configuration parameters
    - SLA thresholds and compliance status
    """

    def __init__(self, threshold: float = HALLUCINATION_THRESHOLD):
        self.threshold = threshold

    def check_grounding(
        self,
        response: str,
        source_documents: list[dict],
        tool_results: list[dict],
    ) -> GroundingResult:
        """Check if response is grounded in provided sources.

        Args:
            response: The AI-generated response text
            source_documents: Retrieved RAG documents
            tool_results: Data from tool calls

        Returns:
            GroundingResult with score and flagged claims
        """
        claims = self._extract_claims(response)

        if not claims:
            return GroundingResult(
                is_grounded=True,
                grounding_score=1.0,
                ungrounded_claims=[],
                recommendation="No factual claims detected",
            )

        # Build evidence corpus from sources and tools
        evidence = self._build_evidence_corpus(source_documents, tool_results)

        # Check each claim against evidence
        grounded_count = 0
        ungrounded = []

        for claim in claims:
            if self._is_claim_supported(claim, evidence):
                grounded_count += 1
            else:
                ungrounded.append(claim)

        score = grounded_count / len(claims) if claims else 1.0
        is_grounded = score >= self.threshold

        if not is_grounded:
            logger.warning(
                "Response failed grounding check",
                extra={
                    "grounding_score": score,
                    "threshold": self.threshold,
                    "ungrounded_claims": ungrounded,
                },
            )

        recommendation = self._get_recommendation(score, ungrounded)

        return GroundingResult(
            is_grounded=is_grounded,
            grounding_score=score,
            ungrounded_claims=ungrounded,
            recommendation=recommendation,
        )

    def _extract_claims(self, response: str) -> list[str]:
        """Extract verifiable factual claims from response.

        Focuses on telecom-specific claims:
        - Numerical values (latency, throughput, availability %)
        - Status assertions (operational, degraded, down)
        - Configuration references
        - SLA compliance statements
        """
        claims = []
        sentences = response.split(".")

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Flag sentences with numerical claims
            if any(char.isdigit() for char in sentence):
                claims.append(sentence)

            # Flag status assertions
            status_keywords = [
                "operational", "degraded", "down", "healthy",
                "compliant", "breached", "exceeded", "below",
                "active", "inactive", "failed", "restored",
            ]
            if any(kw in sentence.lower() for kw in status_keywords):
                claims.append(sentence)

            # Flag configuration references
            config_keywords = [
                "configured", "setting", "threshold", "parameter",
                "policy", "rule", "limit", "capacity",
            ]
            if any(kw in sentence.lower() for kw in config_keywords):
                claims.append(sentence)

        return list(set(claims))  # Deduplicate

    def _build_evidence_corpus(
        self,
        source_documents: list[dict],
        tool_results: list[dict],
    ) -> list[str]:
        """Combine all evidence sources into searchable corpus."""
        evidence = []

        for doc in source_documents:
            content = doc.get("content", doc.get("text", ""))
            if content:
                evidence.append(content)

        for result in tool_results:
            # Flatten tool results into evidence strings
            evidence.append(str(result))

        return evidence

    def _is_claim_supported(self, claim: str, evidence: list[str]) -> bool:
        """Check if a claim is supported by evidence.

        Uses keyword overlap as a baseline heuristic.
        In production, this would use an NLI model for semantic entailment.
        """
        claim_tokens = set(claim.lower().split())
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "has", "have",
            "been", "be", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they",
            "of", "in", "to", "for", "with", "on", "at", "from",
            "by", "about", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "and", "but", "or",
        }
        claim_keywords = claim_tokens - stop_words

        if not claim_keywords:
            return True  # No substantive keywords to check

        for evidence_text in evidence:
            evidence_tokens = set(evidence_text.lower().split())
            overlap = claim_keywords & evidence_tokens
            overlap_ratio = len(overlap) / len(claim_keywords)

            if overlap_ratio >= 0.5:
                return True

        return False

    def _get_recommendation(self, score: float, ungrounded: list[str]) -> str:
        """Generate recommendation based on grounding analysis."""
        if score >= 0.9:
            return "Response well-grounded in sources. Safe to present."
        elif score >= self.threshold:
            return (
                f"Response mostly grounded. {len(ungrounded)} claim(s) "
                "lack direct source support. Consider adding caveats."
            )
        elif score >= 0.4:
            return (
                "Response partially grounded. Recommend human review "
                "before presenting to user. Ungrounded claims should be "
                "removed or flagged as uncertain."
            )
        else:
            return (
                "Response poorly grounded. Recommend escalation to "
                "human operator. Do not present as authoritative."
            )
