"""Hallucination detection for RAG responses.

Verifies that generated responses are grounded in retrieved context.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """A claim extracted from a response."""

    text: str
    is_grounded: bool = False
    supporting_source: Optional[str] = None
    confidence: float = 0.0


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    is_grounded: bool
    confidence_score: float
    claims: list[Claim] = field(default_factory=list)
    grounded_count: int = 0
    ungrounded_count: int = 0
    explanation: str = ""


class HallucinationDetector:
    """Detect hallucinations in AI-generated responses.

    Compares generated claims against retrieved context to identify
    statements not supported by the source documents.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_grounded_ratio: float = 0.8,
    ):
        """Initialize the hallucination detector.

        Args:
            similarity_threshold: Min similarity for claim to be considered grounded
            min_grounded_ratio: Min ratio of grounded claims for response to pass
        """
        self.similarity_threshold = similarity_threshold
        self.min_grounded_ratio = min_grounded_ratio

    def check(
        self,
        response: str,
        context_docs: list[str],
        sources: Optional[list[str]] = None,
    ) -> HallucinationResult:
        """Check a response for hallucinations.

        Args:
            response: The generated response text
            context_docs: Retrieved context documents
            sources: Optional source references

        Returns:
            HallucinationResult with grounding analysis
        """
        if not response or not context_docs:
            return HallucinationResult(
                is_grounded=False,
                confidence_score=0.0,
                explanation="Empty response or context",
            )

        # Extract claims from response
        claims = self._extract_claims(response)

        if not claims:
            return HallucinationResult(
                is_grounded=True,
                confidence_score=1.0,
                explanation="No verifiable claims found",
            )

        # Combine context for checking
        combined_context = " ".join(context_docs).lower()

        # Check each claim against context
        grounded_claims = []
        for claim in claims:
            is_grounded, confidence, source = self._check_claim(
                claim.text, combined_context, sources
            )
            claim.is_grounded = is_grounded
            claim.confidence = confidence
            claim.supporting_source = source
            grounded_claims.append(claim)

        # Calculate overall grounding
        grounded_count = sum(1 for c in grounded_claims if c.is_grounded)
        total_claims = len(grounded_claims)
        grounded_ratio = grounded_count / total_claims if total_claims > 0 else 0

        is_grounded = grounded_ratio >= self.min_grounded_ratio
        confidence_score = grounded_ratio

        # Build explanation
        if is_grounded:
            explanation = f"{grounded_count}/{total_claims} claims are grounded in context"
        else:
            ungrounded = [c.text for c in grounded_claims if not c.is_grounded]
            explanation = (
                f"Only {grounded_count}/{total_claims} claims grounded. "
                f"Ungrounded: {ungrounded[:3]}"
            )

        return HallucinationResult(
            is_grounded=is_grounded,
            confidence_score=confidence_score,
            claims=grounded_claims,
            grounded_count=grounded_count,
            ungrounded_count=total_claims - grounded_count,
            explanation=explanation,
        )

    def _extract_claims(self, text: str) -> list[Claim]:
        """Extract verifiable claims from text.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Skip questions and hedged statements
            if sentence.endswith('?'):
                continue
            if any(hedge in sentence.lower() for hedge in [
                'might', 'maybe', 'perhaps', 'possibly', 'could be',
                'i think', 'i believe', 'it seems'
            ]):
                continue

            # Look for factual claims (contains numbers, names, or definitive statements)
            has_number = bool(re.search(r'\d+', sentence))
            has_definitive = any(word in sentence.lower() for word in [
                'is', 'are', 'was', 'were', 'has', 'have', 'will',
                'always', 'never', 'must', 'should'
            ])

            if has_number or has_definitive:
                claims.append(Claim(text=sentence))

        return claims

    def _check_claim(
        self,
        claim: str,
        context: str,
        sources: Optional[list[str]],
    ) -> tuple[bool, float, Optional[str]]:
        """Check if a claim is grounded in context.

        Args:
            claim: The claim text
            context: Combined context text
            sources: Optional source references

        Returns:
            Tuple of (is_grounded, confidence, supporting_source)
        """
        claim_lower = claim.lower()

        # Extract key terms from claim
        key_terms = self._extract_key_terms(claim_lower)

        if not key_terms:
            # No key terms to verify - consider grounded
            return True, 0.8, None

        # Check how many key terms appear in context
        matches = sum(1 for term in key_terms if term in context)
        match_ratio = matches / len(key_terms)

        # Check for number consistency
        claim_numbers = set(re.findall(r'\d+\.?\d*', claim))
        context_numbers = set(re.findall(r'\d+\.?\d*', context))
        numbers_match = claim_numbers.issubset(context_numbers) if claim_numbers else True

        # Calculate confidence
        confidence = match_ratio * 0.7 + (0.3 if numbers_match else 0)

        is_grounded = confidence >= self.similarity_threshold

        # Find supporting source if grounded
        supporting_source = None
        if is_grounded and sources:
            for source in sources:
                if any(term in source.lower() for term in key_terms[:2]):
                    supporting_source = source
                    break

        return is_grounded, confidence, supporting_source

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for matching.

        Args:
            text: Text to extract terms from

        Returns:
            List of key terms
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'also', 'now', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'any', 'this', 'that', 'these', 'those', 'it', 'its'
        }

        # Extract words
        words = re.findall(r'\b[a-z]+\b', text)

        # Filter and return key terms
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms[:10]  # Limit to top 10 terms
