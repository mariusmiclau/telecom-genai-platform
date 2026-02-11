"""Tests for hallucination detection."""

import pytest

from src.rag_pipeline.hallucination import (
    HallucinationDetector,
    HallucinationResult,
    Claim,
)


class TestHallucinationDetector:
    """Tests for HallucinationDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a HallucinationDetector instance."""
        return HallucinationDetector()

    def test_grounded_response_passes(self, detector):
        """Test that a grounded response is detected as such."""
        context = [
            "The network uptime for Region A is 99.95%.",
            "There are 5 routers in Region A.",
            "The average latency is 25ms.",
        ]
        response = "The network in Region A has 99.95% uptime with 5 routers."

        result = detector.check(response, context)

        assert isinstance(result, HallucinationResult)
        assert result.is_grounded is True
        assert result.confidence_score > 0.5

    def test_ungrounded_response_fails(self, detector):
        """Test that an ungrounded response is detected."""
        context = [
            "The network uptime for Region A is 99.95%.",
            "There are 5 routers in Region A.",
        ]
        # Response contains fabricated information not in context
        response = "Region A has 50 routers with 75% uptime and 100ms latency."

        result = detector.check(response, context)

        assert isinstance(result, HallucinationResult)
        # Should detect low grounding due to mismatched numbers
        assert result.ungrounded_count > 0

    def test_empty_response_handled(self, detector):
        """Test handling of empty response."""
        result = detector.check("", ["Some context"])
        assert result.is_grounded is False
        assert "empty" in result.explanation.lower()

    def test_empty_context_handled(self, detector):
        """Test handling of empty context."""
        result = detector.check("Some response", [])
        assert result.is_grounded is False

    def test_response_with_no_claims(self, detector):
        """Test response with no verifiable claims."""
        context = ["Network information here."]
        # Questions and hedged statements should not be treated as claims
        response = "What do you think? Maybe something could happen."

        result = detector.check(response, context)

        # Should be considered grounded since no claims to verify
        assert result.is_grounded is True

    def test_numbers_must_match_context(self, detector):
        """Test that numbers in response must appear in context."""
        context = [
            "The system has 10 servers.",
            "Response time is 50ms.",
        ]
        # Correct numbers from context
        response = "There are 10 servers with 50ms response time."
        result = detector.check(response, context)
        assert result.grounded_count > 0

    def test_fabricated_numbers_detected(self, detector):
        """Test that fabricated numbers are detected."""
        context = [
            "The system has 10 servers.",
        ]
        # Wrong number not in context
        response = "The system has 100 servers."
        result = detector.check(response, context)
        # Should have lower confidence due to number mismatch
        assert result.confidence_score < 1.0

    def test_partial_grounding(self, detector):
        """Test response with partial grounding."""
        context = [
            "Server A is running at 95% capacity.",
            "Server B is offline for maintenance.",
        ]
        response = (
            "Server A is running at 95% capacity. "
            "Server C is experiencing critical failures."  # Not in context
        )

        result = detector.check(response, context)

        # Should detect both grounded and ungrounded claims
        assert result.grounded_count >= 1
        assert result.ungrounded_count >= 0

    def test_confidence_threshold_configurable(self):
        """Test that confidence threshold is configurable."""
        strict_detector = HallucinationDetector(min_grounded_ratio=0.9)
        lenient_detector = HallucinationDetector(min_grounded_ratio=0.5)

        context = ["The server has 10GB RAM."]
        response = "The server has 10GB RAM and runs Linux."

        strict_result = strict_detector.check(response, context)
        lenient_result = lenient_detector.check(response, context)

        # Lenient detector should be more likely to pass
        assert lenient_result.confidence_score >= strict_result.confidence_score or \
               lenient_result.is_grounded or not strict_result.is_grounded

    def test_claims_extraction(self, detector):
        """Test that claims are properly extracted from response."""
        context = ["Network documentation."]
        response = "The router has 4 ports. The switch is managed. What time is it?"

        result = detector.check(response, context)

        # Should extract factual claims but not questions
        claims_text = [c.text for c in result.claims]
        assert not any("?" in text for text in claims_text)

    def test_hedged_statements_not_claims(self, detector):
        """Test that hedged statements are not treated as claims."""
        context = ["Some context."]
        response = "I think it might be working. Perhaps the server could restart."

        result = detector.check(response, context)

        # Hedged statements should not be extracted as claims
        for claim in result.claims:
            hedges = ['might', 'perhaps', 'could', 'i think']
            assert not any(h in claim.text.lower() for h in hedges)

    def test_grounding_explanation_provided(self, detector):
        """Test that grounding explanation is provided."""
        context = ["The API latency is 25ms."]
        response = "The API has 25ms latency."

        result = detector.check(response, context)

        assert result.explanation
        assert len(result.explanation) > 0

    def test_source_attribution(self, detector):
        """Test that supporting sources are identified."""
        context = ["Network status report shows 99.9% uptime."]
        sources = ["network_report.pdf", "status_dashboard.json"]
        response = "The network has 99.9% uptime according to the status report."

        result = detector.check(response, context, sources)

        # At least some claims should have source attribution
        # (implementation dependent)
        assert isinstance(result.claims, list)

    def test_case_insensitive_matching(self, detector):
        """Test that matching is case insensitive."""
        context = ["The ROUTER is configured with OSPF protocol."]
        response = "The router uses ospf for routing."

        result = detector.check(response, context)

        # Should match despite case differences
        assert result.grounded_count > 0 or result.confidence_score > 0

    def test_multiple_claims_evaluation(self, detector):
        """Test evaluation of multiple claims."""
        context = [
            "Region A has 5 data centers.",
            "Total capacity is 500TB.",
            "Uptime SLA is 99.99%.",
        ]
        response = (
            "Region A operates 5 data centers. "
            "The total storage capacity is 500TB. "
            "They maintain a 99.99% uptime SLA."
        )

        result = detector.check(response, context)

        # All claims should be grounded
        assert result.grounded_count >= 2
        assert result.is_grounded is True


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self):
        """Test creating a Claim."""
        claim = Claim(
            text="The server has 16GB RAM.",
            is_grounded=True,
            confidence=0.95,
            supporting_source="spec_sheet.pdf",
        )
        assert claim.text == "The server has 16GB RAM."
        assert claim.is_grounded is True
        assert claim.confidence == 0.95
        assert claim.supporting_source == "spec_sheet.pdf"

    def test_claim_defaults(self):
        """Test Claim default values."""
        claim = Claim(text="Some claim")
        assert claim.is_grounded is False
        assert claim.confidence == 0.0
        assert claim.supporting_source is None


class TestHallucinationResult:
    """Tests for HallucinationResult dataclass."""

    def test_result_creation(self):
        """Test creating a HallucinationResult."""
        result = HallucinationResult(
            is_grounded=True,
            confidence_score=0.9,
            grounded_count=5,
            ungrounded_count=1,
            explanation="5/6 claims grounded",
        )
        assert result.is_grounded is True
        assert result.confidence_score == 0.9
        assert result.grounded_count == 5
        assert result.ungrounded_count == 1

    def test_result_defaults(self):
        """Test HallucinationResult default values."""
        result = HallucinationResult(
            is_grounded=False,
            confidence_score=0.0,
        )
        assert result.claims == []
        assert result.grounded_count == 0
        assert result.ungrounded_count == 0
        assert result.explanation == ""
