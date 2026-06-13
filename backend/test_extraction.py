"""Unit tests for Slack message extraction."""

from unittest.mock import MagicMock

import pytest

from backend.extraction import ExtractionError, extract_employee_info
from backend.schemas import ExtractionResult

# 10 realistic fake Slack messages for extraction testing.
FAKE_SLACK_MESSAGES = [
    (
        "Hey everyone! I'm Priya Sharma, just joined as a Senior Backend Engineer on the Platform team. "
        "Previously at Stripe. Excited to work on our payment microservices and Kubernetes infra."
    ),
    (
        "Quick intro — Marcus Chen, Staff ML Engineer. I lead the recommendation systems project "
        "and have deep experience with PyTorch, TensorFlow, and MLOps pipelines."
    ),
    (
        "Can someone help debug this Redis connection timeout in the checkout service? "
        "I've tried restarting the pods but no luck."
    ),
    (
        "Shoutout to @Lisa Park for crushing the GraphQL migration last sprint! "
        "She's our go-to for API design and has been driving the Apollo Federation rollout."
    ),
    (
        "I'm Alex Rivera, Product Designer on the Growth team. I specialize in Figma prototyping, "
        "design systems, and user research. Happy to review any onboarding flows."
    ),
    (
        "Team standup reminder: the Phoenix data pipeline deploy is at 3pm today. "
        "James and I will be on call if anything breaks."
    ),
    (
        "Fun fact: outside of work I do competitive rock climbing. "
        "Also happy to mentor anyone interested in Rust or systems programming — "
        "that's been my focus for the past 5 years at Cloudflare."
    ),
    (
        "Welcome @Nina Okonkwo! She's joining as Engineering Manager for the Mobile team. "
        "She ran iOS and Android teams at Spotify and is big on async communication."
    ),
    (
        "Does anyone have experience with SOC 2 compliance audits? "
        "We need someone who understands security controls and AWS IAM policies for the Atlas project."
    ),
    (
        "Lunch poll 🍕 or 🌮? Also unrelated but I'm free after 2pm PST if anyone "
        "wants to pair on the Elasticsearch indexing bug."
    ),
]

EXPECTED_EXTRACTIONS = [
    {
        "person": "Priya Sharma",
        "role": "Senior Backend Engineer",
        "company": None,
        "skills": ["Kubernetes", "microservices"],
        "projects": ["payment microservices"],
        "confidence": 0.92,
    },
    {
        "person": "Marcus Chen",
        "role": "Staff ML Engineer",
        "company": None,
        "skills": ["PyTorch", "TensorFlow", "MLOps"],
        "projects": ["recommendation systems"],
        "confidence": 0.95,
    },
    {
        "person": "unknown",
        "role": None,
        "company": None,
        "skills": ["Redis", "Kubernetes"],
        "projects": ["checkout service"],
        "confidence": 0.25,
    },
    {
        "person": "Lisa Park",
        "role": None,
        "company": None,
        "skills": ["GraphQL", "API design", "Apollo Federation"],
        "projects": ["GraphQL migration", "Apollo Federation rollout"],
        "confidence": 0.88,
    },
    {
        "person": "Alex Rivera",
        "role": "Product Designer",
        "company": None,
        "skills": ["Figma", "design systems", "user research"],
        "projects": [],
        "confidence": 0.93,
    },
    {
        "person": "James",
        "role": None,
        "company": None,
        "skills": [],
        "projects": ["Phoenix data pipeline"],
        "confidence": 0.4,
    },
    {
        "person": "unknown",
        "role": None,
        "company": "Cloudflare",
        "skills": ["Rust", "systems programming"],
        "projects": [],
        "confidence": 0.7,
    },
    {
        "person": "Nina Okonkwo",
        "role": "Engineering Manager",
        "company": "Spotify",
        "skills": ["iOS", "Android"],
        "projects": ["Mobile team"],
        "confidence": 0.94,
    },
    {
        "person": "unknown",
        "role": None,
        "company": None,
        "skills": ["SOC 2", "AWS IAM", "security controls"],
        "projects": ["Atlas project"],
        "confidence": 0.35,
    },
    {
        "person": "unknown",
        "role": None,
        "company": None,
        "skills": ["Elasticsearch"],
        "projects": [],
        "confidence": 0.2,
    },
]


def _mock_model_response(payload: dict) -> MagicMock:
    """Build a mock GenerativeModel that returns JSON text."""
    import json

    model = MagicMock()
    response = MagicMock()
    response.text = json.dumps(payload)
    model.generate_content.return_value = response
    return model


class TestExtractEmployeeInfo:
    """Tests for extract_employee_info with mocked Vertex AI."""

    @pytest.mark.parametrize("message,expected", zip(FAKE_SLACK_MESSAGES, EXPECTED_EXTRACTIONS))
    def test_extracts_all_fake_messages(self, message: str, expected: dict):
        """Each fake Slack message should produce a valid ExtractionResult."""
        model = _mock_model_response(expected)
        result = extract_employee_info(message, model)

        assert isinstance(result, ExtractionResult)
        assert result.person == expected["person"]
        assert result.role == expected["role"]
        assert result.company == expected["company"]
        assert result.confidence == expected["confidence"]
        assert set(result.skills) == set(expected["skills"])
        assert set(result.projects) == set(expected["projects"])
        model.generate_content.assert_called_once()

    def test_empty_message_raises(self):
        """Empty messages should be rejected before calling Vertex AI."""
        model = MagicMock()
        with pytest.raises(ExtractionError, match="Message is empty"):
            extract_employee_info("   ", model)
        model.generate_content.assert_not_called()

    def test_vertex_ai_failure_raises_extraction_error(self):
        """Vertex AI exceptions should be wrapped in ExtractionError."""
        model = MagicMock()
        model.generate_content.side_effect = RuntimeError("API quota exceeded")

        with pytest.raises(ExtractionError, match="Vertex AI extraction failed") as exc_info:
            extract_employee_info(FAKE_SLACK_MESSAGES[0], model)

        assert "API quota exceeded" in exc_info.value.detail

    def test_empty_vertex_response_raises(self):
        """An empty model response should raise ExtractionError."""
        model = MagicMock()
        model.generate_content.return_value = MagicMock(text="")

        with pytest.raises(ExtractionError, match="empty response"):
            extract_employee_info(FAKE_SLACK_MESSAGES[0], model)

    def test_invalid_json_raises(self):
        """Malformed JSON from the model should raise ExtractionError."""
        model = MagicMock()
        model.generate_content.return_value = MagicMock(text="not valid json {{{")

        with pytest.raises(ExtractionError, match="Failed to parse"):
            extract_employee_info(FAKE_SLACK_MESSAGES[0], model)

    def test_whitespace_only_message_raises(self):
        """Whitespace-only input should not reach the model."""
        model = MagicMock()
        with pytest.raises(ExtractionError):
            extract_employee_info("\n\t  ", model)


class TestFakeMessageCorpus:
    """Sanity checks on the test message corpus itself."""

    def test_corpus_has_ten_messages(self):
        """Ensure the test suite includes exactly 10 fake messages."""
        assert len(FAKE_SLACK_MESSAGES) == 10
        assert len(EXPECTED_EXTRACTIONS) == 10

    def test_messages_are_nonempty(self):
        """All fake messages should contain meaningful text."""
        for msg in FAKE_SLACK_MESSAGES:
            assert len(msg.strip()) > 20
