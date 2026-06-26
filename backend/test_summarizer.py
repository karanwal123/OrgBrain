"""Tests for channel summarization."""

import json
from unittest.mock import MagicMock

import pytest

from backend.channel_history import (
    ChannelHistoryError,
    parse_summarize_args,
    parse_timeframe,
    resolve_channel_id,
)
from backend.schemas import (
    ActionItem,
    ChannelSummary,
    Decision,
    ProblemSolution,
    TimelineEvent,
)
from backend.summarizer import (
    _parse_channel_summary,
    _messages_to_prompt_json,
    apply_name_resolution,
    fallback_channel_summary,
    format_block_kit,
    format_loading_modal,
    format_timeline_card,
    resolve_person_name,
    summarize_channel,
)
from backend.channel_history import build_user_lookup


class TestParseSummarizeArgs:
    """Tests for /summarize argument parsing."""

    def test_channel_and_timeframe(self):
        ch, tf = parse_summarize_args("#hackathon- last 2 days")
        assert ch == "hackathon-"
        assert tf == "last 2 days"

    def test_channel_only_defaults_timeframe(self):
        ch, tf = parse_summarize_args("#payment-migration")
        assert ch == "payment-migration"
        assert tf == "last 24 hours"

    def test_timeframe_only(self):
        ch, tf = parse_summarize_args("last 24 hours")
        assert ch is None
        assert tf == "last 24 hours"

    def test_empty_defaults(self):
        ch, tf = parse_summarize_args("")
        assert ch is None
        assert tf == "last 24 hours"


class TestParseTimeframe:
    """Tests for timeframe parsing."""

    def test_last_2_days(self):
        oldest, label = parse_timeframe("last 2 days")
        assert "2 days" in label
        assert oldest > 0

    def test_today(self):
        _, label = parse_timeframe("today")
        assert label == "Today"

    def test_default(self):
        _, label = parse_timeframe("")
        assert label == "Last 24 hours"


class TestResolveChannelId:
    """Tests for channel ID resolution."""

    def test_resolve_by_name(self):
        client = MagicMock()
        client.conversations_list.return_value = {
            "channels": [{"id": "C123", "name": "hackathon-"}],
            "response_metadata": {},
        }
        cid, name = resolve_channel_id(client, "hackathon-", "C999")
        assert cid == "C123"
        assert name == "#hackathon-"

    def test_not_found_raises(self):
        client = MagicMock()
        client.conversations_list.return_value = {
            "channels": [],
            "response_metadata": {},
        }
        with pytest.raises(ChannelHistoryError, match="not found"):
            resolve_channel_id(client, "missing", "C999")


class TestFormatBlockKit:
    """Tests for Block Kit modal builder."""

    def test_loading_modal(self):
        view = format_loading_modal("#hackathon-", "Last 2 days")
        assert view["type"] == "modal"
        assert "Summarizing" in view["title"]["text"]

    def test_full_summary_modal(self):
        summary = ChannelSummary(
            channel="#hackathon-",
            timeframe="Last 2 days",
            health="healthy",
            participant_count=5,
            decision_count=2,
            timeline=[
                TimelineEvent(
                    type="decision",
                    action_type="Decision Made",
                    text="Approved async replication",
                    speaker="Priya",
                    timestamp="Day 1, 10:30",
                )
            ],
            decisions=[
                Decision(decision="Use async replication", decided_by="Priya", approved_by="Rahul")
            ],
            problems=[
                ProblemSolution(
                    problem="DB timeout",
                    reported_by="Rahul",
                    solution="Async replication",
                    fixed_by="Aditya",
                    impact="50% latency reduction",
                )
            ],
            action_items=[
                ActionItem(item="Setup staging env", owner="Vikram", due="Tomorrow", status="pending"),
                ActionItem(item="DB migration", owner="Aditya", status="done"),
            ],
            narrative="Team made good progress on the payment migration.",
        )
        view = format_block_kit(summary)
        body = json.dumps(view)
        assert view["type"] == "modal"
        assert "Timeline" in body
        assert "Decisions Made" in body
        assert "Problems & Solutions" in body
        assert "Action Items" in body
        assert "OrgBrain" in body
        assert len(view["blocks"]) <= 100

    def test_timeline_capped_at_five(self):
        summary = ChannelSummary(
            channel="#test",
            timeframe="Today",
            timeline=[
                TimelineEvent(
                    type="event",
                    action_type=f"Event {i}",
                    text=f"text {i}",
                    speaker="User",
                    timestamp="now",
                )
                for i in range(10)
            ],
        )
        view = format_block_kit(summary)
        timeline_sections = [
            b for b in view["blocks"]
            if b.get("type") == "section" and "Event" in b.get("text", {}).get("text", "")
        ]
        assert len(timeline_sections) <= 5

    def test_timeline_card(self):
        event = TimelineEvent(
            type="decision",
            action_type="Decision Made",
            text="Approved approach",
            speaker="Priya",
            timestamp="10:30",
        )
        card = format_timeline_card(event)
        assert "📌" in card["text"]["text"]
        assert "Priya" in card["text"]["text"]


class TestSummarizeChannel:
    """Tests for Gemini summarization with mocks."""

    def test_parse_channel_summary(self):
        raw = json.dumps(
            {
                "channel": "#hackathon-",
                "timeframe": "Last 2 days",
                "health": "healthy",
                "participant_count": 3,
                "decision_count": 1,
                "timeline": [],
                "decisions": [],
                "problems": [],
                "action_items": [],
                "narrative": "Good progress.",
            }
        )
        result = _parse_channel_summary(raw, "#hackathon-", "Last 2 days")
        assert result.health == "healthy"
        assert result.narrative == "Good progress."

    def test_parse_channel_summary_with_wrapped_json(self):
        raw = (
            "Summary:\n"
            "```json\n"
            + json.dumps(
                {
                    "channel": "#hackathon-",
                    "timeframe": "Last 24 hours",
                    "health": "warning",
                    "participant_count": 2,
                    "decision_count": 0,
                    "timeline": [],
                    "decisions": [],
                    "problems": [],
                    "action_items": [],
                    "narrative": "Detailed extraction unavailable — try a shorter timeframe.",
                }
            )
            + "\n```\nExtra notes ignored."
        )

        result = _parse_channel_summary(raw, "#hackathon-", "Last 24 hours")
        assert result.health == "warning"
        assert "shorter timeframe" in result.narrative

    def test_summarize_with_mock_gemini(self):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                {
                    "channel": "#hackathon-",
                    "timeframe": "Last 2 days",
                    "health": "warning",
                    "participant_count": 2,
                    "decision_count": 0,
                    "timeline": [
                        {
                            "type": "problem",
                            "action_type": "Problem Reported",
                            "text": "DB timeout",
                            "speaker": "Rahul",
                            "timestamp": "14:00",
                        }
                    ],
                    "decisions": [],
                    "problems": [],
                    "action_items": [],
                    "narrative": "Some issues reported.",
                }
            )
        )
        messages = [
            {"time_label": "Jun 13, 10:00", "user_name": "Rahul", "text": "DB timeout issue"},
        ]
        summary = summarize_channel(messages, "#hackathon-", "Last 2 days", mock_model)
        assert summary.health == "warning"
        assert len(summary.timeline) == 1

    def test_fallback_on_empty_messages(self):
        summary = fallback_channel_summary("#test", "Today", [])
        assert summary.participant_count == 0

    def test_repeated_summary_request_uses_cache(self):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                {
                    "channel": "#hackathon-",
                    "timeframe": "Last 2 days",
                    "health": "healthy",
                    "participant_count": 1,
                    "decision_count": 0,
                    "timeline": [],
                    "decisions": [],
                    "problems": [],
                    "action_items": [],
                    "narrative": "Good progress.",
                }
            )
        )
        messages = [
            {"time_label": "Jun 13, 10:00", "user_name": "Rahul", "text": "DB timeout issue"},
        ]

        first = summarize_channel(messages, "#hackathon-", "Last 2 days", mock_model)
        second = summarize_channel(messages, "#hackathon-", "Last 2 days", mock_model)

        assert first.health == "healthy"
        assert second.narrative == "Good progress."
        mock_model.generate_content.assert_called_once()


class TestNameResolution:
    """Tests for Slack user ID → display name resolution."""

    def test_prompt_serialization_prefers_real_name(self):
        payload = _messages_to_prompt_json(
            [
                {
                    "time_label": "Jun 13, 09:24",
                    "user_name": "User A",
                    "user_real_name": "Aditya Karanwal",
                    "text": "Started backend work.",
                }
            ]
        )

        assert "Aditya Karanwal" in payload
        assert "User A" not in payload

    def test_resolve_user_id(self):
        lookup = {"U0BASEAQUCQ": "Aditya Karanwal", "U0BASEAQUCQ".upper(): "Aditya Karanwal"}
        assert resolve_person_name("U0BASEAQUCQ", lookup) == "Aditya Karanwal"
        assert resolve_person_name("<@U0BASEAQUCQ>", lookup) == "Aditya Karanwal"
        assert resolve_person_name("@U0BASEAQUCQ", lookup) == "Aditya Karanwal"

    def test_apply_name_resolution_to_problems(self):
        lookup = build_user_lookup(
            [
                {
                    "user_id": "U0BASEAQUCQ",
                    "user_name": "Aditya Karanwal",
                    "text": "hello",
                }
            ]
        )
        summary = ChannelSummary(
            channel="#hackathon-",
            timeframe="Last 2 days",
            problems=[
                ProblemSolution(
                    problem="DB timeout",
                    reported_by="U0BASEAQUCQ",
                    solution="Async replication",
                    fixed_by="<@U0BASEAQUCQ>",
                )
            ],
        )
        resolved = apply_name_resolution(summary, lookup)
        assert resolved.problems[0].reported_by == "Aditya Karanwal"
        assert resolved.problems[0].fixed_by == "Aditya Karanwal"

    def test_format_block_kit_shows_names_not_ids(self):
        summary = ChannelSummary(
            channel="#hackathon-",
            timeframe="Last 2 days",
            problems=[
                ProblemSolution(
                    problem="DB timeout",
                    reported_by="Aditya Karanwal",
                    solution="Fix applied",
                    fixed_by="Aditya Karanwal",
                )
            ],
        )
        view = format_block_kit(summary)
        body = json.dumps(view)
        assert "Aditya Karanwal" in body
        assert "U0BASE" not in body

    def test_fallback_generates_summary_text(self):
        messages = [
            {
                "time_label": "Jun 13, 10:00",
                "user_name": "Rahul",
                "text": "Deployment is blocked by a timeout in staging.",
            },
            {
                "time_label": "Jun 13, 10:05",
                "user_name": "Priya",
                "text": "I am checking the timeout and restarting the service.",
            },
        ]

        summary = fallback_channel_summary("#test", "Today", messages)
        assert "Detailed extraction unavailable" not in summary.narrative
        assert "message(s) from 2 participant(s)" in summary.narrative
        assert "Latest update" in summary.narrative
