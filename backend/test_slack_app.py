"""Tests for Slack app helpers and profile search."""

from unittest.mock import MagicMock

import pytest

from backend.schemas import EmployeeProfile
from backend.channel_history import fetch_channel_messages
from backend.slack_app import (
    build_about_blocks,
    build_help_blocks,
    build_profile_mini_description,
    build_who_knows_blocks,
    extract_about_query,
    is_processable_message,
    resolve_help_profiles,
    summarize_slash_command,
    recall_slash_command,
)
from backend.extraction import fallback_profile_about


class TestIsProcessableMessage:
    """Tests for Slack message filtering."""

    def test_accepts_normal_message(self):
        assert is_processable_message({"text": "Hello team", "user": "U123"}) is True

    def test_rejects_bot_message(self):
        assert is_processable_message({"text": "hi", "bot_id": "B123"}) is False

    def test_rejects_message_changed_subtype(self):
        assert is_processable_message({"subtype": "message_changed", "text": "hi"}) is False

    def test_rejects_empty_text(self):
        assert is_processable_message({"text": "   "}) is False


class TestBuildWhoKnowsBlocks:
    """Tests for Block Kit response builder."""

    def test_empty_results(self):
        blocks = build_who_knows_blocks("rust", [])
        assert blocks[0]["type"] == "header"
        assert "rust" in blocks[0]["text"]["text"]
        assert any("No profiles found" in b.get("text", {}).get("text", "") for b in blocks)

    def test_with_profiles(self):
        profile = EmployeeProfile(
            person="Priya Sharma",
            role="Senior Backend Engineer",
            skills=["Kubernetes", "Python"],
            projects=["payment microservices"],
            confidence=0.9,
        )
        blocks = build_who_knows_blocks("kubernetes", [profile])
        body = str(blocks)
        assert "Priya Sharma" in body
        assert "Kubernetes" in body

    def test_with_match_reasons(self):
        profile = EmployeeProfile(
            person="Aditya Karanwal",
            role="Senior Backend Engineer",
            skills=["Kubernetes"],
            confidence=0.9,
        )
        reasons = {
            "Aditya Karanwal": (
                "Senior Backend Engineer with strong Kubernetes experience from platform work."
            )
        }
        blocks = build_who_knows_blocks("kubernetes", [profile], reasons)
        body = str(blocks)
        assert "strong Kubernetes experience" in body
        assert "Aditya Karanwal" in body


class TestBuildHelpBlocks:
    """Tests for /help Block Kit builder."""

    def test_help_blocks_with_profiles(self):
        profile = EmployeeProfile(
            person="Aditya Karanwal",
            role="Senior Backend Engineer",
            skills=["Kubernetes"],
            confidence=0.9,
        )
        blocks = build_help_blocks(["Kubernetes"], "Kubernetes deployment issue", [profile])
        body = str(blocks)
        assert "Aditya Karanwal" in body
        assert "Kubernetes deployment issue" in body

    def test_help_blocks_with_match_reasons(self):
        profile = EmployeeProfile(
            person="Aditya Karanwal",
            role="Senior Backend Engineer",
            skills=["Kubernetes"],
            confidence=0.9,
        )
        reasons = {"Aditya Karanwal": "Best fit for Kubernetes deployment and cluster issues."}
        blocks = build_help_blocks(
            ["Kubernetes"], "Kubernetes deployment issue", [profile], reasons
        )
        body = str(blocks)
        assert "Best fit for Kubernetes" in body


class TestProfileMiniDescription:
    """Tests for profile mini description fallback."""

    def test_profile_mini_description(self):
        profile = EmployeeProfile(
            person="Aditya",
            role="Senior Backend Engineer",
            skills=["Kubernetes", "Python"],
        )
        desc = build_profile_mini_description(profile)
        assert "Senior Backend Engineer" in desc
        assert "Kubernetes" in desc


class TestGenerateMatchReasons:
    """Tests for Gemini match reason generation."""

    def test_generate_match_reasons_parses_response(self):
        import json
        from unittest.mock import MagicMock

        from backend.extraction import generate_match_reasons

        profile = EmployeeProfile(
            person="Aditya Karanwal",
            role="Senior Backend Engineer",
            skills=["Kubernetes"],
        )
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                {
                    "reasons": [
                        {
                            "person": "Aditya Karanwal",
                            "reason": "Senior Backend Engineer with Kubernetes platform expertise.",
                        }
                    ]
                }
            )
        )

        reasons = generate_match_reasons("kubernetes", [profile], mock_model)
        assert reasons["Aditya Karanwal"] == (
            "Senior Backend Engineer with Kubernetes platform expertise."
        )

    def test_parse_bare_array_response(self):
        import json

        from backend.extraction import _parse_match_reasons_response

        raw = json.dumps(
            [
                {
                    "person": "Aditya Karanwal",
                    "reason": "Senior Backend Engineer with Kubernetes skills.",
                }
            ]
        )
        result = _parse_match_reasons_response(raw)
        assert result.reasons[0].person == "Aditya Karanwal"
        assert "Kubernetes" in result.reasons[0].reason


class TestResolveHelpProfiles:
    """Tests for help query resolution."""

    def test_resolve_uses_extracted_topics(self):
        from unittest.mock import MagicMock, patch

        from backend.schemas import HelpQueryResult

        mock_model = MagicMock()
        mock_storage = MagicMock()
        mock_storage.search_by_topics.return_value = [
            EmployeeProfile(person="Aditya", skills=["Kubernetes"], confidence=0.9)
        ]

        with patch("backend.slack_app.extract_help_topics") as mock_extract:
            mock_extract.return_value = HelpQueryResult(
                topics=["Kubernetes"],
                summary="Issue with Kubernetes cluster",
                confidence=0.9,
            )
            topics, summary, profiles = resolve_help_profiles(
                "I'm facing an issue in Kubernetes who can I ask",
                mock_model,
                mock_storage,
            )

        assert topics == ["Kubernetes"]
        assert "Kubernetes" in summary
        assert profiles[0].person == "Aditya"
        mock_storage.search_by_topics.assert_called_once_with(["Kubernetes"])


class TestExtractAboutQuery:
    """Tests for /about query parsing."""

    def test_tell_me_about(self):
        assert extract_about_query("tell me about Aditya Karanwal") == "Aditya Karanwal"

    def test_who_is(self):
        assert extract_about_query("who is YASH") == "YASH"

    def test_strips_bot_mention(self):
        assert extract_about_query("<@U123> tell me about Aditya") == "Aditya"


class TestBuildAboutBlocks:
    """Tests for /about Block Kit builder."""

    def test_about_blocks(self):
        profile = EmployeeProfile(
            person="Aditya Karanwal",
            role="Senior Backend Engineer",
            skills=["Kubernetes"],
        )
        about = fallback_profile_about(profile)
        blocks = build_about_blocks(profile, about)
        body = str(blocks)
        assert "Aditya Karanwal" in body
        assert "go-to" in body.lower()


class TestSummarizeSlashCommand:
    """Tests for /summarize handler behavior."""

    def test_modal_open_failure_uses_slash_response(self):
        client = MagicMock()
        client.views_open.side_effect = Exception("expired_trigger_id")
        respond = MagicMock()

        summarize_slash_command(
            {
                "text": "last 24 hours",
                "channel_id": "C123",
                "trigger_id": "T123",
                "user_id": "U123",
            },
            client,
            respond,
            None,
        )

        client.views_update.assert_not_called()
        client.chat_postEphemeral.assert_not_called()
        respond.assert_called_once()
        payload = respond.call_args.kwargs
        assert payload["response_type"] == "ephemeral"
        assert "Could not open summary modal" in payload["text"]

    def test_history_not_in_channel_updates_modal_with_invite_help(self):
        from unittest.mock import patch

        from backend.channel_history import ChannelHistoryError

        client = MagicMock()
        client.views_open.return_value = {"view": {"id": "V123"}}
        respond = MagicMock()

        with patch("backend.slack_app.resolve_channel_id") as mock_resolve:
            with patch("backend.slack_app.fetch_channel_messages") as mock_fetch:
                mock_resolve.return_value = ("C999", "#hackathon-")
                mock_fetch.side_effect = ChannelHistoryError(
                    "Bot is not in this channel. Run `/invite @OrgBrain` first."
                )

                summarize_slash_command(
                    {
                        "text": "#hackathon- last 2 days",
                        "channel_id": "C123",
                        "trigger_id": "T123",
                        "user_id": "U123",
                    },
                    client,
                    respond,
                    MagicMock(),
                )

        client.views_update.assert_called_once()
        updated_view = client.views_update.call_args.kwargs["view"]
        body = str(updated_view)
        assert "invite @OrgBrain" in body
        respond.assert_not_called()


class TestFetchChannelMessages:
    """Tests for Slack history name resolution fallbacks."""

    def setup_method(self):
        from backend.channel_history import _MESSAGES_CACHE
        with _MESSAGES_CACHE._lock:
            _MESSAGES_CACHE._items.clear()

    def test_users_list_preeseds_real_names_for_messages(self):
        client = MagicMock()
        client.users_list.return_value = {
            "members": [
                {
                    "id": "U12345680",
                    "profile": {"real_name": "Priya Sharma", "display_name": "User B"},
                    "name": "priya",
                }
            ],
            "response_metadata": {},
        }
        client.conversations_history.return_value = {
            "messages": [
                {
                    "text": "Working on backend",
                    "ts": "1.0",
                    "user": "U12345680",
                }
            ],
            "response_metadata": {},
        }

        messages, user_cache = fetch_channel_messages(client, "C123", 0)

        assert messages[0]["user_name"] == "Priya Sharma"
        assert user_cache["U12345680"] == "Priya Sharma"

    def test_uses_message_profile_display_name_when_lookup_is_missing(self):
        client = MagicMock()
        client.conversations_history.return_value = {
            "messages": [
                {
                    "text": "Hello",
                    "ts": "1.0",
                    "user": "U12345678",
                    "user_profile": {"display_name": "Ada Lovelace"},
                }
            ],
            "response_metadata": {},
        }

        messages, user_cache = fetch_channel_messages(client, "C123", 0)

        assert messages[0]["user_name"] == "Ada Lovelace"
        assert user_cache["U12345678"] == "Ada Lovelace"

    def test_prefers_real_name_over_placeholder_display_name(self):
        client = MagicMock()
        client.conversations_history.return_value = {
            "messages": [
                {
                    "text": "Hello",
                    "ts": "1.0",
                    "user": "U12345679",
                    "user_profile": {
                        "display_name": "User A",
                        "real_name": "Ada Lovelace",
                    },
                }
            ],
            "response_metadata": {},
        }

        messages, user_cache = fetch_channel_messages(client, "C123", 0)

        assert messages[0]["user_name"] == "Ada Lovelace"
        assert user_cache["U12345679"] == "Ada Lovelace"


class TestFindProfileByName:
    """Tests for name-based profile lookup."""

    def test_find_partial_name(self):
        from unittest.mock import MagicMock, patch

        from backend.storage import ProfileStorage

        mock_collection = MagicMock()
        mock_collection.find_one.side_effect = [
            None,  # exact get_profile path handled separately - first call is regex exact
            {
                "person": "Aditya Karanwal",
                "role": "Senior Backend Engineer",
                "skills": ["Kubernetes"],
                "projects": [],
                "confidence": 0.9,
                "source_messages": [],
            },
        ]

        with patch.object(ProfileStorage, "_ensure_indexes"):
            with patch.object(ProfileStorage, "get_profile", return_value=None):
                storage = ProfileStorage.__new__(ProfileStorage)
                storage._collection = mock_collection
                result = storage.find_profile_by_name("aditya")
                assert result is not None
                assert result.person == "Aditya Karanwal"


class TestSearchBySkill:
    """Tests for MongoDB skill search (requires MongoDB or mock)."""

    def test_search_by_skill_returns_matches(self):
        from unittest.mock import MagicMock, patch

        from backend.storage import ProfileStorage

        mock_collection = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value = [
            {
                "person": "Priya Sharma",
                "role": "Senior Backend Engineer",
                "skills": ["Kubernetes"],
                "projects": [],
                "confidence": 0.9,
                "source_messages": [],
            }
        ]

        with patch.object(ProfileStorage, "_ensure_indexes"):
            storage = ProfileStorage.__new__(ProfileStorage)
            storage._collection = mock_collection

            results = storage.search_by_skill("kubernetes")
            assert len(results) == 1
            assert results[0].person == "Priya Sharma"


class TestRecallSlashCommand:
    """Tests for /recall handler behavior."""

    def test_missing_query_returns_usage(self):
        respond = MagicMock()
        client = MagicMock()
        recall_slash_command(
            {
                "text": "   ",
                "channel_id": "C123",
                "user_name": "Aditya",
            },
            client,
            respond,
            MagicMock(),
        )
        respond.assert_called_once()
        payload = respond.call_args.kwargs
        assert payload["response_type"] == "ephemeral"
        assert "Usage: `/recall <search query>`" in payload["text"]

    def test_missing_memory_service_returns_error(self):
        respond = MagicMock()
        client = MagicMock()
        recall_slash_command(
            {
                "text": "redis",
                "channel_id": "C123",
                "user_name": "Aditya",
            },
            client,
            respond,
            None,
        )
        respond.assert_called_once()
        payload = respond.call_args.kwargs
        assert "unavailable" in payload["text"]

    def test_executes_search_successfully(self):
        from backend.schemas import MemoryRetrievalHit, MemoryUnitType
        
        respond = MagicMock()
        client = MagicMock()
        mock_memory_service = MagicMock()
        
        # Mock search hits
        mock_memory_service.search.return_value = [
            MemoryRetrievalHit(
                memory_id="123",
                channel_id="C456",
                summary="Redis cluster migration completed.",
                unit_type=MemoryUnitType.decision,
                score=0.95,
                semantic_score=0.95,
                recency_score=0.9,
                importance_score=0.9,
                unresolved_score=0.0,
                owners=["Aditya"],
                tags=["redis", "migration"],
            )
        ]
        
        recall_slash_command(
            {
                "text": "redis",
                "channel_id": "C123",
                "user_name": "Aditya",
            },
            client,
            respond,
            mock_memory_service,
        )
        
        mock_memory_service.flush_channel.assert_called_once_with("C123")
        mock_memory_service.search.assert_called_once_with("redis", limit=5)
        respond.assert_called_once()
        payload = respond.call_args.kwargs
        assert payload["response_type"] == "ephemeral"
        assert "blocks" in payload
        blocks = payload["blocks"]
        assert len(blocks) > 0
        body = str(blocks)
        assert "Redis cluster migration completed" in body
        assert "<#C456>" in body
        assert "Aditya" in body

