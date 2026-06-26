"""Tests for channel memory delta and retrieval helpers."""

from datetime import datetime, timedelta, timezone

from backend.memory import (
    MEMORY_SUMMARIZATION_PROMPT,
    build_memory_prompt,
    extract_memory_units,
    memory_unit_fingerprint,
    merge_memory_units,
    rank_memory_hit,
    recency_score,
    select_delta_messages,
    update_channel_memory_state,
)
from backend.schemas import MemoryDeltaBatch, MemoryUnit, MemoryUnitType


def _ts(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


class TestSelectDeltaMessages:
    """Tests for checkpoint-aware message selection."""

    def test_filters_messages_before_checkpoint(self):
        messages = [
            {"id": "1", "ts": 1000, "text": "old"},
            {"id": "2", "ts": 1010, "text": "newer"},
            {"id": "3", "ts": 1020, "text": "newest"},
        ]

        selected = select_delta_messages(
            messages,
            last_summary_ts=_ts(1005),
            last_processed_message_id="1",
        )

        assert [message["id"] for message in selected] == ["2", "3"]

    def test_filters_by_last_processed_message_id(self):
        messages = [
            {"id": "1", "ts": 1000, "text": "old"},
            {"id": "2", "ts": 1010, "text": "skip"},
            {"id": "3", "ts": 1020, "text": "keep"},
        ]

        selected = select_delta_messages(messages, last_processed_message_id="2")
        assert [message["id"] for message in selected] == ["3"]


class TestMemoryExtraction:
    """Tests for message-to-memory compression."""

    def test_extracts_problem_and_action_units(self):
        batch = MemoryDeltaBatch(
            channel_id="C123",
            messages=[
                {
                    "id": "10",
                    "ts": 2000,
                    "user_name": "John",
                    "text": "Redis timeout issue is blocking checkout under load.",
                },
                {
                    "id": "11",
                    "ts": 2010,
                    "user_name": "Priya",
                    "text": "We need to follow up with infra on the cache config.",
                },
            ],
        )

        units = extract_memory_units(batch)

        assert units[0].channel_id == "C123"
        assert units[0].unit_type in {MemoryUnitType.problem, MemoryUnitType.unresolved_issue}
        assert units[0].unresolved is True
        assert units[1].unit_type == MemoryUnitType.action_item
        assert "follow" in " ".join(units[1].tags)

    def test_memory_fingerprint_is_stable(self):
        first = memory_unit_fingerprint("C123", MemoryUnitType.problem, "Redis timeout issue")
        second = memory_unit_fingerprint("C123", MemoryUnitType.problem, "Redis timeout issue")
        assert first == second


class TestMemoryMerging:
    """Tests for channel memory merge behavior."""

    def test_merge_memory_units_deduplicates(self):
        previous = [
            MemoryUnit(
                memory_id="m1",
                channel_id="C123",
                unit_type=MemoryUnitType.problem,
                summary="Redis timeout issue",
                source_message_ids=["1"],
                source_timestamps=[_ts(1000)],
                owners=["John"],
                tags=["redis"],
                unresolved=True,
            )
        ]
        delta = [
            MemoryUnit(
                memory_id="m1",
                channel_id="C123",
                unit_type=MemoryUnitType.problem,
                summary="Redis timeout issue",
                source_message_ids=["2"],
                source_timestamps=[_ts(1010)],
                owners=["Priya"],
                tags=["timeout"],
                unresolved=True,
                importance=0.95,
            )
        ]

        merged = merge_memory_units(previous, delta)

        assert len(merged) == 1
        assert merged[0].source_message_ids == ["1", "2"]
        assert merged[0].importance == 0.95
        assert merged[0].owners == ["John", "Priya"]

    def test_update_channel_memory_state_advances_checkpoint(self):
        batch = MemoryDeltaBatch(
            channel_id="C123",
            messages=[
                {"id": "1", "ts": 2000, "user_name": "John", "text": "We decided to keep Redis cache warm."},
                {"id": "2", "ts": 2010, "user_name": "Priya", "text": "Action item: update the cache config."},
            ],
        )

        state = update_channel_memory_state(None, batch)

        assert state.channel_id == "C123"
        assert state.last_processed_message_id == "2"
        assert state.last_summary_ts == _ts(2010)
        assert "Redis" in state.compressed_context or "redis" in state.compressed_context.lower()
        assert state.pending_messages == []


class TestPromptAndRanking:
    """Tests for prompt construction and retrieval scoring."""

    def test_build_memory_prompt_contains_rules(self):
        batch = MemoryDeltaBatch(channel_id="C123", messages=[{"id": "1", "ts": 1, "text": "hello"}])
        prompt = build_memory_prompt(batch, "previous snapshot")

        assert "channel only" in prompt.lower()
        assert "hello" in prompt
        assert "previous snapshot" in prompt
        assert MEMORY_SUMMARIZATION_PROMPT.splitlines()[0] in prompt

    def test_recency_scores_fresh_items_higher(self):
        now = datetime.now(timezone.utc)
        recent = MemoryUnit(
            memory_id="fresh",
            channel_id="C123",
            unit_type=MemoryUnitType.problem,
            summary="Fresh issue",
            updated_at=now,
            unresolved=True,
            importance=0.9,
        )
        old = MemoryUnit(
            memory_id="old",
            channel_id="C123",
            unit_type=MemoryUnitType.context,
            summary="Old note",
            updated_at=now - timedelta(days=30),
            unresolved=False,
            importance=0.2,
        )

        fresh_hit = rank_memory_hit(recent, semantic_score=0.8, query_channel_id="C123")
        old_hit = rank_memory_hit(old, semantic_score=0.8, query_channel_id="C123")

        assert fresh_hit.score > old_hit.score
        assert fresh_hit.unresolved_score == 1.0
        assert recency_score(now) >= recency_score(now - timedelta(days=30))
