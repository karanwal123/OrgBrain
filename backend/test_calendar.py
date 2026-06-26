"""Tests for the calendar feature — parser, storage, and blocks."""

from __future__ import annotations

import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.calendar_blocks import (
    build_clear_confirmation_blocks,
    build_my_calendar_blocks,
    build_status_confirmation_blocks,
    build_usage_blocks,
    build_who_is_free_blocks,
    build_team_calendar_blocks,
)
from backend.calendar_parser import (
    ParsedCalendarCommand,
    parse_calendar_command,
    parse_date,
    parse_date_range,
    parse_time,
    validate_availability,
)
from backend.schemas import AvailabilityEntry, AvailabilityStatus


# ══════════════════════════════════════════════════════════════════════
#  parse_time
# ══════════════════════════════════════════════════════════════════════


class TestParseTime:
    def test_am_simple(self):
        assert parse_time("10am") == "10:00"

    def test_pm_simple(self):
        assert parse_time("3pm") == "15:00"

    def test_pm_with_minutes(self):
        assert parse_time("3:30pm") == "15:30"

    def test_12pm(self):
        assert parse_time("12pm") == "12:00"

    def test_12am(self):
        assert parse_time("12am") == "00:00"

    def test_24hr(self):
        assert parse_time("14:00") == "14:00"

    def test_midnight(self):
        assert parse_time("0:00") == "00:00"

    def test_invalid(self):
        assert parse_time("abc") is None

    def test_empty(self):
        assert parse_time("") is None


# ══════════════════════════════════════════════════════════════════════
#  parse_date
# ══════════════════════════════════════════════════════════════════════


class TestParseDate:
    def test_month_day(self):
        result = parse_date("May 5")
        assert result is not None
        assert result.endswith("-05-05")

    def test_month_day_ordinal(self):
        result = parse_date("June 15th")
        assert result is not None
        assert "-06-15" in result

    def test_slash_format(self):
        result = parse_date("6/15")
        assert result is not None
        assert "-06-15" in result

    def test_iso_passthrough(self):
        assert parse_date("2026-12-25") == "2026-12-25"

    def test_invalid(self):
        assert parse_date("not a date") is None

    def test_empty(self):
        assert parse_date("") is None

    def test_slash_with_year(self):
        assert parse_date("6/15/2026") == "2026-06-15"

    def test_abbreviated_month(self):
        result = parse_date("Jan 1")
        assert result is not None
        assert "-01-01" in result


# ══════════════════════════════════════════════════════════════════════
#  parse_date_range
# ══════════════════════════════════════════════════════════════════════


class TestParseDateRange:
    def test_single_date(self):
        start, end = parse_date_range("May 5")
        assert start is not None
        assert start == end

    def test_same_month_range(self):
        start, end = parse_date_range("May 5-10")
        assert start is not None
        assert end is not None
        assert start.endswith("-05-05")
        assert end.endswith("-05-10")

    def test_cross_month_range(self):
        start, end = parse_date_range("May 5 - June 2")
        assert start is not None
        assert end is not None
        assert "-05-05" in start
        assert "-06-02" in end

    def test_iso_single(self):
        start, end = parse_date_range("2026-12-25")
        assert start == "2026-12-25"
        assert end == "2026-12-25"


# ══════════════════════════════════════════════════════════════════════
#  parse_calendar_command
# ══════════════════════════════════════════════════════════════════════


class TestParseCalendarCommand:
    def test_empty_gives_help(self):
        result = parse_calendar_command("")
        assert result.action == "help"

    def test_status_full(self):
        result = parse_calendar_command(
            "status busy on May 5 from 10am to 12pm for Client meeting"
        )
        assert result.action == "status"
        assert result.status == "busy"
        assert result.date_start is not None
        assert result.time_start == "10:00"
        assert result.time_end == "12:00"
        assert result.reason == "Client meeting"
        assert not result.errors

    def test_status_no_time(self):
        result = parse_calendar_command("status leave on May 5 for Vacation")
        assert result.action == "status"
        assert result.status == "leave"
        assert result.reason == "Vacation"

    def test_status_free(self):
        result = parse_calendar_command("status free on May 5")
        assert result.action == "status"
        assert result.status == "free"
        assert not result.errors

    def test_status_invalid_format(self):
        result = parse_calendar_command("status something weird")
        assert result.action == "status"
        assert result.errors

    def test_who_is_free(self):
        result = parse_calendar_command("who-is-free May 5")
        assert result.action == "who-is-free"
        assert result.date_start is not None

    def test_who_is_free_bad_date(self):
        result = parse_calendar_command("who-is-free blah")
        assert result.action == "who-is-free"
        assert result.errors

    def test_my_calendar(self):
        result = parse_calendar_command("my-calendar")
        assert result.action == "my-calendar"

    def test_team_calendar(self):
        result = parse_calendar_command("team-calendar")
        assert result.action == "team-calendar"
        result2 = parse_calendar_command("team")
        assert result2.action == "team-calendar"

    def test_clear(self):
        result = parse_calendar_command("clear May 5")
        assert result.action == "clear"
        assert result.date_start is not None

    def test_clear_bad_date(self):
        result = parse_calendar_command("clear nonsense")
        assert result.action == "clear"
        assert result.errors

    def test_set_alias(self):
        result = parse_calendar_command("set busy on May 5")
        assert result.action == "status"
        assert result.status == "busy"

    def test_free_alias(self):
        result = parse_calendar_command("free May 5")
        assert result.action == "who-is-free"

    def test_date_range_in_status(self):
        result = parse_calendar_command("status leave on May 5-10 for Vacation")
        assert result.action == "status"
        assert result.status == "leave"
        assert result.date_start is not None
        assert result.date_end is not None
        assert result.date_start != result.date_end


# ══════════════════════════════════════════════════════════════════════
#  validate_availability
# ══════════════════════════════════════════════════════════════════════


class TestValidateAvailability:
    def test_valid_command(self):
        parsed = parse_calendar_command(
            "status busy on December 25 from 10am to 12pm for Holiday prep"
        )
        errors = validate_availability(parsed)
        assert not errors

    def test_invalid_status(self):
        parsed = ParsedCalendarCommand(
            action="status",
            status="sleeping",
            date_start="2099-01-01",
            date_end="2099-01-01",
        )
        errors = validate_availability(parsed)
        assert any("status" in e.lower() for e in errors)

    def test_past_date(self):
        parsed = ParsedCalendarCommand(
            action="status",
            status="busy",
            date_start="2020-01-01",
            date_end="2020-01-01",
        )
        errors = validate_availability(parsed)
        assert any("past" in e.lower() for e in errors)

    def test_start_after_end_date(self):
        parsed = ParsedCalendarCommand(
            action="status",
            status="busy",
            date_start="2099-01-10",
            date_end="2099-01-05",
        )
        errors = validate_availability(parsed)
        assert any("start date" in e.lower() for e in errors)

    def test_time_order(self):
        parsed = ParsedCalendarCommand(
            action="status",
            status="busy",
            date_start="2099-01-01",
            date_end="2099-01-01",
            time_start="14:00",
            time_end="10:00",
        )
        errors = validate_availability(parsed)
        assert any("start time" in e.lower() for e in errors)

    def test_reason_too_long(self):
        parsed = ParsedCalendarCommand(
            action="status",
            status="busy",
            date_start="2099-01-01",
            date_end="2099-01-01",
            reason="x" * 101,
        )
        errors = validate_availability(parsed)
        assert any("reason" in e.lower() for e in errors)


# ══════════════════════════════════════════════════════════════════════
#  Block Kit builders
# ══════════════════════════════════════════════════════════════════════


def _make_entry(**kwargs) -> AvailabilityEntry:
    """Helper to create a test AvailabilityEntry."""
    defaults = {
        "user_id": "U123",
        "user_name": "testuser",
        "user_display_name": "Test User",
        "user_email": "test@example.com",
        "team_id": "T123",
        "date_start": "2026-06-15",
        "date_end": "2026-06-15",
        "time_start": "10:00",
        "time_end": "12:00",
        "status": AvailabilityStatus.busy,
        "reason": "Team meeting",
        "channel_id": "C123",
        "timezone": "Asia/Kolkata",
    }
    defaults.update(kwargs)
    return AvailabilityEntry(**defaults)


class TestBuildStatusConfirmation:
    def test_basic(self):
        entry = _make_entry()
        blocks = build_status_confirmation_blocks(entry)
        assert len(blocks) >= 2
        assert blocks[0]["type"] == "header"
        assert "Updated" in blocks[0]["text"]["text"]

    def test_with_reason(self):
        entry = _make_entry(reason="Client call")
        blocks = build_status_confirmation_blocks(entry)
        # Should have a reason section
        texts = [b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section"]
        assert any("Client call" in t for t in texts)

    def test_no_reason(self):
        entry = _make_entry(reason=None)
        blocks = build_status_confirmation_blocks(entry)
        # Should still produce valid blocks
        assert len(blocks) >= 2


class TestBuildWhoIsFree:
    def test_empty(self):
        blocks = build_who_is_free_blocks("2026-06-15", [], [], [])
        assert len(blocks) >= 3
        assert blocks[0]["type"] == "header"

    def test_with_entries(self):
        free = [_make_entry(status=AvailabilityStatus.free, user_id=f"U{i}") for i in range(3)]
        busy = [_make_entry(status=AvailabilityStatus.busy)]
        leave = [_make_entry(status=AvailabilityStatus.leave)]
        blocks = build_who_is_free_blocks("2026-06-15", free, busy, leave)
        assert len(blocks) > 5

    def test_pagination_overflow(self):
        """Test that 100+ people don't exceed block limits."""
        free = [
            _make_entry(
                status=AvailabilityStatus.free,
                user_id=f"U{i:04d}",
                user_display_name=f"User {i}",
            )
            for i in range(100)
        ]
        blocks = build_who_is_free_blocks("2026-06-15", free, [], [])
        # Should have overflow message (showing only 10 + "... and 90 more")
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if "text" in b
        )
        assert "90 more" in all_text


class TestBuildMyCalendar:
    def test_empty(self):
        blocks = build_my_calendar_blocks([], "testuser")
        assert len(blocks) >= 1
        assert blocks[0]["type"] == "header"

    def test_with_entries(self):
        entries = [_make_entry() for _ in range(3)]
        blocks = build_my_calendar_blocks(entries, "testuser")
        assert len(blocks) >= 4  # header + divider + 3 entries


class TestBuildClearConfirmation:
    def test_cleared(self):
        blocks = build_clear_confirmation_blocks("2026-06-15", 2)
        text = blocks[0]["text"]["text"]
        assert "2" in text
        assert "Cleared" in text

    def test_nothing_to_clear(self):
        blocks = build_clear_confirmation_blocks("2026-06-15", 0)
        text = blocks[0]["text"]["text"]
        assert "No entries" in text


class TestBuildUsage:
    def test_usage_blocks(self):
        blocks = build_usage_blocks()
        assert len(blocks) >= 2
        assert blocks[0]["type"] == "header"
        all_text = " ".join(
            b.get("text", {}).get("text", "") for b in blocks if "text" in b
        )
        assert "/calendar" in all_text


class TestBuildCalendarModal:
    def test_build_calendar_modal(self):
        from backend.calendar_modal import build_calendar_modal
        modal = build_calendar_modal("C123")
        assert modal["type"] == "modal"
        assert modal["callback_id"] == "calendar_status_modal"
        assert modal["private_metadata"] == "C123"
        block_types = [b["type"] for b in modal["blocks"]]
        assert "input" in block_types

    def test_build_team_calendar_modal_view(self):
        from backend.calendar_modal import build_team_calendar_modal_view
        modal = build_team_calendar_modal_view([])
        assert modal["type"] == "modal"
        assert modal["callback_id"] == "team_calendar_modal"
        assert len(modal["blocks"]) >= 1


class TestBuildTeamCalendarBlocks:
    def test_empty(self):
        blocks = build_team_calendar_blocks([])
        assert len(blocks) >= 2
        assert "No upcoming" in blocks[1]["text"]["text"]

    def test_with_entries(self):
        entry1 = _make_entry(user_id="U1", user_display_name="User One", date_start="2026-06-15")
        entry2 = _make_entry(user_id="U2", user_display_name="User Two", date_start="2026-06-16")
        blocks = build_team_calendar_blocks([entry1, entry2])
        assert len(blocks) >= 5
        body = str(blocks)
        assert "User One" in body
        assert "User Two" in body


