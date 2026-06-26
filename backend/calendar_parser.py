"""Parse /calendar slash command text into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

VALID_STATUSES = {"free", "busy", "leave"}


@dataclass
class ParsedCalendarCommand:
    """Structured result of parsing a /calendar command."""

    action: str  # "status", "who-is-free", "my-calendar", "clear"
    status: Optional[str] = None
    date_start: Optional[str] = None  # YYYY-MM-DD
    date_end: Optional[str] = None  # YYYY-MM-DD
    time_start: Optional[str] = None  # HH:MM
    time_end: Optional[str] = None  # HH:MM
    reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


def parse_time(time_str: str) -> Optional[str]:
    """Convert time strings to HH:MM 24-hour format.

    Examples:
        "10am"   -> "10:00"
        "3:30pm" -> "15:30"
        "14:00"  -> "14:00"
    """
    time_str = time_str.strip().lower()
    match = re.match(r"^(\d{1,2}):?(\d{2})?\s*(am|pm)?$", time_str)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    if period == "pm" and hour != 12:
        hour += 12
    if period == "am" and hour == 12:
        hour = 0

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None

    return f"{hour:02d}:{minute:02d}"


def parse_date(date_str: str) -> Optional[str]:
    """Convert human date strings to YYYY-MM-DD.

    Supported formats:
        "May 5"     -> "2026-05-05"  (current/next occurrence)
        "June 15"   -> "2026-06-15"
        "6/15"      -> "2026-06-15"
        "2026-06-15" -> "2026-06-15" (passthrough)
    """
    date_str = date_str.strip()

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # M/D or M/D/YY format
    slash_match = re.match(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$", date_str)
    if slash_match:
        month = int(slash_match.group(1))
        day = int(slash_match.group(2))
        year = int(slash_match.group(3)) if slash_match.group(3) else datetime.now().year
        if year < 100:
            year += 2000
        try:
            date(year, month, day)  # validate
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            return None

    # "May 5" or "May 5th" format
    parts = date_str.lower().split()
    if len(parts) >= 2:
        month_str = parts[0]
        day_str = re.sub(r"(st|nd|rd|th)$", "", parts[1])
        month = MONTHS.get(month_str)
        if month is not None:
            try:
                day = int(day_str)
                now = datetime.now()
                year = now.year
                # If the date has passed this year, use next year
                target = date(year, month, day)
                if target < now.date():
                    year += 1
                return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, TypeError):
                return None

    return None


def parse_date_range(date_str: str) -> tuple[Optional[str], Optional[str]]:
    """Parse single date or date range.

    Examples:
        "May 5"         -> ("2026-05-05", "2026-05-05")
        "May 5-10"      -> ("2026-05-05", "2026-05-10")
        "May 5 - June 2" -> ("2026-05-05", "2026-06-02")
    """
    date_str = date_str.strip()

    # "May 5-10" — same month range
    same_month = re.match(
        r"^([a-zA-Z]+)\s+(\d{1,2})\s*-\s*(\d{1,2})$", date_str
    )
    if same_month:
        month_str = same_month.group(1)
        day_start = same_month.group(2)
        day_end = same_month.group(3)
        start = parse_date(f"{month_str} {day_start}")
        end = parse_date(f"{month_str} {day_end}")
        return (start, end)

    # "May 5 - June 2" — cross-month range
    cross_month = re.match(
        r"^([a-zA-Z]+\s+\d{1,2})\s*-\s*([a-zA-Z]+\s+\d{1,2})$", date_str
    )
    if cross_month:
        start = parse_date(cross_month.group(1))
        end = parse_date(cross_month.group(2))
        return (start, end)

    # Single date
    single = parse_date(date_str)
    return (single, single)


def parse_calendar_command(text: str) -> ParsedCalendarCommand:
    """Parse raw /calendar command text into structured data.

    Supported commands:
        /calendar status busy on May 5 from 10am to 12pm for Client meeting
        /calendar who-is-free May 5
        /calendar my-calendar
        /calendar clear May 5
    """
    text = text.strip()
    if not text:
        return ParsedCalendarCommand(action="help")

    parts = text.split(None, 1)
    action = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # --- my-calendar (no args) ---
    if action in ("my-calendar", "mycalendar", "my"):
        return ParsedCalendarCommand(action="my-calendar")

    # --- team-calendar (no args) ---
    if action in ("team-calendar", "teamcalendar", "team"):
        return ParsedCalendarCommand(action="team-calendar")

    # --- who-is-free <date> ---
    if action in ("who-is-free", "whofree", "free", "query"):
        date_start, date_end = parse_date_range(rest)
        if not date_start:
            return ParsedCalendarCommand(
                action="who-is-free",
                errors=[f"Could not parse date: '{rest}'"],
            )
        return ParsedCalendarCommand(
            action="who-is-free",
            date_start=date_start,
            date_end=date_end,
        )

    # --- clear <date> ---
    if action == "clear":
        date_start, date_end = parse_date_range(rest)
        if not date_start:
            return ParsedCalendarCommand(
                action="clear",
                errors=[f"Could not parse date: '{rest}'"],
            )
        return ParsedCalendarCommand(
            action="clear",
            date_start=date_start,
            date_end=date_end,
        )

    # --- status <busy|free|leave> on <date> [from <time> to <time>] [for <reason>] ---
    if action in ("status", "set"):
        pattern = re.compile(
            r"^(\w+)\s+on\s+(.+?)(?:\s+from\s+(\S+)\s+to\s+(\S+))?(?:\s+for\s+(.+))?$",
            re.IGNORECASE,
        )
        match = pattern.match(rest)
        if not match:
            return ParsedCalendarCommand(
                action="status",
                errors=["Invalid format. Use: `/calendar status busy on May 5 from 10am to 12pm for Reason`"],
            )

        status_str = match.group(1).lower()
        date_raw = match.group(2).strip()
        time_start_raw = match.group(3)
        time_end_raw = match.group(4)
        reason = match.group(5)

        date_start, date_end = parse_date_range(date_raw)
        time_start = parse_time(time_start_raw) if time_start_raw else "00:00"
        time_end = parse_time(time_end_raw) if time_end_raw else "23:59"

        errors: list[str] = []
        if status_str not in VALID_STATUSES:
            errors.append(f"Invalid status '{status_str}'. Use: free, busy, or leave")
        if not date_start:
            errors.append(f"Could not parse date: '{date_raw}'")
        if time_start_raw and time_start is None:
            errors.append(f"Could not parse start time: '{time_start_raw}'")
        if time_end_raw and time_end is None:
            errors.append(f"Could not parse end time: '{time_end_raw}'")

        return ParsedCalendarCommand(
            action="status",
            status=status_str,
            date_start=date_start,
            date_end=date_end,
            time_start=time_start,
            time_end=time_end,
            reason=reason,
            errors=errors,
        )

    return ParsedCalendarCommand(action="help")


def validate_availability(parsed: ParsedCalendarCommand) -> list[str]:
    """Validate a parsed calendar command, returning a list of error strings."""
    errors = list(parsed.errors)

    if parsed.status and parsed.status not in VALID_STATUSES:
        errors.append(f"Invalid status: {parsed.status}. Must be free, busy, or leave")

    if parsed.date_start:
        today = datetime.now().date().isoformat()
        if parsed.date_start < today:
            errors.append("Cannot set availability for past dates")

    if parsed.date_start and parsed.date_end:
        if parsed.date_start > parsed.date_end:
            errors.append("Start date must be before or equal to end date")

    if parsed.time_start and parsed.time_end:
        # Only validate time ordering on single-day entries
        if parsed.date_start == parsed.date_end and parsed.time_start >= parsed.time_end:
            errors.append("Start time must be before end time")

    if parsed.reason and len(parsed.reason) > 100:
        errors.append("Reason too long (max 100 characters)")

    return errors
