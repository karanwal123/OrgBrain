"""Slack Block Kit builders for calendar availability cards."""

from __future__ import annotations

from backend.schemas import AvailabilityEntry

# Status emoji mapping
STATUS_EMOJI = {
    "free": "✅",
    "busy": "⛔",
    "leave": "🏖️",
}

MAX_PEOPLE_PER_SECTION = 10
MAX_BLOCKS = 140  # Slack limit is 150, leave room for header/footer


def _fmt_time_range(entry: AvailabilityEntry) -> str:
    """Format time range as human-readable string."""
    if entry.time_start == "00:00" and entry.time_end == "23:59":
        return "All day"
    return f"{entry.time_start} – {entry.time_end}"


def _fmt_date_range(entry: AvailabilityEntry) -> str:
    """Format date range as human-readable string."""
    if entry.date_start == entry.date_end:
        return entry.date_start
    return f"{entry.date_start} → {entry.date_end}"


def build_status_confirmation_blocks(entry: AvailabilityEntry) -> list[dict]:
    """Build Block Kit blocks for a status update confirmation card.

    Shows: ✅ Status Updated — date, time, status, reason.
    """
    emoji = STATUS_EMOJI.get(entry.status, "📅")
    date_display = _fmt_date_range(entry)
    time_display = _fmt_time_range(entry)
    reason_line = f"\n📝 *Reason:* {entry.reason}" if entry.reason else ""

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Availability Updated",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Status:*\n{emoji} {entry.status.upper()}"},
                {"type": "mrkdwn", "text": f"*Date:*\n📅 {date_display}"},
                {"type": "mrkdwn", "text": f"*Time:*\n🕐 {time_display}"},
                {"type": "mrkdwn", "text": f"*Timezone:*\n🌍 {entry.timezone}"},
            ],
        },
    ]

    if reason_line:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": reason_line},
        })

    blocks.append({"type": "divider"})
    return blocks


def build_who_is_free_blocks(
    date_str: str,
    free: list[AvailabilityEntry],
    busy: list[AvailabilityEntry],
    leave: list[AvailabilityEntry],
) -> list[dict]:
    """Build Block Kit blocks showing who's free/busy/on-leave for a date.

    Handles pagination: truncates to MAX_PEOPLE_PER_SECTION per group and
    shows overflow counts.
    """
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📅 Team Availability — {date_str}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"✅ *{len(free)} Free*  ·  "
                    f"⛔ *{len(busy)} Busy*  ·  "
                    f"🏖️ *{len(leave)} On Leave*"
                ),
            },
        },
        {"type": "divider"},
    ]

    def _add_group(
        group: list[AvailabilityEntry],
        emoji: str,
        label: str,
    ) -> None:
        if not group:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{emoji} *{label}*\n_No one_"},
            })
            return

        shown = group[:MAX_PEOPLE_PER_SECTION]
        overflow = len(group) - len(shown)

        lines = [f"{emoji} *{label}*"]
        for entry in shown:
            name = entry.user_display_name or entry.user_name or entry.user_id
            time_range = _fmt_time_range(entry)
            reason = f" — _{entry.reason}_" if entry.reason else ""
            lines.append(f"  • <@{entry.user_id}> ({name}) · {time_range}{reason}")

        if overflow > 0:
            lines.append(f"  _... and {overflow} more_")

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })

    _add_group(free, "✅", "FREE")
    blocks.append({"type": "divider"})
    _add_group(busy, "⛔", "BUSY")
    blocks.append({"type": "divider"})
    _add_group(leave, "🏖️", "ON LEAVE")
    blocks.append({"type": "divider"})

    # Safety trim for Slack's 150-block limit
    if len(blocks) > MAX_BLOCKS:
        blocks = blocks[:MAX_BLOCKS]
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "⚠️ _Response truncated due to Slack block limits._",
            },
        })

    return blocks


def build_my_calendar_blocks(
    entries: list[AvailabilityEntry],
    user_name: str,
) -> list[dict]:
    """Build Block Kit blocks showing a user's own calendar entries."""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📅 My Calendar — {user_name}",
                "emoji": True,
            },
        },
    ]

    if not entries:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "_No upcoming availability entries._\nUse `/calendar status busy on May 5 from 10am to 12pm for Reason` to add one.",
            },
        })
        return blocks

    blocks.append({"type": "divider"})

    for entry in entries[:20]:  # Cap at 20 entries
        emoji = STATUS_EMOJI.get(entry.status, "📅")
        date_display = _fmt_date_range(entry)
        time_display = _fmt_time_range(entry)
        reason = f" — _{entry.reason}_" if entry.reason else ""

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *{entry.status.upper()}* · {date_display}\n"
                    f"    🕐 {time_display}{reason}"
                ),
            },
        })

    if len(entries) > 20:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"_... and {len(entries) - 20} more entries_",
            },
        })

    blocks.append({"type": "divider"})
    return blocks


def build_clear_confirmation_blocks(date_str: str, deleted: int) -> list[dict]:
    """Build Block Kit blocks confirming a calendar clear operation."""
    if deleted == 0:
        text = f"ℹ️ No entries found for *{date_str}* to clear."
    else:
        text = f"🗑️ Cleared *{deleted}* availability {'entry' if deleted == 1 else 'entries'} for *{date_str}*."

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        },
    ]


def build_usage_blocks() -> list[dict]:
    """Build Block Kit blocks with /calendar usage instructions."""
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📅 Calendar — Usage Guide",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Set your availability:*\n"
                    "`/calendar status busy on May 5 from 10am to 12pm for Client meeting`\n"
                    "`/calendar status leave on May 5-10 for Vacation`\n"
                    "`/calendar status free on May 5`\n\n"
                    "*See who's available:*\n"
                    "`/calendar who-is-free May 5`\n\n"
                    "*View your calendar:*\n"
                    "`/calendar my-calendar`\n\n"
                    "*View team schedule overview:*\n"
                    "`/calendar team-calendar`\n\n"
                    "*Clear an entry:*\n"
                    "`/calendar clear May 5`"
                ),
            },
        },
        {"type": "divider"},
    ]


def build_team_calendar_blocks(
    entries: list[AvailabilityEntry],
    exclude_header: bool = False,
) -> list[dict]:
    """Build Block Kit blocks showing upcoming entries for the entire team."""
    blocks: list[dict] = []
    if not exclude_header:
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📅 Team Schedule Overview",
                "emoji": True,
            },
        })

    if not entries:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "_No upcoming team availability entries._",
            },
        })
        return blocks

    blocks.append({"type": "divider"})

    # Group entries by date
    from collections import defaultdict
    from datetime import datetime as _datetime
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry.date_start].append(entry)

    # Sort dates
    for d in sorted(grouped.keys())[:7]:  # Show up to next 7 unique dates
        day_entries = grouped[d]
        try:
            date_formatted = _datetime.strptime(d, "%Y-%m-%d").strftime("%A, %b %d")
        except Exception:
            date_formatted = d
        
        lines = [f"*📅 {date_formatted}*"]
        for entry in day_entries:
            emoji = STATUS_EMOJI.get(entry.status, "📅")
            name = entry.user_display_name or entry.user_name or entry.user_id
            time_display = _fmt_time_range(entry)
            reason = f" — _{entry.reason}_" if entry.reason else ""
            lines.append(f"  • <@{entry.user_id}> ({name}) is *{entry.status.upper()}* {emoji}\n    🕐 {time_display}{reason}")
            
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(lines),
            },
        })
        blocks.append({"type": "divider"})

    return blocks

