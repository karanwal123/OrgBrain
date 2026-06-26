"""Channel summarization via Gemini and Block Kit modal formatting."""

from collections import Counter
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from vertexai.generative_models import GenerationConfig, GenerativeModel

from backend.cache import TTLCache
from backend.channel_history import build_user_lookup
from backend.config import get_settings
from backend.extraction import ExtractionError, _strip_json_fences
from backend.schemas import ChannelMemoryState, ChannelSummary, MemoryUnitType, TimelineEvent, MemoryRetrievalHit

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
_AI_CACHE_ENABLED = _SETTINGS.ai_cache_enabled
_SUMMARY_CACHE: TTLCache[ChannelSummary] = TTLCache(
    ttl_seconds=_SETTINGS.ai_cache_ttl_seconds,
    max_entries=_SETTINGS.ai_cache_max_entries,
)

HEALTH_EMOJI = {
    "healthy": "🟢",
    "warning": "🟡",
    "issues": "🔴",
}

TIMELINE_ICONS = {
    "decision": "📌",
    "problem": "🔴",
    "solution": "✓",
    "action": "📋",
    "event": "•",
}

MAX_MESSAGES_CHARS = 80_000

SLACK_USER_ID_PATTERN = re.compile(r"^U[A-Z0-9]{8,}$", re.IGNORECASE)

SUMMARY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "your",
    "you",
    "our",
    "are",
    "was",
    "were",
    "not",
    "but",
    "can",
    "all",
    "any",
    "has",
    "had",
    "into",
    "out",
    "need",
    "just",
    "now",
    "fix",
    "issue",
    "issues",
    "please",
    "thanks",
    "okay",
    "ok",
    "team",
    "today",
    "yesterday",
}

SUMMARIZE_PROMPT = """You are an organizational intelligence analyst summarizing Slack channel activity.

Channel: {channel}
Timeframe: {timeframe}
Message count: {message_count}

Messages (chronological JSON):
{messages_json}

Analyze the discussion and return JSON only with this structure:
{{
  "channel": "{channel}",
  "timeframe": "{timeframe}",
  "health": "healthy" | "warning" | "issues",
  "participant_count": <number of unique speakers>,
  "decision_count": <number of decisions made>,
  "timeline": [
    {{
      "type": "decision" | "problem" | "solution" | "action",
      "action_type": "Decision Made" | "Problem Reported" | "Solution Applied" | etc,
      "text": "what happened",
      "speaker": "person name from messages",
      "timestamp": "time label from message or relative"
    }}
  ],
  "decisions": [
    {{"decision": "...", "decided_by": "name", "approved_by": "name or null", "status": "approved"}}
  ],
  "problems": [
    {{
      "problem": "...",
      "reported_by": "name",
      "solution": "resolution details, or null if unresolved",
      "fixed_by": "name of person, or null if unresolved",
      "timestamp": "optional",
      "impact": "optional"
    }}
  ],
  "action_items": [
    {{"item": "...", "owner": "name", "due": "optional", "status": "pending" | "done"}}
  ],
  "narrative": "2-4 sentence prose summary of the channel activity"
}}

Rules:
- Output valid JSON only, no markdown fences
- Base everything ONLY on the messages provided
- timeline: max 8 events, most important first
- Use human display names from the "user" field in messages — NEVER Slack user IDs like U0BASEAQUCQ
- Never use <@U...> format in output; use plain names only
- health: "issues" if unresolved blockers, "warning" if open items, "healthy" if smooth progress
- If few messages, keep all sections brief and honest
"""


def _messages_to_prompt_json(messages: list[dict[str, Any]]) -> str:
    """Serialize messages for Gemini, respecting char cap."""
    lines: list[str] = []
    total = 0
    for msg in messages:
        entry = json.dumps(
            {
                "time": msg.get("time_label"),
                "user": msg.get("user_real_name") or msg.get("user_name"),
                "text": msg.get("text"),
            },
            ensure_ascii=False,
        )
        if total + len(entry) > MAX_MESSAGES_CHARS:
            break
        lines.append(entry)
        total += len(entry)
    return "[\n" + ",\n".join(lines) + "\n]"


def _model_cache_key(model: GenerativeModel) -> str:
    """Build a stable model identifier for cache key partitioning."""
    model_name = getattr(model, "_model_name", None) or getattr(model, "model_name", None)
    return str(model_name) if model_name else f"model@{id(model)}"


def _build_fallback_narrative(messages: list[dict[str, Any]], timeframe: str) -> str:
    """Build a concise human-readable summary from raw messages."""
    speakers = {
        m.get("user_real_name") or m.get("user_name")
        for m in messages
        if m.get("user_real_name") or m.get("user_name")
    }
    terms: list[str] = []
    for msg in messages:
        text = str(msg.get("text", "")).lower()
        terms.extend(
            token
            for token in re.findall(r"[a-z][a-z0-9+-]{2,}", text)
            if token not in SUMMARY_STOPWORDS
        )

    top_terms = [term for term, _ in Counter(terms).most_common(3)]
    latest_text = str(messages[-1].get("text", "")).strip() if messages else ""
    parts = [
        f"{len(messages)} message(s) from {len(speakers)} participant(s) in {timeframe}.",
    ]
    if top_terms:
        parts.append(f"Main topics: {', '.join(top_terms)}.")
    if latest_text:
        parts.append(f"Latest update: {latest_text[:120]}.")
    return " ".join(parts)


def _parse_json_payload(raw_text: str) -> Any:
    """Parse JSON from model output, tolerating extra prose around the payload.

    Strategies tried in order:
    1. Direct json.loads after stripping markdown fences
    2. json.JSONDecoder.raw_decode from every '{' and '[' position
    3. Regex extraction of the outermost { ... } block
    4. Trailing-comma cleanup + retry
    """
    cleaned = _strip_json_fences(raw_text)

    # Strategy 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: raw_decode from every candidate start position
    decoder = json.JSONDecoder()
    for char in ("{", "["):
        start = -1
        while True:
            start = cleaned.find(char, start + 1)
            if start < 0:
                break
            try:
                payload, _ = decoder.raw_decode(cleaned[start:])
                return payload
            except json.JSONDecodeError:
                continue

    # Strategy 3: regex to grab the outermost { ... } block
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        candidate = obj_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Strategy 4: strip trailing commas before } or ]
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("No valid JSON found in model output", cleaned, 0)


def _parse_channel_summary(raw_text: str, channel: str, timeframe: str) -> ChannelSummary:
    """Parse Gemini JSON into ChannelSummary with tolerant shapes."""
    try:
        payload = _parse_json_payload(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")

        payload.setdefault("channel", channel)
        payload.setdefault("timeframe", timeframe)
        return ChannelSummary.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        # Log the raw output for debugging
        preview = raw_text[:500] if raw_text else "(empty)"
        print(f"[PARSE FAIL] Channel summary raw LLM output ({len(raw_text)} chars):\n{preview}")
        logger.warning(
            "Channel summary parse failed (detail=%s). Raw preview: %s",
            str(exc),
            preview,
        )
        raise ExtractionError(
            "Failed to parse channel summary response",
            detail=str(exc),
        ) from exc


def fallback_channel_summary(
    channel: str,
    timeframe: str,
    messages: list[dict[str, Any]],
) -> ChannelSummary:
    """Minimal summary when Gemini fails."""
    speakers = {m.get("user_name") for m in messages if m.get("user_name")}
    return ChannelSummary(
        channel=channel,
        timeframe=timeframe,
        health="healthy" if messages else "warning",
        participant_count=len(speakers),
        decision_count=0,
        timeline=[
            TimelineEvent(
                type="event",
                action_type="Recent Activity",
                text=messages[-1].get("text", "")[:120] if messages else "No messages",
                speaker=messages[-1].get("user_name", "unknown") if messages else "—",
                timestamp=messages[-1].get("time_label", "") if messages else "",
            )
        ]
        if messages
        else [],
        narrative=_build_fallback_narrative(messages, timeframe) if messages else f"No messages found in {channel} for {timeframe}.",
    )


def summarize_channel(
    messages: list[dict[str, Any]],
    channel: str,
    timeframe: str,
    model: GenerativeModel,
    *,
    user_cache: Optional[dict[str, str]] = None,
    temperature: float = 0.2,
) -> ChannelSummary:
    """
    Summarize channel messages into structured ChannelSummary via Gemini.

    Args:
        messages: Formatted message dicts from channel_history.
        channel: Display channel name e.g. #hackathon-
        timeframe: Human timeframe label.
        model: Vertex AI GenerativeModel.

    Returns:
        ChannelSummary for Block Kit rendering.
    """
    if not messages:
        return ChannelSummary(
            channel=channel,
            timeframe=timeframe,
            health="warning",
            narrative=f"No messages found in {channel} for {timeframe}.",
        )

    messages_json = _messages_to_prompt_json(messages)
    msg_hash = hashlib.sha1(messages_json.encode()).hexdigest()[:10]
    model_key = _model_cache_key(model)
    cache_key = (
        model_key,
        round(float(temperature), 3),
        channel,
        timeframe,
        messages_json,
    )
    print(f"   [SUMMARIZE Cache Key] channel={channel} timeframe={timeframe} msgs={len(messages)} hash={msg_hash} model={model_key}")
    if _AI_CACHE_ENABLED:
        cached = _SUMMARY_CACHE.get(cache_key)
        if cached is not None:
            print(f"")
            print(f"   +------------------------------------------------------+")
            print(f"   |  [CACHE HIT] SUMMARY CACHE HIT - No Gemini needed!   |")
            print(f"   |  Channel : {channel:<42}|")
            print(f"   |  Hash    : {msg_hash:<42}|")
            print(f"   |  Serving cached result instantly (O(1))              |")
            print(f"   +------------------------------------------------------+")
            print(f"")
            lookup = build_user_lookup(messages, user_cache)
            return apply_name_resolution(cached.model_copy(deep=True), lookup)

    prompt = SUMMARIZE_PROMPT.format(
        channel=channel,
        timeframe=timeframe,
        message_count=len(messages),
        messages_json=messages_json,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
                max_output_tokens=4096,
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            fallback = fallback_channel_summary(channel, timeframe, messages)
            if _AI_CACHE_ENABLED:
                _SUMMARY_CACHE.set(cache_key, fallback)
                print(f"[SUMMARIZE] Empty response — cached fallback for {channel}")
            return apply_name_resolution(fallback.model_copy(deep=True), build_user_lookup(messages, user_cache))
        summary = _parse_channel_summary(raw_text, channel, timeframe)
        print(f"[SUMMARIZE] OK for {channel}: health={summary.health}, timeline={len(summary.timeline)} events, action_items={len(summary.action_items)}")
        if _AI_CACHE_ENABLED:
            _SUMMARY_CACHE.set(cache_key, summary)
            print(f"[SUMMARIZE] Stored in cache (key hash={msg_hash})")
        lookup = build_user_lookup(messages, user_cache)
        return apply_name_resolution(summary.model_copy(deep=True), lookup)
    except ExtractionError as exc:
        print(f"[SUMMARIZE] WARN: Parse failed for {channel} -- caching fallback so next call is instant. Detail: {exc.detail}")
        logger.warning("Channel summary parse failed: %s", exc.message)
        fallback = fallback_channel_summary(channel, timeframe, messages)
        if _AI_CACHE_ENABLED:
            _SUMMARY_CACHE.set(cache_key, fallback)
        lookup = build_user_lookup(messages, user_cache)
        return apply_name_resolution(fallback.model_copy(deep=True), lookup)
    except Exception as exc:
        logger.exception("Channel summarization failed")
        fallback = fallback_channel_summary(channel, timeframe, messages)
        if _AI_CACHE_ENABLED:
            _SUMMARY_CACHE.set(cache_key, fallback)
        return apply_name_resolution(
            fallback.model_copy(deep=True),
            build_user_lookup(messages, user_cache),
        )


def _is_slack_user_id(value: str) -> bool:
    """Return True if value looks like a Slack user ID."""
    clean = value.strip().lstrip("@").strip("<>")
    return bool(SLACK_USER_ID_PATTERN.match(clean))


def resolve_person_name(value: Optional[str], lookup: dict[str, str]) -> str:
    """
    Resolve a speaker/owner field to a human display name.

    Handles raw IDs (U0BASE...), <@U...> mentions, and @Name variants.
    """
    if not value:
        return "unknown"

    raw = value.strip()
    mention_match = re.match(r"^<@(U[A-Z0-9]+)>$", raw, re.IGNORECASE)
    if mention_match:
        raw = mention_match.group(1)

    clean = raw.lstrip("@").strip()
    if clean in lookup:
        return lookup[clean]
    if clean.upper() in lookup:
        return lookup[clean.upper()]
    if clean.lower() in lookup:
        return lookup[clean.lower()]

    if _is_slack_user_id(clean):
        return lookup.get(clean, lookup.get(clean.upper(), "Unknown teammate"))

    return clean


def _resolve_text_mentions(text: str, lookup: dict[str, str]) -> str:
    """Replace any remaining <@U...> or bare user IDs in free text."""

    def repl(match: re.Match) -> str:
        uid = match.group(1)
        return lookup.get(uid, lookup.get(uid.upper(), "Unknown teammate"))

    resolved = re.sub(r"<@(U[A-Z0-9]+)>", repl, text, flags=re.IGNORECASE)
    for uid, name in lookup.items():
        if uid.startswith("U") and len(uid) > 8:
            resolved = re.sub(rf"\b{re.escape(uid)}\b", name, resolved, flags=re.IGNORECASE)
    return resolved


def apply_name_resolution(summary: ChannelSummary, lookup: dict[str, str]) -> ChannelSummary:
    """Replace Slack user IDs in all summary person fields with display names."""
    timeline = [
        event.model_copy(
            update={
                "speaker": resolve_person_name(event.speaker, lookup),
                "text": _resolve_text_mentions(event.text, lookup),
            }
        )
        for event in summary.timeline
    ]
    decisions = [
        d.model_copy(
            update={
                "decision": _resolve_text_mentions(d.decision, lookup),
                "decided_by": resolve_person_name(d.decided_by, lookup),
                "approved_by": resolve_person_name(d.approved_by, lookup)
                if d.approved_by
                else None,
            }
        )
        for d in summary.decisions
    ]
    problems = [
        p.model_copy(
            update={
                "problem": _resolve_text_mentions(p.problem, lookup),
                "reported_by": resolve_person_name(p.reported_by, lookup),
                "solution": _resolve_text_mentions(p.solution, lookup) if p.solution else None,
                "fixed_by": resolve_person_name(p.fixed_by, lookup) if p.fixed_by else None,
            }
        )
        for p in summary.problems
    ]
    action_items = [
        a.model_copy(
            update={
                "item": _resolve_text_mentions(a.item, lookup),
                "owner": resolve_person_name(a.owner, lookup),
            }
        )
        for a in summary.action_items
    ]
    return summary.model_copy(
        update={
            "timeline": timeline,
            "decisions": decisions,
            "problems": problems,
            "action_items": action_items,
            "narrative": _resolve_text_mentions(summary.narrative, lookup),
        }
    )


def _mention(name: str) -> str:
    """Format a resolved person name for display (plain name, not @user_id)."""
    clean = resolve_person_name(name, {})
    if _is_slack_user_id(clean):
        return "Unknown teammate"
    return clean


def _ensure_complete_sentence(text: str) -> str:
    """Ensure the text ends with a sentence-ending punctuation."""
    if not text or text.strip() == "":
        return text
    text = text.strip()
    if text[-1] in ".!?":
        return text
    return text + "."


def format_timeline_card(event: TimelineEvent) -> dict[str, Any]:
    """Build a Block Kit section for one timeline event."""
    icon = TIMELINE_ICONS.get(event.type, "•")
    # Use a compact two-line layout: headline and body, with speaker/timestamp meta
    headline = f"{icon} *{_escape_mrkdwn(event.action_type)}*"
    body = _escape_mrkdwn(event.text)
    meta = f"_{_mention(event.speaker)} · {event.timestamp}_"
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"{headline}\n{body}\n{meta}"},
    }


def format_loading_modal(channel: str, timeframe: str) -> dict[str, Any]:
    """Build a loading modal view opened immediately on /summarize."""
    label = channel if channel.startswith("#") else f"#{channel}" if channel else "channel"
    return {
        "type": "modal",
        "callback_id": "summary_loading",
        "title": {"type": "plain_text", "text": "Summarizing..."},
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{_escape_mrkdwn(label)}* • {_escape_mrkdwn(timeframe)}\n\n:hourglass_flowing_sand: Analyzing channel messages with Gemini...",
                },
            },
        ],
    }


def format_error_modal(error_message: str) -> dict[str, Any]:
    """Build an error modal view."""
    return {
        "type": "modal",
        "callback_id": "summary_error",
        "title": {"type": "plain_text", "text": "Summary Failed"},
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":warning: {_escape_mrkdwn(error_message)}",
                },
            },
        ],
    }


def _escape_mrkdwn(text: str) -> str:
    """Escape characters that break Slack mrkdwn."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_block_kit(summary: ChannelSummary) -> dict[str, Any]:
    """
    Build the full summary modal view from ChannelSummary.

    Args:
        summary: Structured channel summary from Gemini.

    Returns:
        Slack modal view dict for views.open / views.update.
    """
    health_icon = HEALTH_EMOJI.get(summary.health, "🟢")
    health_label = summary.health.capitalize()

    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{_escape_mrkdwn(summary.channel)}* • {_escape_mrkdwn(summary.timeframe)}\n"
                    f"{health_icon} {health_label} | "
                    f"{summary.participant_count} participants | "
                    f"{summary.decision_count} decisions"
                ),
            },
        },
        {"type": "divider"},
    ]

    if summary.timeline:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Timeline*"},
            }
        )
        for event in summary.timeline[:5]:
            blocks.append(format_timeline_card(event))
        blocks.append({"type": "divider"})

    if summary.decisions:
        decision_lines = []
        for d in summary.decisions:
            line = f"✓ {d.decision} — {_mention(d.decided_by)}"
            if d.approved_by:
                line += f", approved by {_mention(d.approved_by)}"
            decision_lines.append(line)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Decisions Made*\n" + "\n".join(decision_lines),
                },
            }
        )
        blocks.append({"type": "divider"})

    if summary.problems:
        problem_lines = []
        for p in summary.problems:
            ts = f" ({p.timestamp})" if p.timestamp else ""
            line = f"🔴 *{p.problem}*{ts}\n   Reported by: {_mention(p.reported_by)}"
            if p.solution:
                fixed_by_str = f" — {_mention(p.fixed_by)}" if p.fixed_by else ""
                line += f"\n   ✓ Solution: {p.solution}{fixed_by_str}"
            if p.impact:
                line += f"\n   Impact: {p.impact}"
            problem_lines.append(line)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Problems & Solutions*\n" + "\n\n".join(problem_lines),
                },
            }
        )
        blocks.append({"type": "divider"})

    if summary.action_items:
        action_lines = []
        for a in summary.action_items:
            icon = "✓" if a.status == "done" else "☐"
            line = f"{icon} {a.item} — Owner: {_mention(a.owner)}"
            if a.due:
                line += f" (Due: {a.due})"
            action_lines.append(line)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Action Items*\n" + "\n".join(action_lines),
                },
            }
        )
        blocks.append({"type": "divider"})

    if summary.narrative:
        narrative = _ensure_complete_sentence(summary.narrative)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary*\n{_escape_mrkdwn(narrative)}",
                },
            }
        )

    generated_at = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Generated by OrgBrain • {generated_at}",
                }
            ],
        }
    )

    return {
        "type": "modal",
        "callback_id": "summary_modal",
        "title": {
            "type": "plain_text",
            "text": f"Summary: {summary.channel.lstrip('#')[:20]}",
        },
        "blocks": blocks,
    }

def format_memory_modal(state, channel_name):
    """Build a Block Kit modal from pre-built ChannelMemoryState. O(1): no Slack fetch, no Gemini."""
    from datetime import datetime, timezone
    from backend.schemas import MemoryUnitType

    now = datetime.now(timezone.utc)
    age_str = "just now"
    if state.last_summary_ts:
        age_sec = int((now - state.last_summary_ts.astimezone(timezone.utc)).total_seconds())
        if age_sec < 60:
            age_str = "just now"
        elif age_sec < 3600:
            age_str = f"{age_sec // 60}m ago"
        else:
            age_str = f"{age_sec // 3600}h ago"

    label = channel_name if channel_name.startswith("#") else f"#{channel_name}"
    unit_count = len(state.memory_store)

    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*{esc(label)}* -- Persistent Memory\n"
            f":zap: Served from memory ({unit_count} units, updated {age_str})"
        )}},
        {"type": "divider"},
    ]

    by_type = {}
    for unit in sorted(state.memory_store, key=lambda u: u.importance, reverse=True):
        by_type.setdefault(unit.unit_type.value, []).append(unit)

    type_labels = {
        "decision":         ("Decisions",    ">>"),
        "problem":          ("Problems",     "[!]"),
        "unresolved_issue": ("Open Issues",  "[?]"),
        "action_item":      ("Action Items", "->"),
        "agreement":        ("Agreements",   "[ok]"),
        "context":          ("Context",      "--"),
    }

    shown_any = False
    for unit_type_val, (section_label, icon) in type_labels.items():
        units = by_type.get(unit_type_val, [])
        if not units:
            continue
        lines = []
        for unit in units[:5]:
            flag = " [OPEN]" if unit.unresolved else ""
            owners = f" -- {', '.join(unit.owners)}" if unit.owners else ""
            lines.append(f"{icon} {esc(unit.summary)}{owners}{flag}")


def format_memory_modal(state, channel_name):
    """Build a Block Kit modal from pre-built ChannelMemoryState. O(1): no Slack fetch, no Gemini."""
    from datetime import datetime, timezone
    from backend.schemas import MemoryUnitType

    now = datetime.now(timezone.utc)
    age_str = "just now"
    if state.last_summary_ts:
        age_sec = int((now - state.last_summary_ts.astimezone(timezone.utc)).total_seconds())
        if age_sec < 60:
            age_str = "just now"
        elif age_sec < 3600:
            age_str = f"{age_sec // 60}m ago"
        else:
            age_str = f"{age_sec // 3600}h ago"

    label = channel_name if channel_name.startswith("#") else f"#{channel_name}"
    unit_count = len(state.memory_store)

    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*{esc(label)}* -- Persistent Memory\n"
            f":zap: Served from memory ({unit_count} units, updated {age_str})"
        )}},
        {"type": "divider"},
    ]

    by_type = {}
    for unit in sorted(state.memory_store, key=lambda u: u.importance, reverse=True):
        by_type.setdefault(unit.unit_type.value, []).append(unit)

    type_labels = {
        "decision":         ("Decisions",    ">>"),
        "problem":          ("Problems",     "[!]"),
        "unresolved_issue": ("Open Issues",  "[?]"),
        "action_item":      ("Action Items", "->"),
        "agreement":        ("Agreements",   "[ok]"),
        "context":          ("Context",      "--"),
    }

    shown_any = False
    for unit_type_val, (section_label, icon) in type_labels.items():
        units = by_type.get(unit_type_val, [])
        if not units:
            continue
        lines = []
        for unit in units[:5]:
            flag = " [OPEN]" if unit.unresolved else ""
            owners = f" -- {', '.join(unit.owners)}" if unit.owners else ""
            lines.append(f"{icon} {esc(unit.summary)}{owners}{flag}")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{section_label}*\n" + "\n".join(lines)}})
        blocks.append({"type": "divider"})
        shown_any = True

    if not shown_any:
        ctx = state.cached_summary_state or state.compressed_context
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Memory Snapshot*\n{esc(ctx[:2000])}"}})

    generated_at = now.strftime("%b %d, %Y %H:%M UTC")
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"OrgBrain Persistent Memory -- {generated_at} -- No Gemini call made"}]})

    return {
        "type": "modal",
        "callback_id": "summary_modal",
        "title": {"type": "plain_text", "text": f"Memory: {label.lstrip('#')[:20]}"},
        "blocks": blocks,
    }


def format_recall_results(query: str, hits: list[MemoryRetrievalHit]) -> dict[str, Any]:
    """Build a Block Kit message structure for search recall results."""
    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"🔍 *Recall Results for:* \"{esc(query)}\""
            }
        },
        {"type": "divider"}
    ]

    type_emojis = {
        "decision": "📌",
        "problem": "🔴",
        "unresolved_issue": "❓",
        "action_item": "📋",
        "agreement": "🤝",
        "context": "💬",
    }

    if not hits:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "_No memories found matching your query._"
            }
        })
    else:
        for hit in hits:
            emoji = type_emojis.get(hit.unit_type.value if hasattr(hit.unit_type, "value") else str(hit.unit_type), "•")
            owners_str = f" · Owners: {', '.join(hit.owners)}" if hit.owners else ""
            tags_str = f" · Tags: {', '.join(hit.tags)}" if hit.tags else ""
            score_percentage = int(hit.score * 100)
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{esc(hit.summary)}*\n"
                        f"Scope: <#{hit.channel_id}> · Score: {score_percentage}%{owners_str}{tags_str}"
                    )
                }
            })
            blocks.append({"type": "divider"})

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "Org Brain · Semantic Search Recall"
            }
        ]
    })

    return {"blocks": blocks}
