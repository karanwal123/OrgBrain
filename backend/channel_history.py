"""Fetch and parse Slack channel history for summarization."""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backend.cache import TTLCache

# Cache fetched messages for 2 minutes — ensures repeated /summarize calls within the
# window use IDENTICAL messages_json, enabling _SUMMARY_CACHE hits downstream.
_MESSAGES_CACHE: TTLCache[tuple] = TTLCache(ttl_seconds=120, max_entries=200)

logger = logging.getLogger(__name__)

SLACK_USER_ID_PATTERN = re.compile(r"^U[A-Z0-9]{8,}$", re.IGNORECASE)
SLACK_MENTION_PATTERN = re.compile(r"<@(U[A-Z0-9]+)>")

BOT_SUBTYPES = frozenset(
    {"bot_message", "message_changed", "message_deleted", "channel_join", "channel_leave"}
)

TIMEFRAME_PATTERNS = (
    (re.compile(r"(?i)last\s+(\d+)\s*days?"), "days"),
    (re.compile(r"(?i)last\s+(\d+)\s*hours?"), "hours"),
    (re.compile(r"(?i)last\s+(\d+)\s*h\b"), "hours"),
    (re.compile(r"(?i)last\s+(\d+)\s*d\b"), "days"),
    (re.compile(r"(?i)(\d+)\s*days?"), "days"),
    (re.compile(r"(?i)(\d+)\s*hours?"), "hours"),
    (re.compile(r"(?i)1\s*week"), "week"),
    (re.compile(r"(?i)(\d+)\s*weeks?"), "weeks"),
    (re.compile(r"(?i)today"), "today"),
)


class ChannelHistoryError(Exception):
    """Raised when channel history cannot be fetched."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def parse_timeframe(text: str) -> tuple[float, str]:
    """
    Parse a timeframe string into oldest Unix timestamp and display label.

    Args:
        text: e.g. "last 2 days", "1 week", "today"

    Returns:
        Tuple of (oldest_ts, human_label). Defaults to last 24 hours.
    """
    text = text.strip()
    now = datetime.now(timezone.utc)

    if not text:
        return (now - timedelta(hours=24)).timestamp(), "Last 24 hours"

    lower = text.lower()
    if re.search(r"(?i)\btoday\b", lower):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), "Today"

    for pattern, unit in TIMEFRAME_PATTERNS:
        if unit == "today":
            continue
        match = pattern.search(text)
        if not match:
            continue

        if unit == "week":
            return (now - timedelta(weeks=1)).timestamp(), "Last 1 week"

        if unit == "weeks":
            n = int(match.group(1))
            return (now - timedelta(weeks=n)).timestamp(), f"Last {n} weeks"

        n = int(match.group(1))
        if unit == "days":
            return (now - timedelta(days=n)).timestamp(), f"Last {n} days"
        if unit == "hours":
            return (now - timedelta(hours=n)).timestamp(), f"Last {n} hours"

    return (now - timedelta(hours=24)).timestamp(), "Last 24 hours"


def parse_summarize_args(text: str) -> tuple[Optional[str], str]:
    """
    Parse /summarize command text into channel reference and timeframe.

    Examples:
        "#hackathon- last 2 days" -> ("hackathon-", "last 2 days")
        "last 24 hours" -> (None, "last 24 hours")
        "#payment-migration 1 week" -> ("payment-migration", "1 week")
        "*#hackathon-* • Last 24 hours" -> ("hackathon-", "Last 24 hours")  # Slack modal format

    Returns:
        (channel_name_without_hash or None, timeframe_string)
    """
    text = text.strip()
    if not text:
        return None, "last 24 hours"

    # Strip Slack markdown formatting that appears when the modal passes text back
    # e.g. "*#hackathon-* • Last 24 hours" -> "#hackathon- Last 24 hours"
    text = re.sub(r"\*([^*]+)\*", r"\1", text)   # *bold* -> plain
    text = re.sub(r"[•·|]", " ", text)             # bullet/pipe separators -> space
    text = re.sub(r"\s{2,}", " ", text).strip()   # collapse whitespace

    channel_match = re.match(r"#([\w-]+)\s*(.*)", text)
    if channel_match:
        channel = channel_match.group(1)
        remainder = channel_match.group(2).strip()
        return channel, remainder or "last 24 hours"

    return None, text


def resolve_channel_id(
    client: Any,
    channel_ref: Optional[str],
    current_channel_id: str,
) -> tuple[str, str]:
    """
    Resolve a channel name to Slack channel ID.

    Args:
        client: Slack WebClient.
        channel_ref: Channel name without #, or None for current channel.
        current_channel_id: Channel where the command was invoked.

    Returns:
        Tuple of (channel_id, display_name e.g. #hackathon-)

    Raises:
        ChannelHistoryError: If channel cannot be found.
    """
    if not channel_ref:
        try:
            info = client.conversations_info(channel=current_channel_id)
            name = info["channel"].get("name", "channel")
            return current_channel_id, f"#{name}"
        except Exception as exc:
            raise ChannelHistoryError(
                f"Could not resolve current channel: {exc}"
            ) from exc

    channel_ref_lower = channel_ref.lower().lstrip("#")
    cursor: Optional[str] = None

    for _ in range(20):
        try:
            resp = client.conversations_list(
                types="public_channel,private_channel",
                limit=200,
                cursor=cursor,
            )
        except Exception as exc:
            raise ChannelHistoryError(f"Failed to list channels: {exc}") from exc

        for ch in resp.get("channels", []):
            if ch.get("name", "").lower() == channel_ref_lower:
                return ch["id"], f"#{ch['name']}"

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    raise ChannelHistoryError(
        f"Channel #{channel_ref} not found. Check the name or invite @OrgBrain to the channel."
    )


def _get_user_name(client: Any, user_id: str, cache: dict[str, str]) -> str:
    """Resolve Slack user ID to display name with caching."""
    if user_id in cache:
        return cache[user_id]
    try:
        resp = client.users_info(user=user_id)
        user = resp["user"]
        profile = user.get("profile", {})
        real = (profile.get("real_name") or user.get("real_name") or "").strip()
        display = (profile.get("display_name") or "").strip()
        name = real or display or (user.get("name") or "").strip() or user_id
        cache[user_id] = name
        return name
    except Exception:
        logger.warning("Slack user lookup failed for %s", user_id)
        cache[user_id] = user_id
        return user_id


def _prime_user_cache(client: Any, cache: dict[str, str]) -> None:
    """Populate the cache with Slack users.list results when available."""
    cursor: Optional[str] = None

    while True:
        try:
            resp = client.users_list(limit=200, cursor=cursor)
        except Exception as exc:
            logger.warning("Slack users list lookup failed: %s", exc)
            return

        for user in resp.get("members", []):
            user_id = user.get("id", "")
            profile = user.get("profile", {})
            real = (profile.get("real_name") or user.get("real_name") or "").strip()
            display = (profile.get("display_name") or "").strip()
            name = real or display or (user.get("name") or "").strip()
            if not user_id or not name or SLACK_USER_ID_PATTERN.match(name):
                continue
            cache[user_id] = name
            cache[user_id.upper()] = name
            cache[name.lower()] = name

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not isinstance(cursor, str) or not cursor:
            break


def resolve_mentions_in_text(text: str, client: Any, cache: dict[str, str]) -> str:
    """Replace Slack <@U123> mentions in message text with display names."""

    def repl(match: re.Match) -> str:
        return _get_user_name(client, match.group(1), cache)

    return SLACK_MENTION_PATTERN.sub(repl, text)


def build_user_lookup(
    messages: list[dict[str, Any]],
    user_cache: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Build a lookup map from Slack user IDs and name variants to display names.

    Args:
        messages: Message dicts from fetch_channel_messages.
        user_cache: Optional Slack user ID cache from fetch (includes mentioned users).

    Returns:
        Mapping e.g. {"U0BASEAQUCQ": "Aditya", "aditya": "Aditya"}.
    """
    lookup: dict[str, str] = {}

    def add_user(user_id: str, user_name: str) -> None:
        clean_name = user_name.strip()
        if not user_id or not clean_name or SLACK_USER_ID_PATTERN.match(clean_name):
            return
        lookup[user_id] = clean_name
        lookup[user_id.upper()] = clean_name
        lookup[clean_name.lower()] = clean_name

    if user_cache:
        for user_id, user_name in user_cache.items():
            add_user(user_id, user_name)

    for msg in messages:
        add_user(msg.get("user_id", ""), msg.get("user_real_name") or msg.get("user_name") or "")
    return lookup


def fetch_channel_messages(
    client: Any,
    channel_id: str,
    oldest_ts: float,
    *,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Fetch channel messages since oldest_ts for summarization.

    Args:
        client: Slack WebClient.
        channel_id: Target channel ID.
        oldest_ts: Oldest message timestamp (Unix).
        limit: Maximum messages to return.

    Returns:
        Tuple of message dicts (ts, user_id, user_name, text, time_label) and
        a user ID → display name cache (authors plus @mentioned users).

    Raises:
        ChannelHistoryError: If history cannot be fetched or channel is inaccessible.
    """
    # Bucket oldest_ts to the nearest 3600-second (1-hour) boundary so requests
    # within the same clock-hour share the same cache key. Using 60s was too fine —
    # "last 24 hours" shifts by seconds each call, causing constant cache misses.
    ts_bucket = int(oldest_ts // 3600) * 3600
    cache_key = (channel_id, ts_bucket, limit)
    cached = _MESSAGES_CACHE.get(cache_key)
    if cached is not None:
        cached_messages, cached_user_cache = cached
        print(f"   [Messages Cache HIT] Reusing {len(cached_messages)} cached messages for channel {channel_id} (bucket={ts_bucket})")
        return list(cached_messages), dict(cached_user_cache)

    print(f"   [Messages Cache MISS] Fetching fresh messages from Slack for channel {channel_id} (bucket={ts_bucket})")

    messages: list[dict[str, Any]] = []
    user_cache: dict[str, str] = {}
    cursor: Optional[str] = None

    _prime_user_cache(client, user_cache)

    def resolve_message_user_name(msg: dict[str, Any]) -> str:
        profile = msg.get("user_profile") or {}
        real_name = (profile.get("real_name") or "").strip()
        display_name = (profile.get("display_name") or "").strip()
        fallback_name = real_name or display_name or (msg.get("username") or "").strip() or (msg.get("user_name") or "").strip()
        user_id = msg.get("user", "unknown")
        if fallback_name and not SLACK_USER_ID_PATTERN.match(fallback_name):
            user_cache[user_id] = fallback_name
            user_cache[user_id.upper()] = fallback_name
            user_cache[fallback_name.lower()] = fallback_name
            return fallback_name
        return _get_user_name(client, user_id, user_cache)

    while len(messages) < limit:
        try:
            resp = client.conversations_history(
                channel=channel_id,
                oldest=str(oldest_ts),
                limit=min(100, limit - len(messages)),
                cursor=cursor,
            )
        except Exception as exc:
            err = str(exc).lower()
            if "not_in_channel" in err or "channel_not_found" in err:
                raise ChannelHistoryError(
                    "Bot is not in this channel. Run `/invite @OrgBrain` first."
                ) from exc
            raise ChannelHistoryError(f"Failed to fetch channel history: {exc}") from exc

        batch = resp.get("messages", [])
        if not batch:
            break

        for msg in batch:
            if msg.get("bot_id") or msg.get("subtype") in BOT_SUBTYPES:
                continue
            text = msg.get("text", "").strip()
            if not text:
                continue

            user_id = msg.get("user", "unknown")
            ts = float(msg.get("ts", 0))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            normalized_text = resolve_mentions_in_text(text, client, user_cache)
            messages.append(
                {
                    "ts": ts,
                    "user_id": user_id,
                    "user_name": resolve_message_user_name(msg),
                    "user_real_name": (msg.get("user_profile") or {}).get("real_name", "").strip(),
                    "user_display_name": (msg.get("user_profile") or {}).get("display_name", "").strip(),
                    "text": normalized_text,
                    "time_label": dt.strftime("%b %d, %H:%M"),
                }
            )

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not isinstance(cursor, str) or not cursor:
            break

    messages.sort(key=lambda m: m["ts"])
    result = messages[:limit], dict(user_cache)
    _MESSAGES_CACHE.set(cache_key, result)
    print(f"   [Messages Cache SET] Cached {len(result[0])} messages for channel {channel_id} (bucket={ts_bucket}, TTL=120s)")
    return result
