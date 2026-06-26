"""Slack Bolt app — listens to messages, learns profiles, answers /who-knows."""

from datetime import datetime
import logging
import re
from typing import Any, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from vertexai.generative_models import GenerativeModel

from backend.config import configure_logging, get_settings, init_vertex_ai
from backend.extraction import (
    ExtractionError,
    extract_employee_info,
    extract_help_topics,
    fallback_profile_about,
    generate_match_reasons,
    generate_profile_about,
)
from backend.schemas import EmployeeProfile, ProfileAboutResult
from backend.storage import ProfileStorage
from backend.channel_history import (
    ChannelHistoryError,
    fetch_channel_messages,
    parse_summarize_args,
    parse_timeframe,
    resolve_channel_id,
)
from backend.memory import ChannelMemoryService, create_channel_memory_service
from backend.summarizer import (
    format_block_kit,
    format_error_modal,
    format_loading_modal,
    format_memory_modal,
    summarize_channel,
    format_recall_results,
)
from backend.calendar_parser import parse_calendar_command, validate_availability
from backend.calendar_storage import AvailabilityStorage
from backend.calendar_blocks import (
    build_clear_confirmation_blocks,
    build_my_calendar_blocks,
    build_status_confirmation_blocks,
    build_usage_blocks,
    build_who_is_free_blocks,
    build_team_calendar_blocks,
)
from backend.schemas import AvailabilityEntry, AvailabilityStatus
from backend.calendar_modal import build_calendar_modal, build_team_calendar_modal_view

logger = logging.getLogger(__name__)

BOT_MESSAGE_SUBTYPES = frozenset(
    {"bot_message", "message_changed", "message_deleted", "channel_join", "channel_leave"}
)

ABOUT_QUERY_PATTERNS = (
    re.compile(r"(?i)tell me about\s+(.+)"),
    re.compile(r"(?i)who is\s+(.+)"),
    re.compile(r"(?i)^about\s+(.+)"),
)


def is_processable_message(event: dict[str, Any]) -> bool:
    """Return True if the event is a user message worth extracting from."""
    if event.get("bot_id"):
        return False
    if event.get("subtype") in BOT_MESSAGE_SUBTYPES:
        return False
    return bool(event.get("text", "").strip())


def build_profile_mini_description(profile: EmployeeProfile) -> str:
    """Build a one-line mini description for a profile."""
    parts: list[str] = []
    if profile.role:
        parts.append(profile.role)
    if profile.team:
        parts.append(profile.team)
    if profile.skills:
        parts.append(f"Skills: {', '.join(profile.skills[:5])}")
    if profile.projects:
        parts.append(f"Projects: {', '.join(profile.projects[:2])}")
    if profile.areas_of_expertise:
        parts.append(f"Expertise: {', '.join(profile.areas_of_expertise[:2])}")
    return " · ".join(parts) if parts else "_No details recorded yet_"


def strip_bot_mention(text: str) -> str:
    """Remove leading @bot mention from message text."""
    if text.startswith("<@"):
        return text.split(">", 1)[-1].strip()
    return text.strip()


def extract_about_query(text: str) -> Optional[str]:
    """
    Extract a person name or Slack mention from a natural-language about query.

    Examples:
        "tell me about Aditya" -> "Aditya"
        "<@U123>" -> "<@U123>"
        "about Kubernetes expert" -> "Kubernetes expert"
    """
    text = strip_bot_mention(text)
    if not text:
        return None

    for pattern in ABOUT_QUERY_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    return text


def resolve_profile_from_query(
    query: str,
    storage: ProfileStorage,
    client: Any,
) -> Optional[EmployeeProfile]:
    """
    Resolve an EmployeeProfile from a Slack @mention or name string.

    Args:
        query: `<@U123>` mention or person name.
        storage: MongoDB profile storage.
        client: Slack WebClient for user lookup.

    Returns:
        Matching profile or None.
    """
    query = query.strip()
    if not query:
        return None

    mention = re.match(r"<@(\w+)>", query)
    if mention:
        user_id = mention.group(1)
        try:
            resp = client.users_info(user=user_id)
            user = resp["user"]
            profile_data = user.get("profile", {})
            candidates = [
                user.get("real_name"),
                profile_data.get("real_name"),
                profile_data.get("display_name"),
                user.get("name"),
            ]
            for name in candidates:
                if name:
                    found = storage.find_profile_by_name(name)
                    if found:
                        return found
            return None
        except Exception as exc:
            logger.warning("Slack user lookup failed for %s: %s", user_id, exc)
            return None

    return storage.find_profile_by_name(query)


def get_profile_about(
    profile: EmployeeProfile,
    model: Optional[GenerativeModel],
) -> ProfileAboutResult:
    """Generate or fallback profile about card."""
    if model is None:
        return fallback_profile_about(profile)
    try:
        return generate_profile_about(profile, model)
    except ExtractionError as exc:
        logger.warning("Profile about generation failed: %s", exc.message)
        return fallback_profile_about(profile)


def build_about_blocks(
    profile: EmployeeProfile,
    about: ProfileAboutResult,
) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks for /about results."""
    role_line = profile.role or "Team member"
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{profile.person}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{role_line}*",
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": about.description},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"> _{about.tagline}_",
            },
        },
    ]

    extras: list[str] = []
    if profile.skills:
        extras.append(f"*Skills:* {', '.join(profile.skills[:6])}")
    if profile.projects:
        extras.append(f"*Projects:* {', '.join(profile.projects[:3])}")
    if extras:
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": " · ".join(extras)}]}
        )

    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "Org Brain · organizational memory"}],
        }
    )
    return blocks


def _send_ephemeral_response(respond: Any, *, text: str, blocks: Optional[list[dict[str, Any]]] = None) -> None:
    """Send a slash-command fallback response without assuming channel membership."""
    payload: dict[str, Any] = {
        "response_type": "ephemeral",
        "text": text,
    }
    if blocks is not None:
        payload["blocks"] = blocks

    try:
        respond(**payload)
    except Exception as exc:
        logger.warning("Failed to send summarize fallback response: %s", exc)


def _send_command_response(
    respond: Any,
    *,
    text: str,
    blocks: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Send a slash-command response via the response_url-backed callback."""
    _send_ephemeral_response(respond, text=text, blocks=blocks)


def summarize_slash_command(
    command: dict[str, Any],
    client: Any,
    respond: Any,
    model: Optional[GenerativeModel],
    memory_service: Optional[ChannelMemoryService] = None,
) -> None:
    """Run the /summarize command flow."""
    text = command.get("text", "").strip()
    current_channel_id = command["channel_id"]
    trigger_id = command["trigger_id"]

    channel_ref, timeframe_text = parse_summarize_args(text)
    oldest_ts, timeframe_label = parse_timeframe(timeframe_text)

    display_channel = f"#{channel_ref}" if channel_ref else "this channel"
    loading_view = format_loading_modal(display_channel, timeframe_label)

    view_id = None
    modal_opened = False
    try:
        open_resp = client.views_open(trigger_id=trigger_id, view=loading_view)
        view_id = open_resp["view"]["id"]
        modal_opened = True
    except Exception as exc:
        logger.error("Failed to open loading modal: %s", exc)
        _send_ephemeral_response(
            respond,
            text=f"Could not open summary modal: {exc}. I will post results here when ready.",
        )
        # If the trigger expired we can still continue and post the finished
        # summary as an ephemeral message; for other failures, stop.
        err_text = str(exc).lower()
        if "expired_trigger_id" in err_text and model is not None:
            modal_opened = False
        else:
            return

    try:
        if model is None:
            raise ChannelHistoryError("Vertex AI is not available. Check GCP credentials.")

        channel_id, channel_name = resolve_channel_id(
            client, channel_ref, current_channel_id
        )

        # ----------------------------------------------------------------
        # Step 1: Check pre-built persistent memory (O(1), no Slack/Gemini)
        # ----------------------------------------------------------------
        if memory_service is not None:
            state = memory_service.get_state(channel_id)
            has_memory = bool(state.memory_store or state.cached_summary_state)
            memory_age_ok = False
            if state.last_summary_ts is not None:
                from datetime import datetime, timezone
                age_min = (datetime.now(timezone.utc) - state.last_summary_ts.astimezone(timezone.utc)).total_seconds() / 60
                memory_age_ok = age_min < 25  # fresher than one worker cycle

            if has_memory and memory_age_ok:
                print(f"\n[SUMMARIZE] MEMORY HIT for {channel_name} -- returning pre-built state instantly (no Gemini, no Slack fetch)")
                modal = format_memory_modal(state, channel_name)
                if modal_opened:
                    client.views_update(view_id=view_id, view=modal)
                else:
                    client.chat_postEphemeral(channel=channel_id, user=command.get("user_id"), blocks=modal["blocks"], text=f"Memory for {channel_name}")
                logger.info("Served memory-first summary for %s (%d units)", channel_name, len(state.memory_store))
                return
            else:
                print(f"   [SUMMARIZE] No fresh memory for {channel_name} (has_memory={has_memory}, age_ok={memory_age_ok}) -- falling back to live summarization")

        # ----------------------------------------------------------------
        # Step 2: Live fallback -- fetch from Slack + Gemini
        # ----------------------------------------------------------------
        messages, user_cache = fetch_channel_messages(client, channel_id, oldest_ts)

        if not messages:
            error_message = f"No messages found in {channel_name} for {timeframe_label}."
            if modal_opened:
                client.views_update(view_id=view_id, view=format_error_modal(error_message))
            else:
                client.chat_postEphemeral(channel=channel_id, user=command.get("user_id"), text=error_message)
            return

        summary = summarize_channel(
            messages,
            channel_name,
            timeframe_label,
            model,
            user_cache=user_cache,
        )
        if modal_opened:
            client.views_update(view_id=view_id, view=format_block_kit(summary))
        else:
            # Fallback: post the summary blocks as an ephemeral message to the user
            try:
                client.chat_postEphemeral(
                    channel=channel_id,
                    user=command.get("user_id"),
                    blocks=format_block_kit(summary)["blocks"],
                    text=f"Summary for {channel_name}",
                )
            except Exception:
                # As a last resort, send a simple ephemeral text response
                client.chat_postEphemeral(
                    channel=channel_id,
                    user=command.get("user_id"),
                    text=(f"Summary ready for {channel_name}, but failed to open modal."),
                )
        logger.info(
            "Channel summary generated for %s (%s, %d messages)",
            channel_name,
            timeframe_label,
            len(messages),
        )
    except ChannelHistoryError as exc:
        logger.warning("Channel history error: %s", exc.message)
        client.views_update(view_id=view_id, view=format_error_modal(exc.message))
    except Exception as exc:
        logger.exception("Summarize failed")
        client.views_update(
            view_id=view_id,
            view=format_error_modal(f"Summary failed: {exc}"),
        )


def recall_slash_command(
    command: dict[str, Any],
    client: Any,
    respond: Any,
    memory_service: Optional[ChannelMemoryService] = None,
) -> None:
    """Run the /recall command flow."""
    query = command.get("text", "").strip()
    print(f"   [Slack Command] Acknowledged: Processing /recall from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
    print(f"   Query: \"{query}\"")

    if not query:
        print("   [Slack Command] Response: Missing search query, sending usage instructions.")
        _send_command_response(
            respond,
            text="Usage: `/recall <search query>` — e.g. `/recall Redis timeout`"
        )
        return

    if memory_service is None:
        print("   [Slack Command] FAILED: Memory service not initialized.")
        _send_command_response(
            respond,
            text="❌ Semantic search memory service is currently unavailable."
        )
        return

    try:
        # Flush pending memories for the current channel to ensure up-to-date search results
        channel_id = command.get("channel_id")
        if channel_id:
            try:
                print(f"   [Slack Command] Flushing pending memory for channel: {channel_id}")
                memory_service.flush_channel(channel_id)
            except Exception as flush_exc:
                print(f"   [Slack Command] WARN: Failed to flush channel memory: {flush_exc}")

        # Query the memory store (search globally across channels)
        hits = memory_service.search(query, limit=5)
        blocks = format_recall_results(query, hits)
        _send_command_response(
            respond,
            blocks=blocks["blocks"],
            text=f"Recall results for {query}",
        )
        print("   [Slack Command] SUCCESS: Posted search results to user.")
    except Exception as exc:
        print(f"   [Slack Command] FAILED: /recall command failed: {exc}")
        _send_command_response(
            respond,
            text=f"❌ Failed to execute recall query: {exc}"
        )


def reply_with_about(
    query: str,
    storage: ProfileStorage,
    model: Optional[GenerativeModel],
    client: Any,
    channel_id: str,
    user_id: str,
) -> None:
    """Look up a person and post their profile card."""
    profile = resolve_profile_from_query(query, storage, client)
    if not profile:
        client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=(
                f"No profile found for *{query.strip()}* yet.\n"
                "Try `/intro` or `@OrgBrain I'm Name, Role, expert in X` to add them."
            ),
        )
        return

    about = get_profile_about(profile, model)
    blocks = build_about_blocks(profile, about)
    client.chat_postEphemeral(
        channel=channel_id,
        user=user_id,
        blocks=blocks,
        text=f"About {profile.person}",
    )


def build_who_knows_blocks(
    skill: str,
    profiles: list,
    match_reasons: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """
    Build Slack Block Kit blocks for /who-knows results.

    Args:
        skill: The skill search term.
        profiles: Matching EmployeeProfile instances.

    Returns:
        Block Kit block list.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Who knows {skill}?"},
        },
        {"type": "divider"},
    ]

    if not profiles:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"No profiles found for *{skill}* yet.\n"
                        "Post expertise in a channel and Org Brain will learn automatically."
                    ),
                },
            }
        )
        return blocks

    for profile in profiles:
        role_line = f"*{profile.role}*" if profile.role else "_Role unknown_"
        reason = (match_reasons or {}).get(profile.person)
        if reason:
            text = f"*{profile.person}* — {role_line}\n_{reason}_"
        else:
            mini = build_profile_mini_description(profile)
            text = f"*{profile.person}* — {role_line}\n{mini}"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Found {len(profiles)} profile(s) · powered by Org Brain",
                }
            ],
        }
    )
    return blocks


def build_help_blocks(
    topics: list[str],
    summary: str,
    profiles: list[EmployeeProfile],
    match_reasons: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build Slack Block Kit blocks for /help results."""
    topic_label = ", ".join(topics) if topics else "your issue"
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"People who can help with {topic_label}"},
        },
    ]

    if summary:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Your issue:* {summary}"},
            }
        )
    blocks.append({"type": "divider"})

    if not profiles:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"No experts found for *{topic_label}* yet.\n"
                        "Ask teammates to `@OrgBrain` with their skills, or use `/intro`."
                    ),
                },
            }
        )
        return blocks

    for i, profile in enumerate(profiles, start=1):
        reason = (match_reasons or {}).get(profile.person)
        if reason:
            line = f"_{reason}_"
        else:
            line = build_profile_mini_description(profile)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{i}. {profile.person}*\n{line}",
                },
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Matched {len(profiles)} person(s) · Org Brain organizational memory"
                    ),
                }
            ],
        }
    )
    return blocks


def resolve_help_profiles(
    query: str,
    model: Optional[GenerativeModel],
    storage: ProfileStorage,
) -> tuple[list[str], str, list[EmployeeProfile]]:
    """Parse a help query with Gemini and find matching profiles."""
    topics: list[str] = []
    summary = query

    if model is not None:
        try:
            help_result = extract_help_topics(query, model)
            topics = help_result.topics
            if help_result.summary:
                summary = help_result.summary
        except ExtractionError as exc:
            logger.warning("Help topic extraction failed: %s", exc.message)

    if not topics:
        topics = [query]

    profiles = storage.search_by_topics(topics)
    return topics, summary, profiles


def get_match_reasons(
    context: str,
    profiles: list[EmployeeProfile],
    model: Optional[GenerativeModel],
) -> dict[str, str]:
    """Generate Gemini match reasons, falling back to empty dict."""
    if model is None or not profiles:
        return {}
    return generate_match_reasons(context, profiles, model)


def process_message_for_learning(
    text: str,
    model: Optional[GenerativeModel],
    storage: ProfileStorage,
    threshold: float,
    logger: logging.Logger,
) -> None:
    """Extract employee info from message text and upsert into MongoDB."""
    text = text.strip()
    if not text:
        return
    if model is None:
        print("   [Learning Service] Skipping profile learning: Vertex AI unavailable")
        logger.warning("Skipping extraction — Vertex AI unavailable")
        return

    try:
        print(f"   [Learning Service] Sending extraction request to Vertex AI Gemini model...")
        result = extract_employee_info(text, model)
        print(f"   [Learning Service] Received Vertex AI response (Confidence: {result.confidence:.2f}, Person: {result.person})")
        if result.confidence < threshold:
            print(f"   [Learning Service] Acknowledged: Skipped low-confidence extraction ({result.confidence:.2f} < threshold {threshold})")
            logger.info("Skipped low-confidence extraction (%.2f)", result.confidence)
            return
        if result.person.lower() == "unknown":
            print(f"   [Learning Service] Acknowledged: Skipped extraction with unknown person")
            logger.info("Skipped extraction with unknown person")
            return

        print(f"   [Learning Service] Sending upsert request to MongoDB for profile: {result.person}...")
        storage.upsert_from_extraction(result, text)
        print(f"   [Learning Service] SUCCESS: Upserted profile update for {result.person} into MongoDB")
        logger.info(
            "Learned profile update for %s (confidence=%.2f, skills=%s)",
            result.person,
            result.confidence,
            result.skills,
        )
    except ExtractionError as exc:
        print(f"   [Learning Service] FAILED: Profile extraction failed: {exc.message}")
        logger.warning("Extraction failed: %s", exc.message)
    except Exception as exc:
        print(f"   [Learning Service] FAILED: Unexpected error in profile learning: {exc}")
        logger.warning("Unexpected error during learning: %s", exc)


def _ingest_channel_memory(
    memory_service: Optional[ChannelMemoryService],
    *,
    channel_id: str,
    message: dict[str, Any],
) -> None:
    """Record a channel message into the memory layer when enabled."""
    if memory_service is None:
        print("   [Memory Service] Skipping memory ingestion: Service not enabled")
        return
    try:
        print(f"   [Memory Service] Sending request to ingest message into MongoDB buffer (Channel: {channel_id})...")
        memory_service.ingest_message(channel_id, message)
        print(f"   [Memory Service] SUCCESS: Acknowledged message ingestion for Channel: {channel_id}")

        # Step 1 of incremental pipeline: O(1) Redis timestamp tracking
        ts = float(message.get("ts") or 0)
        if ts > 0:
            memory_service.track_arrival(channel_id, ts)
    except Exception as exc:
        print(f"   [Memory Service] FAILED: Channel memory ingestion failed for Channel: {channel_id}. Error: {exc}")
        logger.warning("Channel memory ingestion failed for %s: %s", channel_id, exc)


def register_handlers(
    app: App,
    model: Optional[GenerativeModel],
    storage: ProfileStorage,
    memory_service: Optional[ChannelMemoryService] = None,
    availability_storage: Optional[AvailabilityStorage] = None,
) -> None:
    """
    Register Slack event and slash-command handlers on a Bolt app.

    Args:
        app: Slack Bolt application instance.
        model: Vertex AI GenerativeModel (None if unavailable).
        storage: MongoDB profile storage.
        availability_storage: Optional AvailabilityStorage instance.
    """
    settings = get_settings()
    threshold = settings.extraction_confidence_threshold
    avail_store = availability_storage or AvailabilityStorage(settings)

    @app.middleware
    def log_incoming_payload(body, next):
        """Log every Slack payload for debugging event delivery."""
        event = body.get("event", {})
        if event:
            print(f"\n[Slack HTTP Request] Received event: type={event.get('type')}, subtype={event.get('subtype')}, channel={event.get('channel')}")
            logger.info(
                "Slack event: type=%s subtype=%s channel=%s",
                event.get("type"),
                event.get("subtype"),
                event.get("channel"),
            )
        elif body.get("command"):
            print(f"\n[Slack HTTP Request] Received command: {body.get('command')} with text: \"{body.get('text')}\"")
            logger.info("Slack command: %s", body.get("command"))
        return next()

    @app.event("message")
    def handle_message(event: dict[str, Any], logger: logging.Logger) -> None:
        """Extract employee info from channel messages and upsert profiles."""
        logger.info(
            "message handler: subtype=%s bot_id=%s text_len=%s",
            event.get("subtype"),
            event.get("bot_id"),
            len(event.get("text", "")),
        )
        if not is_processable_message(event):
            print(f"   [Slack Event] Acknowledged: Message event skipped (bot_id={event.get('bot_id')}, subtype={event.get('subtype')})")
            logger.info("message skipped (bot/subtype/empty)")
            return

        print(f"   [Slack Event] Acknowledged: Processing message from User @{event.get('user') or 'unknown'} in Channel {event.get('channel')}")
        print(f"   Text: \"{event.get('text')}\"")

        _ingest_channel_memory(
            memory_service,
            channel_id=event.get("channel", "unknown"),
            message={
                "id": event.get("client_msg_id") or event.get("ts") or event.get("event_ts"),
                "ts": float(event.get("ts") or event.get("event_ts") or 0),
                "user_name": event.get("user") or event.get("username") or "unknown",
                "text": event.get("text", ""),
            },
        )

        process_message_for_learning(
            event["text"], model, storage, threshold, logger
        )

    @app.event("app_mention")
    def handle_app_mention(event: dict[str, Any], logger: logging.Logger, client) -> None:
        """Learn from mentions, or answer 'tell me about X' queries."""
        text = event.get("text", "")
        about_query = extract_about_query(text)
        print(f"   [Slack Event] Acknowledged: Processing app_mention from User @{event.get('user') or 'unknown'} in Channel {event.get('channel')}")
        print(f"   Text: \"{text}\"")

        if about_query and re.search(
            r"(?i)(tell me about|who is|^about\s)", strip_bot_mention(text)
        ):
            print(f"   [Slack Event] Route: /about query for user description \"{about_query}\"")
            reply_with_about(
                about_query,
                storage,
                model,
                client,
                event["channel"],
                event["user"],
            )
            return

        print("   [Slack Event] Route: Mentions-based learning fallback")
        _ingest_channel_memory(
            memory_service,
            channel_id=event.get("channel", "unknown"),
            message={
                "id": event.get("client_msg_id") or event.get("ts") or event.get("event_ts"),
                "ts": float(event.get("ts") or event.get("event_ts") or 0),
                "user_name": event.get("user") or event.get("username") or "unknown",
                "text": strip_bot_mention(text),
            },
        )

        process_message_for_learning(
            strip_bot_mention(text), model, storage, threshold, logger
        )

    @app.command("/intro")
    def handle_intro(ack, command: dict[str, Any], client, respond) -> None:
        """
        Manually register expertise via slash command.

        Works without Event Subscriptions — use for hackathon demo if message
        events are not configured yet.
        """
        ack()
        text = command.get("text", "").strip()
        print(f"   [Slack Command] Acknowledged: Processing /intro from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
        print(f"   Intro Text: \"{text}\"")

        if not text:
            print("   [Slack Command] Response: Missing intro text, sending usage instructions.")
            _send_command_response(
                respond,
                text=(
                    "Usage: `/intro <your intro>`\n"
                    "Example: `/intro I'm Aditya, Senior Backend Engineer, expert in Kubernetes`"
                ),
            )
            return

        log = logging.getLogger(__name__)
        _ingest_channel_memory(
            memory_service,
            channel_id=command.get("channel_id", "unknown"),
            message={
                "id": command.get("client_msg_id") or command.get("trigger_id") or command.get("ts") or command.get("event_ts"),
                "ts": float(command.get("event_ts") or command.get("ts") or 0),
                "user_name": command.get("user_name") or command.get("user_id") or "unknown",
                "text": text,
            },
        )
        process_message_for_learning(text, model, storage, threshold, log)

        _send_command_response(
            respond,
            text="Got it — Org Brain learned from your intro. Try `/who-knows <skill>` now.",
        )

    @app.command("/who-knows")
    def handle_who_knows(ack, command: dict[str, Any], client, respond) -> None:
        """Search stored profiles for employees with a given skill."""
        ack()
        skill = command.get("text", "").strip()
        print(f"   [Slack Command] Acknowledged: Processing /who-knows from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
        print(f"   Query Skill: \"{skill}\"")

        if not skill:
            print("   [Slack Command] Response: Missing skill parameter, sending usage instructions.")
            _send_command_response(
                respond,
                text="Usage: `/who-knows <skill>` — e.g. `/who-knows kubernetes`",
            )
            return

        try:
            print(f"   [Slack Command] Route: Searching MongoDB for skill: \"{skill}\"...")
            profiles = storage.search_by_skill(skill)
            print(f"   [Slack Command] Route: Found {len(profiles)} matching profiles. Generating match reasons with Gemini...")
            reasons = get_match_reasons(skill, profiles, model)
            blocks = build_who_knows_blocks(skill, profiles, reasons)
            _send_command_response(
                respond,
                blocks=blocks,
                text=f"Experts for {skill}",
            )
            print("   [Slack Command] SUCCESS: Posted search results to user.")
        except Exception as exc:
            print(f"   [Slack Command] FAILED: /who-knows command failed: {exc}")

    @app.command("/help")
    def handle_help(ack, command: dict[str, Any], client, respond) -> None:
        """
        Find experts for a natural-language help request.

        Example: /help I'm facing an issue in Kubernetes, who can I ask?
        """
        ack()
        query = command.get("text", "").strip()
        print(f"   [Slack Command] Acknowledged: Processing /help from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
        print(f"   Query text: \"{query}\"")

        if not query:
            print("   [Slack Command] Response: Missing query parameter, sending usage instructions.")
            _send_command_response(
                respond,
                text=(
                    "Usage: `/help <describe your issue>`\n"
                    "Example: `/help I'm stuck on a Kubernetes deployment, who can help?`"
                ),
            )
            return

        try:
            print(f"   [Slack Command] Route: Resolving help profiles for query with Gemini/MongoDB...")
            topics, summary, profiles = resolve_help_profiles(query, model, storage)
            context = summary or ", ".join(topics)
            print(f"   [Slack Command] Route: Resolved topics={topics}, summary=\"{summary}\". Found {len(profiles)} profiles. Generating reasons...")
            reasons = get_match_reasons(context, profiles, model)
            blocks = build_help_blocks(topics, summary, profiles, reasons)
            _send_command_response(
                respond,
                blocks=blocks,
                text=f"Help with {', '.join(topics)}",
            )
            print("   [Slack Command] SUCCESS: Posted /help response to user.")
        except Exception as exc:
            print(f"   [Slack Command] FAILED: /help command failed: {exc}")

    @app.command("/about")
    def handle_about(ack, command: dict[str, Any], client, respond) -> None:
        """
        Get a profile card with description and quirky tagline.

        Usage: /about Aditya  OR  /about @Aditya
        """
        ack()
        query = command.get("text", "").strip()
        print(f"   [Slack Command] Acknowledged: Processing /about from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
        print(f"   Target Query: \"{query}\"")

        if not query:
            print("   [Slack Command] Response: Missing name query, sending usage instructions.")
            _send_command_response(
                respond,
                text=(
                    "Usage: `/about <name>` or `/about @person`\n"
                    "Example: `/about Aditya Karanwal`"
                ),
            )
            return

        try:
            print(f"   [Slack Command] Route: Resolving profile from query: \"{query}\"...")
            profile = resolve_profile_from_query(query, storage, client)
            if not profile:
                print(f"   [Slack Command] Response: No profile found for query: \"{query}\"")
                _send_command_response(
                    respond,
                    text=(
                        f"No profile found for *{query.strip()}* yet.\n"
                        "Try `/intro` or `@OrgBrain I'm Name, Role, expert in X` to add them."
                    ),
                )
                return

            print(f"   [Slack Command] Route: Found profile for {profile.person}. Generating about card information...")
            about = get_profile_about(profile, model)
            blocks = build_about_blocks(profile, about)
            _send_command_response(
                respond,
                blocks=blocks,
                text=f"About {profile.person}",
            )
            print("   [Slack Command] SUCCESS: Posted about card to user.")
        except Exception as exc:
            print(f"   [Slack Command] FAILED: /about command failed: {exc}")

    @app.command("/summarize")
    def handle_summarize(ack, command: dict[str, Any], client, respond) -> None:
        """
        Summarize channel activity in a Block Kit modal.

        Usage: /summarize #channel last 2 days  OR  /summarize last 24 hours
        """
        ack()
        print(f"   [Slack Command] Acknowledged: Processing /summarize from User @{command.get('user_name')} in Channel {command.get('channel_id')}")
        print(f"   Arguments: \"{command.get('text')}\"")
        try:
            summarize_slash_command(command, client, respond, model, memory_service)
            print("   [Slack Command] SUCCESS: Channel summary modal opened.")
        except Exception as exc:
            print(f"   [Slack Command] FAILED: /summarize command failed: {exc}")

    @app.command("/recall")
    def handle_recall(ack, command: dict[str, Any], client, respond) -> None:
        """
        Search public channel memories using semantic search.

        Usage: /recall <query>
        """
        ack()
        recall_slash_command(command, client, respond, memory_service)

    @app.event("app_home_opened")
    def update_home_tab(client, event, logger) -> None:
        """Publish the visual expertise and project directory to the App Home tab."""
        user_id = event["user"]
        print(f"   [Slack Event] Acknowledged: app_home_opened for User @{user_id}")
        try:
            # Fetch profiles
            profiles = storage.list_profiles(limit=50)
            
            # Count unique projects and skills
            projects_set = set()
            skills_set = set()
            for p in profiles:
                if p.projects:
                    projects_set.update(p.projects)
                if p.skills:
                    skills_set.update(p.skills)

            # Build blocks
            blocks: list[dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Org Brain - Workspace Intelligence", "emoji": False},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Welcome back, <@{user_id}>! *Org Brain* maps team capabilities and extracts project memories automatically from public Slack conversations.",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Status:* Connected  |  *Active Experts:* {len(profiles)}  |  *Indexed Projects:* {len(projects_set)}  |  *Skills:* {len(skills_set)}"
                        }
                    ]
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Quick Actions*",
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Register My Skills", "emoji": False},
                            "action_id": "action_intro_dialog",
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Update Availability", "emoji": False},
                            "action_id": "action_calendar_modal",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View Team Schedule", "emoji": False},
                            "action_id": "action_view_team_calendar",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Find Experts", "emoji": False},
                            "action_id": "action_find_experts",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Summarize Channel", "emoji": False},
                            "action_id": "action_summarize_dialog",
                        }
                    ]
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Employee Capabilities Directory*",
                    },
                },
            ]

            if not profiles:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "_No profiles learned yet. Chat naturally in channels or introduce yourself to start learning!_"
                    }
                })
            else:
                for profile in profiles:
                    profile_text = f"*{profile.person}*"
                    if profile.role:
                        profile_text += f" - _{profile.role}_"
                    if profile.team:
                        profile_text += f" ({profile.team})"
                    
                    skills_str = ", ".join(profile.skills[:5]) if profile.skills else "None"
                    projects_str = ", ".join(profile.projects[:3]) if profile.projects else "None"
                    confidence_val = int((profile.confidence or 0.0) * 100)
                    
                    details_text = (
                        f"{profile_text}\n"
                        f"*Skills:* {skills_str}\n"
                        f"*Projects:* {projects_str}\n"
                        f"*Confidence:* {confidence_val}%"
                    )
                    
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": details_text
                        },
                        "accessory": {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View About Card", "emoji": False},
                            "value": profile.person,
                            "action_id": "home_request_about"
                        }
                    })
                    blocks.append({"type": "divider"})

            # Context
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "_Tip: Run `/intro` anywhere to update your skills, or invite the bot using `/invite @OrgBrain` to monitor channels._"
                    }
                ]
            })

            # Publish the view
            client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": blocks
                }
            )
            print(f"   [Slack Event] SUCCESS: Published App Home to User @{user_id}")
        except Exception as exc:
            logger.error("Failed to publish app home view: %s", exc)
            print(f"   [Slack Event] FAILED: app_home_opened handler failed: {exc}")

    @app.action("home_request_about")
    def handle_home_about(ack, body, client) -> None:
        """Trigger about card on request from Home Tab."""
        ack()
        person = body["actions"][0]["value"]
        user_id = body["user"]["id"]
        print(f"   [Slack Action] Acknowledged: home_request_about for {person}")
        try:
            profile = storage.find_profile_by_name(person)
            if profile:
                about = get_profile_about(profile, model)
                blocks = build_about_blocks(profile, about)
                client.chat_postMessage(
                    channel=user_id,
                    blocks=blocks,
                    text=f"About {profile.person}",
                )
        except Exception as exc:
            print(f"   [Slack Action] FAILED: home_request_about card failed: {exc}")

    @app.action("action_intro_dialog")
    def handle_intro_dialog(ack, body, client) -> None:
        ack()
        user_id = body["user"]["id"]
        client.chat_postMessage(
            channel=user_id,
            text="To introduce yourself and update your skills, run this command in any chat:\n`/intro I'm [Name], [Role], expert in [skill1], [skill2]`"
        )

    @app.action("action_find_experts")
    def handle_find_experts_dialog(ack, body, client) -> None:
        ack()
        user_id = body["user"]["id"]
        client.chat_postMessage(
            channel=user_id,
            text="To find team experts, run this command in any chat:\n`/who-knows [skill]` — e.g. `/who-knows Kubernetes`"
        )

    @app.action("action_summarize_dialog")
    def handle_summarize_dialog(ack, body, client) -> None:
        ack()
        user_id = body["user"]["id"]
        client.chat_postMessage(
            channel=user_id,
            text="To summarize any channel's activity, run this command in any chat:\n`/summarize [#channel-name] last [X] days`"
        )

    @app.action("action_calendar_modal")
    def handle_calendar_modal_action(ack, body, client) -> None:
        ack()
        user_id = body["user"]["id"]
        trigger_id = body["trigger_id"]
        print(f"   [Slack Action] Acknowledged: action_calendar_modal for User @{user_id}")
        try:
            modal = build_calendar_modal(channel_id=user_id)
            client.views_open(trigger_id=trigger_id, view=modal)
            print(f"   [Slack Action] SUCCESS: Opened calendar modal for User @{user_id}")
        except Exception as exc:
            logger.error("Failed to open calendar modal from Action: %s", exc)
            print(f"   [Slack Action] FAILED: calendar modal action failed: {exc}")

    @app.action("action_view_team_calendar")
    def handle_view_team_calendar_action(ack, body, client) -> None:
        ack()
        user_id = body["user"]["id"]
        trigger_id = body["trigger_id"]
        print(f"   [Slack Action] Acknowledged: action_view_team_calendar for User @{user_id}")
        try:
            from datetime import datetime as _dt
            today = _dt.now().strftime("%Y-%m-%d")
            entries = avail_store.get_upcoming_team_entries(date_start=today)
            modal = build_team_calendar_modal_view(entries)
            client.views_open(trigger_id=trigger_id, view=modal)
            print(f"   [Slack Action] SUCCESS: Opened team calendar modal for User @{user_id}")
        except Exception as exc:
            logger.error("Failed to open team calendar modal from Action: %s", exc)
            print(f"   [Slack Action] FAILED: team calendar modal action failed: {exc}")

    @app.view("calendar_status_modal")
    def handle_calendar_status_modal_submission(ack, body, client, view) -> None:
        # Extract view state values
        state_values = view["state"]["values"]
        status = state_values["status_block"]["status_select"]["selected_option"]["value"]
        
        # Start date is required
        date_start = state_values["date_start_block"]["date_start_picker"]["selected_date"]
        
        # End date is optional, defaults to start date
        date_end_el = state_values["date_end_block"]["date_end_picker"].get("selected_date")
        date_end = date_end_el if date_end_el else date_start
        
        # Start time is optional, defaults to 00:00
        time_start_el = state_values["time_start_block"]["time_start_picker"].get("selected_time")
        time_start = time_start_el if time_start_el else "00:00"
        
        # End time is optional, defaults to 23:59
        time_end_el = state_values["time_end_block"]["time_end_picker"].get("selected_time")
        time_end = time_end_el if time_end_el else "23:59"
        
        # Reason is optional
        reason_el = state_values["reason_block"]["reason_input"].get("value")
        reason = reason_el.strip() if reason_el else None

        user_id = body["user"]["id"]
        user_name = body["user"]["username"]
        team_id = body["team"]["id"]
        channel_id = view.get("private_metadata") or user_id

        # Validate
        errors = {}
        today = datetime.now().date().isoformat()
        if date_start < today:
            errors["date_start_block"] = "Cannot set availability for past dates"
        if date_start > date_end:
            errors["date_end_block"] = "Start date must be before or equal to end date"
        if date_start == date_end and time_start >= time_end:
            errors["time_end_block"] = "Start time must be before end time"
        if reason and len(reason) > 100:
            errors["reason_block"] = "Reason too long (max 100 characters)"

        if errors:
            ack(response_action="errors", errors=errors)
            return

        # Acknowledge the submission
        ack()

        # Fetch user info
        display_name = user_name
        user_email = ""
        timezone = "Asia/Kolkata"
        try:
            user_resp = client.users_info(user=user_id)
            user_obj = user_resp.get("user", {})
            profile = user_obj.get("profile", {})
            display_name = (
                profile.get("display_name")
                or user_obj.get("real_name")
                or user_name
            )
            user_email = profile.get("email", "")
            timezone = user_obj.get("tz", "Asia/Kolkata")
        except Exception as exc:
            logger.warning("Could not fetch user info for %s: %s", user_id, exc)

        entry = AvailabilityEntry(
            user_id=user_id,
            user_name=user_name,
            user_display_name=display_name,
            user_email=user_email,
            team_id=team_id,
            date_start=date_start,
            date_end=date_end,
            time_start=time_start,
            time_end=time_end,
            status=AvailabilityStatus(status),
            reason=reason,
            channel_id=channel_id,
            timezone=timezone,
        )

        try:
            avail_store.save_entry(entry)
            blocks = build_status_confirmation_blocks(entry)
            client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="Availability updated via Interactive Modal",
                blocks=blocks,
            )
            print(f"   [Calendar] SUCCESS: Saved modal status={entry.status} for {user_id}")
        except Exception as exc:
            logger.error("Calendar modal save failed: %s", exc)
            client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=f"❌ Failed to save availability: {exc}"
            )

    # ── Calendar: /calendar command ─────────────────────────────────────

    @app.command("/calendar")
    def handle_calendar(ack, command: dict[str, Any], client, respond) -> None:
        """Slack-native availability tracker — set status, query, view, clear."""
        ack()
        text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        team_id = command.get("team_id", "")
        print(
            f"   [Slack Command] Acknowledged: Processing /calendar from "
            f"User @{command.get('user_name')} — text: \"{text}\""
        )

        if not text or text.lower() == "ui":
            trigger_id = command.get("trigger_id")
            if trigger_id:
                try:
                    modal = build_calendar_modal(channel_id=channel_id)
                    client.views_open(trigger_id=trigger_id, view=modal)
                    print(f"   [Calendar] SUCCESS: Opened interactive calendar modal for {user_id}")
                    return
                except Exception as exc:
                    logger.error("Failed to open calendar modal: %s", exc)
                    _send_command_response(respond, text=f"❌ Failed to open interactive modal: {exc}")
                    return

        parsed = parse_calendar_command(text)

        # ── help ──
        if parsed.action == "help":
            _send_command_response(respond, text="Calendar usage", blocks=build_usage_blocks())
            return

        # ── status (set availability) ──
        if parsed.action == "status":
            errors = validate_availability(parsed)
            if errors:
                _send_command_response(
                    respond,
                    text=f"❌ {'; '.join(errors)}",
                )
                return

            # Fetch user info from Slack for name/email/timezone
            user_name = command.get("user_name", "")
            display_name = user_name
            user_email = ""
            timezone = "Asia/Kolkata"
            try:
                user_resp = client.users_info(user=user_id)
                user_obj = user_resp.get("user", {})
                profile = user_obj.get("profile", {})
                display_name = (
                    profile.get("display_name")
                    or user_obj.get("real_name")
                    or user_name
                )
                user_email = profile.get("email", "")
                timezone = user_obj.get("tz", "Asia/Kolkata")
            except Exception as exc:
                logger.warning("Could not fetch user info for %s: %s", user_id, exc)

            entry = AvailabilityEntry(
                user_id=user_id,
                user_name=user_name,
                user_display_name=display_name,
                user_email=user_email,
                team_id=team_id,
                date_start=parsed.date_start or "",
                date_end=parsed.date_end or parsed.date_start or "",
                time_start=parsed.time_start or "00:00",
                time_end=parsed.time_end or "23:59",
                status=AvailabilityStatus(parsed.status or "busy"),
                reason=parsed.reason,
                channel_id=channel_id,
                timezone=timezone,
            )

            try:
                avail_store.save_entry(entry)
                blocks = build_status_confirmation_blocks(entry)
                _send_command_response(respond, text="Availability updated", blocks=blocks)
                print(f"   [Calendar] SUCCESS: Saved status={entry.status} for {user_id}")
            except Exception as exc:
                logger.error("Calendar save failed: %s", exc)
                _send_command_response(respond, text=f"❌ Failed to save: {exc}")
            return

        # ── who-is-free ──
        if parsed.action == "who-is-free":
            if parsed.errors:
                _send_command_response(respond, text=f"❌ {'; '.join(parsed.errors)}")
                return

            try:
                date_query = parsed.date_start or ""
                entries = avail_store.get_entries_for_date(date_query)

                free = [e for e in entries if e.status == AvailabilityStatus.free]
                busy = [e for e in entries if e.status == AvailabilityStatus.busy]
                leave = [e for e in entries if e.status == AvailabilityStatus.leave]

                blocks = build_who_is_free_blocks(date_query, free, busy, leave)
                _send_command_response(
                    respond,
                    text=f"Team availability for {date_query}",
                    blocks=blocks,
                )
                print(f"   [Calendar] SUCCESS: who-is-free {date_query} → {len(entries)} entries")
            except Exception as exc:
                logger.error("Calendar query failed: %s", exc)
                _send_command_response(respond, text=f"❌ Query failed: {exc}")
            return

        # ── my-calendar ──
        if parsed.action == "my-calendar":
            try:
                from datetime import datetime as _dt
                today = _dt.now().strftime("%Y-%m-%d")
                entries = avail_store.get_entries_for_user(user_id, date_start=today)
                user_display = command.get("user_name", user_id)
                blocks = build_my_calendar_blocks(entries, user_display)
                _send_command_response(
                    respond,
                    text="Your calendar",
                    blocks=blocks,
                )
                print(f"   [Calendar] SUCCESS: my-calendar for {user_id} → {len(entries)} entries")
            except Exception as exc:
                logger.error("Calendar my-calendar failed: %s", exc)
                _send_command_response(respond, text=f"❌ Failed to load calendar: {exc}")
            return

        # ── team-calendar ──
        if parsed.action == "team-calendar":
            try:
                from datetime import datetime as _dt
                today = _dt.now().strftime("%Y-%m-%d")
                entries = avail_store.get_upcoming_team_entries(date_start=today)

                # Check if UI requested
                if "ui" in text.lower():
                    trigger_id = command.get("trigger_id")
                    if trigger_id:
                        modal = build_team_calendar_modal_view(entries)
                        client.views_open(trigger_id=trigger_id, view=modal)
                        print(f"   [Calendar] SUCCESS: Opened interactive team calendar modal for {user_id}")
                        return

                blocks = build_team_calendar_blocks(entries)
                _send_command_response(
                    respond,
                    text="Team Schedule Overview",
                    blocks=blocks,
                )
                print(f"   [Calendar] SUCCESS: team-calendar → {len(entries)} entries")
            except Exception as exc:
                logger.error("Calendar team-calendar failed: %s", exc)
                _send_command_response(respond, text=f"❌ Failed to load team calendar: {exc}")
            return

        # ── clear ──
        if parsed.action == "clear":
            if parsed.errors:
                _send_command_response(respond, text=f"❌ {'; '.join(parsed.errors)}")
                return

            try:
                date_str = parsed.date_start or ""
                deleted = avail_store.clear_user_date(user_id, date_str)
                blocks = build_clear_confirmation_blocks(date_str, deleted)
                _send_command_response(respond, text="Calendar cleared", blocks=blocks)
                print(f"   [Calendar] SUCCESS: Cleared {deleted} entries for {user_id} on {date_str}")
            except Exception as exc:
                logger.error("Calendar clear failed: %s", exc)
                _send_command_response(respond, text=f"❌ Failed to clear: {exc}")
            return

        # Fallback
        _send_command_response(respond, text="Calendar usage", blocks=build_usage_blocks())


def create_slack_app(
    model: Optional[GenerativeModel] = None,
    storage: Optional[ProfileStorage] = None,
    memory_service: Optional[ChannelMemoryService] = None,
    availability_storage: Optional[AvailabilityStorage] = None,
) -> App:
    """
    Create and configure a Slack Bolt app with Org Brain handlers.

    Args:
        model: Optional pre-initialized GenerativeModel.
        storage: Optional ProfileStorage instance.
        availability_storage: Optional AvailabilityStorage instance.

    Returns:
        Configured Slack Bolt App.

    Raises:
        RuntimeError: If required Slack tokens are missing.
    """
    settings = get_settings()
    if not settings.slack_configured():
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required")

    app = App(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )
    profile_storage = storage or ProfileStorage(settings)
    avail = availability_storage or AvailabilityStorage(settings)
    register_handlers(app, model, profile_storage, memory_service, avail)
    return app


def main() -> None:
    """Start the Slack app in Socket Mode (local development)."""
    settings = get_settings()
    configure_logging(settings)

    if not settings.slack_socket_mode_ready():
        raise RuntimeError(
            "Socket Mode requires SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, and SLACK_APP_TOKEN"
        )

    model: Optional[GenerativeModel] = None
    try:
        model = init_vertex_ai(settings)
    except RuntimeError as exc:
        logger.error("Vertex AI unavailable: %s", exc)

    storage = ProfileStorage(settings)
    memory_service = create_channel_memory_service(settings, model=model)
    avail_storage = AvailabilityStorage(settings)
    app = create_slack_app(
        model=model,
        storage=storage,
        memory_service=memory_service,
        availability_storage=avail_storage,
    )
    handler = SocketModeHandler(app, settings.slack_app_token)

    logger.info("Org Brain Slack app running (Socket Mode)")
    logger.info(
        "Ensure Event Subscriptions are ON with message.channels + app_mention "
        "(see docs/SLACK_APP_SETUP.md)"
    )
    handler.start()


if __name__ == "__main__":
    main()
