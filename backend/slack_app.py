"""Slack Bolt app — listens to messages, learns profiles, answers /who-knows."""

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
from backend.summarizer import (
    format_block_kit,
    format_error_modal,
    format_loading_modal,
    summarize_channel,
)

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
        logger.warning("Skipping extraction — Vertex AI unavailable")
        return

    try:
        result = extract_employee_info(text, model)
        if result.confidence < threshold:
            logger.info("Skipped low-confidence extraction (%.2f)", result.confidence)
            return
        if result.person.lower() == "unknown":
            logger.info("Skipped extraction with unknown person")
            return

        storage.upsert_from_extraction(result, text)
        logger.info(
            "Learned profile update for %s (confidence=%.2f, skills=%s)",
            result.person,
            result.confidence,
            result.skills,
        )
    except ExtractionError as exc:
        logger.warning("Extraction failed: %s", exc.message)


def register_handlers(
    app: App,
    model: Optional[GenerativeModel],
    storage: ProfileStorage,
) -> None:
    """
    Register Slack event and slash-command handlers on a Bolt app.

    Args:
        app: Slack Bolt application instance.
        model: Vertex AI GenerativeModel (None if unavailable).
        storage: MongoDB profile storage.
    """
    settings = get_settings()
    threshold = settings.extraction_confidence_threshold

    @app.middleware
    def log_incoming_payload(body, next):
        """Log every Slack payload for debugging event delivery."""
        event = body.get("event", {})
        if event:
            logger.info(
                "Slack event: type=%s subtype=%s channel=%s",
                event.get("type"),
                event.get("subtype"),
                event.get("channel"),
            )
        elif body.get("command"):
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
            logger.info("message skipped (bot/subtype/empty)")
            return

        process_message_for_learning(
            event["text"], model, storage, threshold, logger
        )

    @app.event("app_mention")
    def handle_app_mention(event: dict[str, Any], logger: logging.Logger, client) -> None:
        """Learn from mentions, or answer 'tell me about X' queries."""
        text = event.get("text", "")
        about_query = extract_about_query(text)

        if about_query and re.search(
            r"(?i)(tell me about|who is|^about\s)", strip_bot_mention(text)
        ):
            reply_with_about(
                about_query,
                storage,
                model,
                client,
                event["channel"],
                event["user"],
            )
            return

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

        if not text:
            _send_command_response(
                respond,
                text=(
                    "Usage: `/intro <your intro>`\n"
                    "Example: `/intro I'm Aditya, Senior Backend Engineer, expert in Kubernetes`"
                ),
            )
            return

        log = logging.getLogger(__name__)
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

        if not skill:
            _send_command_response(
                respond,
                text="Usage: `/who-knows <skill>` — e.g. `/who-knows kubernetes`",
            )
            return

        profiles = storage.search_by_skill(skill)
        reasons = get_match_reasons(skill, profiles, model)
        blocks = build_who_knows_blocks(skill, profiles, reasons)
        _send_command_response(
            respond,
            blocks=blocks,
            text=f"Experts for {skill}",
        )

    @app.command("/help")
    def handle_help(ack, command: dict[str, Any], client, respond) -> None:
        """
        Find experts for a natural-language help request.

        Example: /help I'm facing an issue in Kubernetes, who can I ask?
        """
        ack()
        query = command.get("text", "").strip()

        if not query:
            _send_command_response(
                respond,
                text=(
                    "Usage: `/help <describe your issue>`\n"
                    "Example: `/help I'm stuck on a Kubernetes deployment, who can help?`"
                ),
            )
            return

        topics, summary, profiles = resolve_help_profiles(query, model, storage)
        context = summary or ", ".join(topics)
        reasons = get_match_reasons(context, profiles, model)
        blocks = build_help_blocks(topics, summary, profiles, reasons)
        _send_command_response(
            respond,
            blocks=blocks,
            text=f"Help with {', '.join(topics)}",
        )

    @app.command("/about")
    def handle_about(ack, command: dict[str, Any], client, respond) -> None:
        """
        Get a profile card with description and quirky tagline.

        Usage: /about Aditya  OR  /about @Aditya
        """
        ack()
        query = command.get("text", "").strip()

        if not query:
            _send_command_response(
                respond,
                text=(
                    "Usage: `/about <name>` or `/about @person`\n"
                    "Example: `/about Aditya Karanwal`"
                ),
            )
            return

        profile = resolve_profile_from_query(query, storage, client)
        if not profile:
            _send_command_response(
                respond,
                text=(
                    f"No profile found for *{query.strip()}* yet.\n"
                    "Try `/intro` or `@OrgBrain I'm Name, Role, expert in X` to add them."
                ),
            )
            return

        about = get_profile_about(profile, model)
        blocks = build_about_blocks(profile, about)
        _send_command_response(
            respond,
            blocks=blocks,
            text=f"About {profile.person}",
        )

    @app.command("/summarize")
    def handle_summarize(ack, command: dict[str, Any], client, respond) -> None:
        """
        Summarize channel activity in a Block Kit modal.

        Usage: /summarize #channel last 2 days  OR  /summarize last 24 hours
        """
        ack()
        summarize_slash_command(command, client, respond, model)


def create_slack_app(
    model: Optional[GenerativeModel] = None,
    storage: Optional[ProfileStorage] = None,
) -> App:
    """
    Create and configure a Slack Bolt app with Org Brain handlers.

    Args:
        model: Optional pre-initialized GenerativeModel.
        storage: Optional ProfileStorage instance.

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
    register_handlers(app, model, profile_storage)
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
    app = create_slack_app(model=model, storage=storage)
    handler = SocketModeHandler(app, settings.slack_app_token)

    logger.info("Org Brain Slack app running (Socket Mode)")
    logger.info(
        "Ensure Event Subscriptions are ON with message.channels + app_mention "
        "(see docs/SLACK_APP_SETUP.md)"
    )
    handler.start()


if __name__ == "__main__":
    main()
