from datetime import datetime
from typing import Any

def build_calendar_modal(channel_id: str = "") -> dict[str, Any]:
    """
    Build the Slack Block Kit modal payload for updating availability.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    return {
        "type": "modal",
        "callback_id": "calendar_status_modal",
        "title": {"type": "plain_text", "text": "Set Availability"},
        "submit": {"type": "plain_text", "text": "Save"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": channel_id,
        "blocks": [
            {
                "type": "input",
                "block_id": "status_block",
                "element": {
                    "type": "static_select",
                    "action_id": "status_select",
                    "placeholder": {"type": "plain_text", "text": "Select a status"},
                    "options": [
                        {
                            "text": {"type": "plain_text", "text": "✅ Free"},
                            "value": "free"
                        },
                        {
                            "text": {"type": "plain_text", "text": "⛔ Busy"},
                            "value": "busy"
                        },
                        {
                            "text": {"type": "plain_text", "text": "🏖️ On Leave"},
                            "value": "leave"
                        }
                    ]
                },
                "label": {"type": "plain_text", "text": "Status"}
            },
            {
                "type": "input",
                "block_id": "date_start_block",
                "element": {
                    "type": "datepicker",
                    "action_id": "date_start_picker",
                    "initial_date": today_str,
                    "placeholder": {"type": "plain_text", "text": "Select start date"}
                },
                "label": {"type": "plain_text", "text": "Start Date"}
            },
            {
                "type": "input",
                "block_id": "date_end_block",
                "optional": True,
                "element": {
                    "type": "datepicker",
                    "action_id": "date_end_picker",
                    "placeholder": {"type": "plain_text", "text": "Select end date (defaults to start date)"}
                },
                "label": {"type": "plain_text", "text": "End Date"}
            },
            {
                "type": "input",
                "block_id": "time_start_block",
                "optional": True,
                "element": {
                    "type": "timepicker",
                    "action_id": "time_start_picker",
                    "initial_time": "00:00",
                    "placeholder": {"type": "plain_text", "text": "Select start time"}
                },
                "label": {"type": "plain_text", "text": "Start Time"}
            },
            {
                "type": "input",
                "block_id": "time_end_block",
                "optional": True,
                "element": {
                    "type": "timepicker",
                    "action_id": "time_end_picker",
                    "initial_time": "23:59",
                    "placeholder": {"type": "plain_text", "text": "Select end time"}
                },
                "label": {"type": "plain_text", "text": "End Time"}
            },
            {
                "type": "input",
                "block_id": "reason_block",
                "optional": True,
                "element": {
                    "type": "plain_text_input",
                    "action_id": "reason_input",
                    "multiline": False,
                    "placeholder": {"type": "plain_text", "text": "e.g. Vacation, Client meeting"}
                },
                "label": {"type": "plain_text", "text": "Reason"}
            }
        ]
    }


def build_team_calendar_modal_view(entries: list[Any]) -> dict[str, Any]:
    """
    Build the Slack Block Kit modal payload for viewing team calendar.
    """
    from backend.calendar_blocks import build_team_calendar_blocks
    blocks = build_team_calendar_blocks(entries, exclude_header=True)
    return {
        "type": "modal",
        "callback_id": "team_calendar_modal",
        "title": {"type": "plain_text", "text": "Team Schedule"},
        "close": {"type": "plain_text", "text": "Close"},
        "blocks": blocks
    }

