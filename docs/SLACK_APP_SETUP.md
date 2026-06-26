# Slack App Setup for Org Brain

Follow these steps to create and configure your Slack app for the hackathon demo.

## 1. Create the app

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From scratch**
3. Name: `Org Brain`
4. Pick your hackathon workspace

## 2. Enable Socket Mode (local development)

1. **Settings** → **Socket Mode** → Enable
2. **Basic Information** → **App-Level Tokens** → Generate token
3. Scope: `connections:write`
4. Copy token → `SLACK_APP_TOKEN` in `backend/.env`

## 3. Bot token scopes

**OAuth & Permissions** → **Bot Token Scopes**, add:

| Scope | Why |
|-------|-----|
| `channels:history` | Read public channel messages |
| `groups:history` | Read private channel messages |
| `im:history` | Read DMs |
| `chat:write` | Post responses |
| `commands` | Slash commands |
| `users:read` | Resolve user names |
| `app_mentions:read` | Receive @OrgBrain mentions |

Click **Install to Workspace**, then copy **Bot User OAuth Token** → `SLACK_BOT_TOKEN`

## 4. Signing secret

**Basic Information** → **App Credentials** → copy **Signing Secret** → `SLACK_SIGNING_SECRET`

## 5. Event subscriptions (REQUIRED — even with Socket Mode)

`/who-knows` works without this, but **auto-learning from messages will not**.

1. **Event Subscriptions** → **Enable Events** → ON
2. Request URL: leave blank or use `https://example.com` (ignored in Socket Mode)
3. **Subscribe to bot events**, add all three:
   - `message.channels`
   - `message.groups`
   - `message.im`
   - `app_mention`
4. **Save Changes**
5. **Reinstall app** to workspace (OAuth & Permissions → Reinstall to Workspace)

## 6. Slash command

**Slash Commands** → **Create New Command**:

| Field | Value |
|-------|-------|
| Command | `/who-knows` |
| Short Description | Find who has a skill in your org |
| Usage Hint | `[skill]` e.g. `kubernetes` |

Create a second command for manual learning (works without Event Subscriptions):

| Field | Value |
|-------|-------|
| Command | `/intro` |
| Short Description | Register your skills with Org Brain |
| Usage Hint | `I'm Name, Role, expert in skill1, skill2` |

Create a third command for natural-language help:

| Field | Value |
|-------|-------|
| Command | `/help` |
| Short Description | Find experts for your issue |
| Usage Hint | `I'm stuck on Kubernetes, who can help?` |

Create a fourth command for profile cards:

| Field | Value |
|-------|-------|
| Command | `/about` |
| Short Description | Get a profile card for a teammate |
| Usage Hint | `Aditya Karanwal` or `@person` |

Create a fifth command for channel summarization:

| Field | Value |
|-------|-------|
| Command | `/summarize` |
| Short Description | Summarize a channel's activity |
| Usage Hint | `#channel last 2 days` |

Add scope if missing:

| Scope | Why |
|-------|-----|
| `channels:read` | List public channels for name lookup |

## 7. Invite the bot to channels

In Slack, run `/invite @OrgBrain` in your demo channel (e.g. `#hackathon-`).

## 8. Configure backend/.env

```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=...
```

## 9. Run locally

```bash
# Start data stores
docker compose up -d

# Socket Mode (no public URL needed)
python -m backend.slack_app
```

## 10. Verify

**Quick demo (no Event Subscriptions needed):**
```
/intro I'm Aditya Karanwal, Senior Backend Engineer, expert in Kubernetes
/who-knows kubernetes
/summarize #hackathon- last 2 days
```

**Auto-learning (requires Event Subscriptions from step 5):**
1. Post in a channel: *"I'm Priya, Senior Backend Engineer, expert in Kubernetes"*
2. Terminal should log: `Slack event: type=message ...`
3. Run `/who-knows kubernetes` — you should see Priya in results

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `/who-knows` works but no profiles learned | **Enable Event Subscriptions** + add `message.channels` and `app_mention`, then reinstall app |
| Bot not seeing messages | Invite bot to channel; check scopes |
| `/who-knows` not found | Reinstall app after adding slash command |
| Socket Mode won't connect | Verify `SLACK_APP_TOKEN` has `connections:write` |
| No profiles returned | Check terminal for `Learned profile update` logs after posting |
| `/summarize` empty | Invite bot to channel; ensure messages exist in timeframe |
| Summary modal stuck loading | Check terminal for Gemini errors; verify GCP credentials |


.\.venv\Scripts\Activate.ps1
python -m backend.slack_app