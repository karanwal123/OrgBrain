# Slack Marketplace Submission Checklist

Use this when submitting Org Brain before the hackathon deadline.

## Pre-submission requirements

- [ ] App works in a real Slack workspace (demo script passes)
- [ ] Public HTTPS endpoint deployed (Cloud Run — see below)
- [ ] Socket Mode **disabled** for production app
- [ ] Event Subscriptions pointing to `https://YOUR_URL/slack/events`
- [ ] Privacy policy hosted at a public URL (use `docs/PRIVACY_POLICY.md` on GitHub Pages)

## App listing assets

| Asset | Spec |
|-------|------|
| App name | Org Brain |
| Short description | Learns employee expertise from Slack conversations and surfaces who to ask via `/who-knows` |
| Long description | See template below |
| App icon | 512×512 PNG |
| Screenshots | 3 minimum: auto-learning message, `/who-knows` result, profile growth |

### Long description template

```
Org Brain is an organizational memory agent for Slack.

Instead of being a chatbot, it quietly observes workplace conversations and builds
a living map of who knows what — skills, projects, and roles extracted automatically
using Google Vertex AI.

When someone needs help, they run /who-knows <skill> and instantly see relevant
colleagues instead of digging through months of chat history.

Perfect for:
• Faster onboarding
• Cross-team expertise discovery
• Reducing "who should I ask?" friction

Built with Slack Bolt, FastAPI, Vertex AI Gemini, and MongoDB.
```

## Submission steps

1. [api.slack.com/apps](https://api.slack.com/apps) → your app
2. **Manage Distribution** → **Share App with Other Workspaces**
3. Complete **Prepare for Marketplace** checklist
4. Submit for review

## Deploy to Cloud Run (production HTTP mode)

```bash
# Build and deploy
gcloud run deploy org-brain \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=your-project,GOOGLE_CLOUD_REGION=us-central1,GEMINI_MODEL=gemini-2.5-flash,MONGODB_URI=your-mongo-uri,SLACK_BOT_TOKEN=xoxb-...,SLACK_SIGNING_SECRET=..." \
  --set-secrets "GOOGLE_APPLICATION_CREDENTIALS=org-brain-sa:latest"
```

After deploy:

1. Copy Cloud Run URL
2. Slack app → **Event Subscriptions** → Request URL: `https://YOUR_URL/slack/events`
3. Slack app → **Slash Commands** → `/who-knows` Request URL: `https://YOUR_URL/slack/events`
4. Disable Socket Mode
5. Reinstall app to workspace

## Health check

```bash
curl https://YOUR_URL/health
# {"status":"ok","vertex_ai":"ready","slack":"ready","environment":"production"}
```

## Review timeline

Slack Marketplace review can take days. For hackathon judging, a **working installed app + demo video** is often sufficient even if Marketplace approval is pending. Submit early.
