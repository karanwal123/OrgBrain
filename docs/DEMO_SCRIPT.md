# Org Brain — Hackathon Demo Script

2-minute demo for judges. Run with Socket Mode: `python -m backend.slack_app`

## Prerequisites

- MongoDB running: `docker compose up -d`
- Slack app configured per [SLACK_APP_SETUP.md](SLACK_APP_SETUP.md)
- Bot invited to `#engineering` and `#general`

---

## Act 1: Auto-learning (60 seconds)

**You say:** "Org Brain silently watches conversations and builds a living expertise map."

**Post in `#engineering`:**

> Hey team! I'm Priya Sharma, just joined as Senior Backend Engineer on Platform.
> Previously at Stripe. I work on payment microservices and Kubernetes infra.

**Post in `#engineering`:**

> Quick intro — Marcus Chen here, Staff ML Engineer. I lead the recommendation
> systems project. Deep experience with PyTorch, TensorFlow, and MLOps.

**Post in `#general`:**

> Shoutout to Lisa Park for the GraphQL migration! She's our go-to for API design
> and Apollo Federation.

**You say:** "No forms, no surveys — expertise is extracted automatically from normal Slack chatter."

---

## Act 2: Expertise discovery (30 seconds)

**Run in Slack:**

```
/who-knows kubernetes
```

**Expected:** Priya Sharma — Senior Backend Engineer — Skills: Kubernetes, microservices

**Run:**

```
/who-knows graphql
```

**Expected:** Lisa Park — Skills: GraphQL, API design, Apollo Federation

**You say:** "New hires and PMs can instantly find who to ping instead of scrolling months of history."

---

## Act 3: The problem + tech (30 seconds)

**You say:**

> "Organizations lose expertise in chat history. Onboarding is slow. People don't know who to ask.
> Org Brain solves this with Gemini extraction on Vertex AI, MongoDB profiles, and a `/who-knows` slash command inside Slack."

**Optional API demo:**

```bash
curl "http://localhost:8000/search?skill=kubernetes"
```

---

## Backup messages (if live extraction is slow)

If Vertex AI is slow, pre-seed via API:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "I am Priya Sharma, Senior Backend Engineer, expert in Kubernetes"}'
```

---

## Key talking points

1. **Not a chatbot** — organizational memory, not Q&A
2. **Continuous learning** — profiles improve with every relevant message
3. **Low noise** — low-confidence extractions are filtered out
4. **Slack-native** — slash command demo, no separate UI required
