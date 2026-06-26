# OrgBrain 🧠

> Find who knows what — without leaving Slack.

OrgBrain is a Slack-native expertise discovery platform that builds a **living expertise map** of your organization by silently learning from public Slack conversations.

No forms. No wikis. No manual tagging. Just ask.

---

## What It Does

Most organizations lose hours every week to one question: *"Who do I ask about this?"* OrgBrain answers that — instantly, inside Slack.

It listens to how your team talks, extracts skills and roles automatically using Gemini AI, and makes that knowledge searchable the moment you need it.

---

## Core Features

**Auto-Learning Profile Extraction**
Uses Gemini (Vertex AI) to analyze public Slack messages and extract user skills and roles in the background. Low-confidence signals are filtered out to keep profiles clean and noise-free.

**`/who-knows <skill>` — Expertise Discovery**
Find domain experts without context-switching. Ask directly in Slack and get matched to the right person instantly.

**Semantic Vector Search**
Powered by Qdrant. Understands conceptual matches — searching "infra" surfaces Kubernetes experts. Searching "payments" finds Stripe specialists.

**`/about <@user>` — Instant Profiles**
Pull up a team member's inferred expertise, role context, and skills at a glance.

**`/summarize <#channel>` — Channel Intelligence**
Summarize what a channel has been discussing — useful for async catch-up and onboarding.

**App Home Dashboard**
A central hub showing your own profile, org-wide skill distribution, and search — all without leaving Slack.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Slack Integration | Slack Bolt SDK (Socket Mode) |
| AI / LLM | Gemini via Vertex AI |
| Search | Real-Time Search (RTS) API |
| Vector DB | Qdrant |
| Primary DB | MongoDB Atlas |
| Cache | Redis (Upstash) |
| Backend | FastAPI |

---

## How It Works

```
Slack message posted
        ↓
Gemini extracts skills + roles (with confidence filtering)
        ↓
Profile stored in MongoDB, embedding indexed in Qdrant
        ↓
/who-knows query → semantic search → ranked expert list → posted in Slack
```

---

## Slash Commands

| Command | What it does |
|---|---|
| `/who-knows <skill>` | Find experts for a skill or topic |
| `/about <@user>` | View a team member's inferred profile |
| `/summarize <#channel>` | Get an AI summary of channel activity |
| `/intro` | Onboard yourself and set profile context |

---

## How to Run

### 1. Prerequisites
Ensure you have Python 3.10+ and Docker installed.

### 2. Environment Setup
Copy the example environment file and fill in your API keys and tokens:
```bash
cp .env.example backend/.env
```
*(Configure MongoDB, Qdrant, Redis, Slack, and Vertex AI credentials in `backend/.env`)*

### 3. Start Infrastructure
Launch the local database and services containerized:
```bash
docker compose up -d
```

### 4. Install Dependencies
Set up your virtual environment and install the required Python packages:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Start the Services

#### Run the Slack Bot (Socket Mode)
For local development, start the Slack Socket Mode daemon:
```bash
python -m backend.slack_app
```

#### Run the FastAPI Server & Dashboard
Start the web dashboard and REST API endpoints:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Visit `http://localhost:8000` to view the web dashboard.

---

## Built For

The **Slack Agent Builder Challenge** — Organizations Track.

OrgBrain is designed to scale across real enterprise teams, surface institutional knowledge that would otherwise stay siloed, and make expert-finding a zero-friction, zero-context-switch experience.