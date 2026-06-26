# OrgBrain

Identify team expertise and schedule availability directly inside Slack.

![OrgBrain Dashboard Banner](global_assest/for_readme.png)

OrgBrain is a Slack-native organizational intelligence agent. By analyzing public conversation dynamics and structured commands, it automatically extracts employee capabilities, builds a real-time expertise map, and tracks availability without requiring manual profiles or form updates.

---

## Quickstart

### 1. Prerequisites
Ensure Python 3.10+ and Docker are installed on the host machine.

### 2. Configuration
Copy the template configuration file to the backend directory:
```bash
cp .env.example backend/.env
```
Populate `backend/.env` with the necessary API keys and credentials for MongoDB, Qdrant, Redis, Slack, and Google Cloud Vertex AI.

### 3. Start Databases
Launch local containerized database infrastructure:
```bash
docker compose up -d
```

### 4. Installation
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Run Services
Run both services in separate terminals to start the application:

*   **Slack Integration (Socket Mode):**
    ```bash
    python -m backend.slack_app
    ```

*   **FastAPI Dashboard & REST API:**
    ```bash
    python -m uvicorn backend.main:app --reload --port 8000
    ```

Access the dashboard at `http://localhost:8000`.

---

## Core Features

*   **Automated Capability Extraction:** Automatically parses public messages using Gemini on Vertex AI to build employee profiles. Low-confidence extractions are automatically filtered out to preserve dataset quality.
*   **Semantic Expert Discovery:** Matches users to skills using conceptual relationships (e.g., searching for "infrastructure" returns users tagged with Kubernetes).
*   **Calendar & Availability Tracking:** Allows team members to declare their schedules and query team availability in real time using text-based slash commands or an interactive modal.
*   **Channel Intelligence:** Creates automated activity summaries and permits semantic search query lookups across channel conversation history.

---

## Slash Commands

| Command | Function |
|---|---|
| `/who-knows <skill>` | Search for team members with a specific skill or area of expertise. |
| `/about <@user>` | Retrieve the automatically generated profile for a specific user. |
| `/summarize <#channel>` | Open an interactive modal summarizing recent conversation history. |
| `/recall <query>` | Conduct a semantic search query across public channel memories. |
| `/calendar` | Open the interactive schedule planner modal. |
| `/calendar status <status> on <date>` | Declare availability (e.g., `/calendar status leave on May 5`). |
| `/calendar who-is-free <date>` | List all team members who are free on a given date. |
| `/calendar team-calendar` | Display an overview of the upcoming schedule for the team. |
| `/intro <introduction>` | Introduce yourself manually to register skills. |

---

## Technical Architecture

```
Slack Message Ingestion
        │
        ▼
Vertex AI Gemini Extraction (Confidence Filtering Applied)
        │
        ▼
MongoDB (Profiles & Metadata) & Qdrant (Vector Embeddings)
        │
        ▼
Semantic Match / Search Execution -> Slack Block Kit Output
```

### Stack Components
*   **Slack Bolt SDK:** Powers the interactive Slack application and Socket Mode listener.
*   **Vertex AI (Gemini):** Extracts structured capabilities and generates natural-language profiles.
*   **MongoDB Atlas:** Stores structured employee profiles and metadata.
*   **Qdrant:** Houses high-dimensional vector embeddings for semantic search.
*   **Redis (Upstash):** Handles caching layer and memory indexes.
*   **FastAPI:** Serves the backend API endpoints and web dashboard.

---

## Deployment Guidelines

For production environments, the recommended path is deploying to **Google Cloud Run**:
*   The application includes a standard `Dockerfile` that packages the FastAPI application.
*   By configuring a GCP Service Account with the appropriate Vertex AI permissions, Cloud Run handles GCP API authentication natively without requiring storage of a service account private key file in the container.
*   Ensure the Slack application configuration is switched from Socket Mode to HTTP Webhook Mode, pointing to the `/slack/events` endpoint.
*   Utilize managed serverless offerings for database infrastructure (MongoDB Atlas, Upstash Redis, and Qdrant Cloud) to avoid container management overhead.

---

## Project Status

Developed for the **Slack Agent Builder Challenge** (Organizations Track). OrgBrain is built to scale across enterprise Slack workspaces, uncovering institutional knowledge that is typically lost in chat history.