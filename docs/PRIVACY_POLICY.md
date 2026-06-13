# Privacy Policy — Org Brain

**Last updated:** June 2025

## Overview

Org Brain ("we", "the app") is a Slack application that extracts organizational expertise from workplace conversations to help teams discover who has relevant skills.

## Data we collect

- **Slack message text** from channels where the bot is invited
- **Extracted profile data**: names, roles, skills, projects, and confidence scores
- **Slack metadata**: channel IDs, user IDs (for slash command responses)

## Data we do not collect

- Direct messages in channels where the bot is not present
- Passwords, payment information, or unrelated personal data
- Full workspace message history on install

## How we use data

- Extract structured employee expertise using Google Vertex AI (Gemini)
- Store profiles in MongoDB for expertise search
- Return search results via the `/who-knows` slash command

## Data storage

- Profiles stored in MongoDB (configurable host)
- Embeddings stored in Qdrant (when enabled)
- Google Cloud credentials used only for Vertex AI API calls

## Data retention

- Profiles persist until manually deleted by a workspace admin
- Source messages are stored for provenance and debugging

## Third-party services

| Service | Purpose |
|---------|---------|
| Slack | Message delivery and slash commands |
| Google Vertex AI | Information extraction |
| MongoDB | Profile storage |
| Qdrant | Skill embeddings (optional) |

## Security

- API keys and credentials stored in environment variables, never in source code
- `.env` and service account files are gitignored
- HTTPS required for production Slack endpoints

## Your rights

Workspace admins can:
- Remove the app to stop data collection
- Request profile deletion by contacting the app developer

## Contact

For privacy questions, contact your hackathon team or open an issue in the project repository.

## Changes

We may update this policy. Continued use of the app constitutes acceptance of changes.
