"""Environment configuration and Vertex AI GenerativeModel initialization."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import vertexai
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from backend/.env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Google Cloud / Vertex AI
    google_application_credentials: Optional[str] = Field(
        default=None,
        description="Path to GCP service account JSON (relative to backend/ or absolute)",
    )
    google_cloud_project: str
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION"),
    )
    vertex_model_name: str = Field(
        default="gemini-2.0-flash-001",
        validation_alias=AliasChoices("VERTEX_MODEL_NAME", "GEMINI_MODEL"),
    )
    google_maps_api_key: Optional[str] = None

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "org_brain"
    mongodb_profiles_collection: str = "employee_profiles"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "skill_embeddings"
    embedding_dimension: int = 768

    # Slack
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None
    extraction_confidence_threshold: float = 0.3

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    allowed_origins: str = "*"

    def slack_configured(self) -> bool:
        """Return True when Slack bot token and signing secret are set."""
        return bool(self.slack_bot_token and self.slack_signing_secret)

    def slack_socket_mode_ready(self) -> bool:
        """Return True when all tokens required for Socket Mode are set."""
        return self.slack_configured() and bool(self.slack_app_token)


def resolve_credentials_path(credentials: str) -> Path:
    """
    Resolve a credentials path relative to backend/ when not absolute.

    Args:
        credentials: Path from GOOGLE_APPLICATION_CREDENTIALS.

    Returns:
        Absolute resolved path to the credentials file.
    """
    path = Path(credentials)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()


def configure_google_credentials(settings: Settings) -> Optional[Path]:
    """
    Export GOOGLE_APPLICATION_CREDENTIALS to an absolute path for GCP SDKs.

    Args:
        settings: Application settings.

    Returns:
        Resolved credentials path when configured, otherwise None.

    Raises:
        FileNotFoundError: If the credentials file does not exist.
    """
    if not settings.google_application_credentials:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set")
        return None

    creds_path = resolve_credentials_path(settings.google_application_credentials)
    if not creds_path.is_file():
        raise FileNotFoundError(f"Google credentials file not found: {creds_path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    logger.info("Google credentials configured: %s", creds_path)
    return creds_path


def parse_allowed_origins(origins: str) -> list[str]:
    """Parse ALLOWED_ORIGINS into a list for CORS middleware."""
    if origins.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings loaded from backend/.env."""
    return Settings()


def init_vertex_ai(settings: Optional[Settings] = None) -> GenerativeModel:
    """
    Initialize Vertex AI and return a configured GenerativeModel instance.

    Args:
        settings: Optional settings override. Uses cached settings when omitted.

    Returns:
        A ready-to-use GenerativeModel for extraction.

    Raises:
        RuntimeError: If credentials or Vertex AI initialization fails.
    """
    cfg = settings or get_settings()

    try:
        configure_google_credentials(cfg)
        vertexai.init(project=cfg.google_cloud_project, location=cfg.google_cloud_location)
        model = GenerativeModel(cfg.vertex_model_name)
        logger.info(
            "Vertex AI initialized (project=%s, location=%s, model=%s)",
            cfg.google_cloud_project,
            cfg.google_cloud_location,
            cfg.vertex_model_name,
        )
        return model
    except FileNotFoundError as exc:
        logger.exception("Google credentials missing")
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to initialize Vertex AI")
        raise RuntimeError(f"Vertex AI initialization failed: {exc}") from exc


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure root logging based on settings."""
    cfg = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
