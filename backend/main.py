"""FastAPI application entry point for the organizational intelligence agent."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel

from backend.config import (
    configure_logging,
    get_settings,
    init_vertex_ai,
    parse_allowed_origins,
)
from backend.extraction import ExtractionError, extract_employee_info
from backend.schemas import EmployeeProfile, ExtractionErrorResponse, ExtractionResult
from backend.storage import ProfileStorage

logger = logging.getLogger(__name__)


class ExtractRequest(BaseModel):
    """Request body for the extraction endpoint."""

    message: str = Field(..., min_length=1, description="Raw Slack message text")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vertex_ai: str
    slack: str
    mongodb: str
    environment: str


class DbStatusResponse(BaseModel):
    """MongoDB connectivity and collection stats."""

    status: str
    database: str
    collection: str
    profile_count: int


# Module-level references initialized at startup.
_generative_model: Optional[GenerativeModel] = None
_profile_storage: Optional[ProfileStorage] = None
_slack_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Vertex AI, storage, and Slack HTTP handler on startup."""
    global _generative_model, _profile_storage, _slack_handler
    settings = get_settings()
    configure_logging(settings)

    try:
        _generative_model = init_vertex_ai(settings)
        logger.info("Vertex AI ready")
    except RuntimeError as exc:
        logger.error("Vertex AI unavailable at startup: %s", exc)
        _generative_model = None

    try:
        _profile_storage = ProfileStorage(settings)
        logger.info("MongoDB profile storage ready")
    except Exception as exc:
        logger.error("MongoDB unavailable at startup: %s", exc)
        _profile_storage = None

    if settings.slack_configured():
        try:
            from slack_bolt.adapter.fastapi import SlackRequestHandler

            from backend.slack_app import create_slack_app

            slack_app = create_slack_app(model=_generative_model, storage=_profile_storage)
            _slack_handler = SlackRequestHandler(slack_app)
            logger.info("Slack HTTP handler ready at /slack/events")
        except Exception as exc:
            logger.error("Slack handler failed to initialize: %s", exc)
            _slack_handler = None

    logger.info("Application startup complete")
    yield

    if _profile_storage is not None:
        _profile_storage.close()
    _generative_model = None
    _profile_storage = None
    _slack_handler = None
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Org Brain",
    description="Slack organizational intelligence agent — extracts and stores employee expertise from conversations.",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(_settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model() -> GenerativeModel:
    """Return the initialized GenerativeModel or raise a service-unavailable error."""
    if _generative_model is None:
        raise HTTPException(
            status_code=503,
            detail="Vertex AI is not initialized. Check credentials and configuration.",
        )
    return _generative_model


def get_storage() -> ProfileStorage:
    """Return profile storage or raise service-unavailable."""
    if _profile_storage is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available. Start docker compose up -d.",
        )
    return _profile_storage


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and dependency availability."""
    settings = get_settings()
    slack_status = "ready" if _slack_handler is not None else "unavailable"
    mongo_status = "unavailable"
    if _profile_storage is not None:
        try:
            _profile_storage.ping()
            mongo_status = "ready"
        except Exception:
            mongo_status = "unavailable"
    return HealthResponse(
        status="ok",
        vertex_ai="ready" if _generative_model is not None else "unavailable",
        slack=slack_status,
        mongodb=mongo_status,
        environment=settings.app_env,
    )


@app.get("/db/status", response_model=DbStatusResponse)
async def db_status() -> DbStatusResponse:
    """Return MongoDB connection status and profile count."""
    storage = get_storage()
    try:
        stats = storage.ping()
        return DbStatusResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {exc}") from exc


@app.get("/profiles", response_model=list[EmployeeProfile])
async def list_profiles(limit: int = 100) -> list[EmployeeProfile]:
    """List all stored employee profiles."""
    storage = get_storage()
    return storage.list_profiles(limit=limit)


@app.post(
    "/extract",
    response_model=ExtractionResult,
    responses={
        422: {"description": "Invalid input"},
        502: {"model": ExtractionErrorResponse, "description": "Extraction failed"},
        503: {"description": "Vertex AI unavailable"},
    },
)
async def extract(request: ExtractRequest) -> ExtractionResult:
    """
    Extract structured employee information from a Slack message.

    Returns JSON with person, role, company, skills, projects, and confidence.
    """
    model = get_model()
    try:
        return extract_employee_info(request.message, model)
    except ExtractionError as exc:
        logger.warning("Extraction failed: %s", exc.message)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "extraction_failed",
                "message": exc.message,
                "detail": exc.detail,
            },
        ) from exc


@app.get("/search", response_model=list[EmployeeProfile])
async def search_profiles(skill: str, limit: int = 10) -> list[EmployeeProfile]:
    """
    Search employee profiles by skill, project, or expertise area.

    Args:
        skill: Case-insensitive search term.
        limit: Maximum results to return.
    """
    storage = get_storage()
    return storage.search_by_skill(skill, limit=limit)


@app.post("/slack/events")
async def slack_events(request: Request):
    """Slack Events API + slash command endpoint (HTTP mode for production)."""
    if _slack_handler is None:
        raise HTTPException(
            status_code=503,
            detail="Slack is not configured. Set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET.",
        )
    return await _slack_handler.handle(request)
