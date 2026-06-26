"""FastAPI application entry point for the organizational intelligence agent."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from backend.memory import ChannelMemoryService, create_channel_memory_service
from backend.memory_worker import MemoryWorker
from backend.schemas import EmployeeProfile, ExtractionErrorResponse, ExtractionResult, MemoryRetrievalHit
from backend.storage import MemoryStorage, ProfileStorage

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


class MemoryStatusResponse(BaseModel):
    """MongoDB channel memory collection stats."""

    status: str
    database: str
    collection: str
    channel_count: int
    channels_with_pending: int


class MemorySearchResponse(BaseModel):
    """Semantic memory search results."""

    query: str
    channel_id: Optional[str] = None
    hits: list[MemoryRetrievalHit] = Field(default_factory=list)


# Module-level references initialized at startup.
_generative_model: Optional[GenerativeModel] = None
_profile_storage: Optional[ProfileStorage] = None
_memory_service: Optional[ChannelMemoryService] = None
_memory_worker: Optional[MemoryWorker] = None
_memory_storage: Optional[MemoryStorage] = None
_slack_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Vertex AI, storage, and Slack HTTP handler on startup."""
    global _generative_model, _profile_storage, _memory_service, _memory_worker, _memory_storage, _slack_handler
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

    try:
        _memory_service = create_channel_memory_service(settings, model=_generative_model)
        _memory_storage = _memory_service.storage
        logger.info("Channel memory service ready")
    except Exception as exc:
        logger.error("Channel memory service unavailable: %s", exc)
        _memory_service = ChannelMemoryService(settings=settings, model=_generative_model)
        _memory_storage = None

    _memory_worker = MemoryWorker(_memory_service, settings=settings)
    _memory_worker.start()
    asyncio.create_task(_memory_worker.run_once())
    logger.info("Memory worker started")

    if settings.slack_configured():
        try:
            from slack_bolt.adapter.fastapi import SlackRequestHandler

            from backend.slack_app import create_slack_app

            slack_app = create_slack_app(
                model=_generative_model,
                storage=_profile_storage,
                memory_service=_memory_service,
            )
            _slack_handler = SlackRequestHandler(slack_app)
            logger.info("Slack HTTP handler ready at /slack/events")
        except Exception as exc:
            logger.error("Slack handler failed to initialize: %s", exc)
            _slack_handler = None

    logger.info("Application startup complete")
    yield

    if _memory_worker is not None:
        await _memory_worker.stop()
    if _profile_storage is not None:
        _profile_storage.close()
    if _memory_storage is not None:
        _memory_storage.close()
    _generative_model = None
    _profile_storage = None
    _memory_service = None
    _memory_worker = None
    _memory_storage = None
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


def get_memory_service() -> ChannelMemoryService:
    """Return channel memory service or raise service-unavailable."""
    if _memory_service is None:
        raise HTTPException(status_code=503, detail="Channel memory service is not available.")
    return _memory_service


@app.get("/memory/status", response_model=MemoryStatusResponse)
async def memory_status() -> MemoryStatusResponse:
    """Return MongoDB channel memory collection stats."""
    if _memory_storage is None:
        raise HTTPException(status_code=503, detail="Memory storage is not available.")
    try:
        stats = _memory_storage.ping()
        return MemoryStatusResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Memory storage unavailable: {exc}") from exc


@app.get("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    q: str,
    channel_id: Optional[str] = None,
    limit: int = 10,
) -> MemorySearchResponse:
    """Search compressed channel memory by semantic relevance."""
    service = get_memory_service()
    hits = service.search(q, channel_id=channel_id, limit=limit)
    return MemorySearchResponse(query=q, channel_id=channel_id, hits=hits)


@app.post("/memory/flush")
async def flush_memory(channel_id: Optional[str] = None) -> dict:
    """Force delta summarization for one channel or all pending channels."""
    service = get_memory_service()
    if channel_id:
        service.flush_channel(channel_id)
        return {"flushed": [channel_id]}
    return {"flushed": service.flush_all()}


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


# Serve static files from backend/static
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_dashboard():
    """Serve the visual dashboard home page."""
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<h1>Welcome to Org Brain Dashboard</h1><p>Setup in progress...</p>")
    return FileResponse(index_path)
