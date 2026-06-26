"""Pydantic models for extraction output and employee profiles."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Structured employee information extracted from a Slack message."""

    person: str = Field(..., description="Identified person name or Slack display name")
    role: Optional[str] = Field(None, description="Job title or designation")
    company: Optional[str] = Field(None, description="Current or mentioned company")
    skills: list[str] = Field(default_factory=list, description="Technical or domain skills")
    projects: list[str] = Field(default_factory=list, description="Projects mentioned or owned")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the extraction (0.0–1.0)",
    )


class EmployeeProfile(BaseModel):
    """MongoDB document schema for a living employee profile."""

    person: str = Field(..., description="Canonical person identifier (name or Slack ID)")
    role: Optional[str] = None
    company: Optional[str] = None
    team: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    previous_company: Optional[str] = None
    experience: Optional[str] = None
    working_style: Optional[str] = None
    communication_style: Optional[str] = None
    availability: Optional[str] = None
    interests: list[str] = Field(default_factory=list)
    areas_of_expertise: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_messages: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def merge_extraction(self, extraction: ExtractionResult, source_message: str) -> "EmployeeProfile":
        """Merge a new extraction into this profile, keeping the highest-confidence values."""

        if extraction.role and (not self.role or extraction.confidence >= self.confidence):
            self.role = extraction.role
        if extraction.company and (not self.company or extraction.confidence >= self.confidence):
            self.company = extraction.company

        self.skills = sorted(set(self.skills) | set(extraction.skills))
        self.projects = sorted(set(self.projects) | set(extraction.projects))
        self.confidence = max(self.confidence, extraction.confidence)
        self.source_messages.append(source_message)
        self.updated_at = datetime.utcnow()
        return self


class ExtractionErrorResponse(BaseModel):
    """Error payload returned when extraction fails."""

    error: str
    message: str
    detail: Optional[str] = None


class HelpQueryResult(BaseModel):
    """Topics and summary extracted from a natural-language help request."""

    topics: list[str] = Field(
        default_factory=list,
        description="Skills, technologies, or domains the user needs help with",
    )
    summary: str = Field(
        default="",
        description="One-line summary of the user's issue or question",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that topics were correctly identified",
    )


class PersonMatchReason(BaseModel):
    """Gemini-generated reason why a person matches a help or skill query."""

    person: str
    reason: str = Field(..., description="One-line explanation of why this person is a good match")


class MatchReasonsResult(BaseModel):
    """Batch of match reasons for expert search results."""

    reasons: list[PersonMatchReason] = Field(default_factory=list)


class ProfileAboutResult(BaseModel):
    """Gemini-generated profile card for /about."""

    description: str = Field(..., description="2-3 sentence friendly profile overview")
    tagline: str = Field(
        ...,
        description="Quirky one-liner referral, e.g. 'If you ever face X, ___ is your go-to'",
    )


class TimelineEvent(BaseModel):
    """A single event on the channel summary timeline."""

    type: str = Field(default="event", description="decision | problem | solution | action")
    action_type: str = Field(..., description="Human-readable label, e.g. Decision Made")
    text: str = Field(..., description="What happened")
    speaker: str = Field(..., description="Slack user name or @mention")
    timestamp: str = Field(..., description="Relative or absolute time label")


class Decision(BaseModel):
    """A decision extracted from channel discussion."""

    decision: str
    decided_by: str
    approved_by: Optional[str] = None
    status: str = Field(default="approved")


class ProblemSolution(BaseModel):
    """A problem and its resolution from channel discussion."""

    problem: str
    reported_by: str
    solution: Optional[str] = None
    fixed_by: Optional[str] = None
    timestamp: Optional[str] = None
    impact: Optional[str] = None


class ActionItem(BaseModel):
    """An action item from channel discussion."""

    item: str
    owner: Optional[str] = "unassigned"
    due: Optional[str] = None
    status: str = Field(default="pending", description="pending | done")


class ChannelSummary(BaseModel):
    """Structured channel summary for /summarize Block Kit modal."""

    channel: str
    timeframe: str
    health: str = Field(default="healthy", description="healthy | warning | issues")
    participant_count: int = 0
    decision_count: int = 0
    timeline: list[TimelineEvent] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    problems: list[ProblemSolution] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    narrative: str = ""


class MemoryUnitType(str, Enum):
    """Canonical kinds of compressed organizational memory."""

    decision = "decision"
    problem = "problem"
    agreement = "agreement"
    action_item = "action_item"
    unresolved_issue = "unresolved_issue"
    context = "context"


class MemoryUnit(BaseModel):
    """Compressed semantic memory extracted from a channel delta."""

    memory_id: str = Field(..., description="Stable identifier for the memory unit")
    channel_id: str = Field(..., description="Slack channel identifier")
    unit_type: MemoryUnitType = Field(..., description="Type of compressed memory")
    summary: str = Field(..., description="Canonical memory statement")
    source_message_ids: list[str] = Field(default_factory=list)
    source_timestamps: list[datetime] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    unresolved: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChannelMemoryState(BaseModel):
    """Per-channel durable state for delta summarization and retrieval."""

    channel_id: str = Field(..., description="Slack channel identifier")
    memory_store: list[MemoryUnit] = Field(default_factory=list)
    last_summary_ts: Optional[datetime] = None
    last_summary_timestamp: Optional[datetime] = None
    last_processed_message_id: Optional[str] = None
    compressed_context: str = ""
    cached_summary_state: str = ""
    pending_messages: list[str] = Field(default_factory=list)
    pending_message_payloads: list[dict] = Field(
        default_factory=list,
        description="Buffered Slack messages awaiting the next delta summarization cycle",
    )
    cached_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryRetrievalHit(BaseModel):
    """Ranked memory result returned from semantic retrieval."""

    memory_id: str
    channel_id: str
    summary: str
    unit_type: MemoryUnitType
    score: float = Field(..., ge=0.0)
    semantic_score: float = Field(..., ge=0.0)
    recency_score: float = Field(..., ge=0.0)
    importance_score: float = Field(..., ge=0.0)
    unresolved_score: float = Field(..., ge=0.0)
    source_message_ids: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemoryDeltaBatch(BaseModel):
    """New messages and checkpoint metadata to summarize as a delta."""

    channel_id: str
    last_summary_ts: Optional[datetime] = None
    last_processed_message_id: Optional[str] = None
    messages: list[dict] = Field(default_factory=list)


class MemoryCheckpoint(BaseModel):
    """Stored checkpoint for idempotent channel summarization."""

    channel_id: str
    last_summary_ts: Optional[datetime] = None
    last_processed_message_id: Optional[str] = None
    cached_summary_state: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AvailabilityStatus(str, Enum):
    """Valid status values for employee availability."""

    free = "free"
    busy = "busy"
    leave = "leave"


class AvailabilityEntry(BaseModel):
    """MongoDB document schema for an employee availability entry."""

    user_id: str = Field(..., description="Slack user ID")
    user_name: str = Field(default="", description="Slack username")
    user_display_name: str = Field(default="", description="Slack display name")
    user_email: str = Field(default="", description="User email from Slack profile")
    team_id: str = Field(default="", description="Slack workspace/team ID")

    date_start: str = Field(..., description="Start date YYYY-MM-DD")
    date_end: str = Field(..., description="End date YYYY-MM-DD")
    time_start: str = Field(default="00:00", description="Start time HH:MM 24hr")
    time_end: str = Field(default="23:59", description="End time HH:MM 24hr")

    status: AvailabilityStatus = Field(..., description="free, busy, or leave")
    reason: Optional[str] = Field(default=None, description="Optional reason")

    channel_id: str = Field(default="", description="Slack channel where posted")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    timezone: str = Field(default="Asia/Kolkata", description="User timezone")

