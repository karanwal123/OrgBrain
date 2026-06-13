"""Pydantic models for extraction output and employee profiles."""

from datetime import datetime
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
    solution: str
    fixed_by: str
    timestamp: Optional[str] = None
    impact: Optional[str] = None


class ActionItem(BaseModel):
    """An action item from channel discussion."""

    item: str
    owner: str
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
