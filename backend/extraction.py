"""Extract structured employee information from Slack messages via Vertex AI Gemini."""

import json
import logging
import re
from typing import Optional

from vertexai.generative_models import GenerationConfig, GenerativeModel

from backend.schemas import ExtractionResult, HelpQueryResult, MatchReasonsResult, ProfileAboutResult

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an organizational intelligence extractor for workplace Slack messages.

Analyze the message below and extract employee-related information as JSON only.
Return a single JSON object with these fields:
- person (string, required): the person being described; use "unknown" if unclear
- role (string or null): job title or designation
- company (string or null): current or mentioned company
- skills (array of strings): technical or domain skills mentioned
- projects (array of strings): projects mentioned or owned
- confidence (number 0.0-1.0): how confident you are in this extraction

Rules:
- Output valid JSON only, no markdown fences or commentary
- Use empty arrays when skills or projects are not mentioned
- Set confidence low (< 0.3) when the message has no employee information

Message:
{message}
"""

HELP_TOPICS_PROMPT = """You parse workplace help requests to find what expertise is needed.

Analyze the message below and return JSON only with:
- topics (array of strings): skills, technologies, tools, or domains the user needs help with
- summary (string): one short sentence describing their issue or question
- confidence (number 0.0-1.0): how confident you are in the identified topics

Rules:
- Output valid JSON only, no markdown fences
- Extract concrete topics like "Kubernetes", "GraphQL", "SOC 2", "Python"
- Use empty topics array if no clear subject is mentioned
- summary should be under 120 characters

Message:
{message}
"""


class ExtractionError(Exception):
    """Raised when message extraction fails."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_extraction_response(raw_text: str) -> ExtractionResult:
    """Parse and validate Gemini JSON output into an ExtractionResult."""
    try:
        payload = json.loads(_strip_json_fences(raw_text))
        return ExtractionResult.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionError(
            "Failed to parse extraction response",
            detail=str(exc),
        ) from exc


def extract_employee_info(
    message: str,
    model: GenerativeModel,
    *,
    temperature: float = 0.1,
) -> ExtractionResult:
    """
    Extract structured employee information from a raw Slack message.

    Args:
        message: Raw Slack message text.
        model: Initialized Vertex AI GenerativeModel.
        temperature: Sampling temperature for generation.

    Returns:
        Validated ExtractionResult with person, role, company, skills, projects, confidence.

    Raises:
        ExtractionError: If Vertex AI fails or the response cannot be parsed.
    """
    if not message or not message.strip():
        raise ExtractionError("Message is empty")

    prompt = EXTRACTION_PROMPT.format(message=message.strip())

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            raise ExtractionError("Vertex AI returned an empty response")
        return _parse_extraction_response(raw_text)
    except ExtractionError:
        raise
    except Exception as exc:
        logger.exception("Vertex AI extraction failed")
        raise ExtractionError(
            "Vertex AI extraction failed",
            detail=str(exc),
        ) from exc


def _parse_help_response(raw_text: str) -> HelpQueryResult:
    """Parse and validate Gemini JSON output into a HelpQueryResult."""
    try:
        payload = json.loads(_strip_json_fences(raw_text))
        return HelpQueryResult.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionError(
            "Failed to parse help query response",
            detail=str(exc),
        ) from exc


def extract_help_topics(
    message: str,
    model: GenerativeModel,
    *,
    temperature: float = 0.1,
) -> HelpQueryResult:
    """
    Extract topics and issue summary from a natural-language help request.

    Args:
        message: User's help question, e.g. "I'm stuck on Kubernetes, who can help?"
        model: Initialized Vertex AI GenerativeModel.
        temperature: Sampling temperature for generation.

    Returns:
        HelpQueryResult with topics, summary, and confidence.

    Raises:
        ExtractionError: If Vertex AI fails or the response cannot be parsed.
    """
    if not message or not message.strip():
        raise ExtractionError("Message is empty")

    prompt = HELP_TOPICS_PROMPT.format(message=message.strip())

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            raise ExtractionError("Vertex AI returned an empty response")
        return _parse_help_response(raw_text)
    except ExtractionError:
        raise
    except Exception as exc:
        logger.exception("Vertex AI help topic extraction failed")
        raise ExtractionError(
            "Vertex AI help topic extraction failed",
            detail=str(exc),
        ) from exc


MATCH_REASONS_PROMPT = """You explain why each employee is a good match for a workplace help request.

Context (what the user needs help with):
{context}

Employee profiles (JSON array):
{profiles_json}

Return JSON only with:
- reasons (array of objects): each object has:
  - person (string): exact person name from the profile
  - reason (string): ONE concise sentence (max 120 chars) explaining why they are a good match

Rules:
- Output valid JSON only, no markdown fences
- One reason per profile, in the same order as input
- Mention role, relevant skills, or projects when available
- Be specific to the context — explain the fit, don't just repeat skills
- If profile data is sparse, say what is known briefly

Example reason: "Senior Backend Engineer with strong Kubernetes experience from platform and payment microservices work."
"""


def _profile_to_match_dict(profile) -> dict:
    """Serialize profile fields relevant for match reasoning."""
    return {
        "person": profile.person,
        "role": profile.role,
        "team": profile.team,
        "skills": profile.skills,
        "projects": profile.projects,
        "areas_of_expertise": profile.areas_of_expertise,
        "company": profile.company,
        "previous_company": profile.previous_company,
    }


def _normalize_match_reason_item(item: dict) -> dict:
    """Normalize alternate Gemini field names into person/reason."""
    person = item.get("person") or item.get("name") or item.get("employee") or ""
    reason = (
        item.get("reason")
        or item.get("explanation")
        or item.get("why")
        or item.get("summary")
        or ""
    )
    return {"person": str(person).strip(), "reason": str(reason).strip()}


def _parse_match_reasons_response(raw_text: str) -> MatchReasonsResult:
    """Parse and validate Gemini JSON output into MatchReasonsResult."""
    try:
        payload = json.loads(_strip_json_fences(raw_text))

        # Gemini often returns a bare array instead of {"reasons": [...]}
        if isinstance(payload, list):
            items = [_normalize_match_reason_item(i) for i in payload if isinstance(i, dict)]
            return MatchReasonsResult.model_validate({"reasons": items})

        if isinstance(payload, dict):
            for key in ("reasons", "matches", "results", "people"):
                if key in payload and isinstance(payload[key], list):
                    items = [
                        _normalize_match_reason_item(i)
                        for i in payload[key]
                        if isinstance(i, dict)
                    ]
                    return MatchReasonsResult.model_validate({"reasons": items})

        return MatchReasonsResult.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionError(
            "Failed to parse match reasons response",
            detail=str(exc),
        ) from exc


def generate_match_reasons(
    context: str,
    profiles: list,
    model: GenerativeModel,
    *,
    temperature: float = 0.2,
) -> dict[str, str]:
    """
    Generate one-line "why this person" reasons for expert search results.

    Args:
        context: Skill or issue the user needs help with.
        profiles: EmployeeProfile instances to explain.
        model: Initialized Vertex AI GenerativeModel.

    Returns:
        Mapping of person name to reason string. Empty dict on failure.
    """
    if not profiles or not context.strip():
        return {}

    profiles_json = json.dumps([_profile_to_match_dict(p) for p in profiles], indent=2)
    prompt = MATCH_REASONS_PROMPT.format(
        context=context.strip(),
        profiles_json=profiles_json,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            return {}
        result = _parse_match_reasons_response(raw_text)
        return {r.person: r.reason for r in result.reasons}
    except ExtractionError as exc:
        logger.warning("Match reason generation failed: %s", exc.message)
        return {}
    except Exception as exc:
        logger.exception("Vertex AI match reason generation failed")
        logger.warning("Match reason generation error: %s", exc)
        return {}


PROFILE_ABOUT_PROMPT = """You write engaging employee profile cards for a workplace Slack bot called Org Brain.

Employee profile (JSON):
{profile_json}

Return JSON only with:
- description (string): 2-3 sentence professional but friendly overview — role, skills, projects, what they're known for
- tagline (string): ONE quirky, memorable one-liner (max 100 chars) like a cool colleague referral.
  Example: "If you ever face a Kubernetes meltdown, don't worry — Aditya is your go-to."
  Use their first name. Be warm, slightly witty, workplace-appropriate. Not cringe.

Rules:
- Output valid JSON only, no markdown fences
- Base everything ONLY on profile data provided — do not invent facts
- If data is sparse, keep it brief and honest
"""


def _parse_profile_about_response(raw_text: str) -> ProfileAboutResult:
    """Parse Gemini JSON into ProfileAboutResult, tolerating bare-object or wrapped formats."""
    try:
        payload = json.loads(_strip_json_fences(raw_text))
        if isinstance(payload, dict):
            if "description" in payload and "tagline" in payload:
                return ProfileAboutResult.model_validate(payload)
            for key in ("about", "profile", "result"):
                if key in payload and isinstance(payload[key], dict):
                    return ProfileAboutResult.model_validate(payload[key])
        raise ValueError("Unexpected response shape")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionError(
            "Failed to parse profile about response",
            detail=str(exc),
        ) from exc


def generate_profile_about(
    profile,
    model: GenerativeModel,
    *,
    temperature: float = 0.4,
) -> ProfileAboutResult:
    """
    Generate a friendly description and quirky tagline for an employee profile.

    Args:
        profile: EmployeeProfile to describe.
        model: Initialized Vertex AI GenerativeModel.

    Returns:
        ProfileAboutResult with description and tagline.

    Raises:
        ExtractionError: If generation or parsing fails.
    """
    profile_json = json.dumps(_profile_to_match_dict(profile), indent=2)
    prompt = PROFILE_ABOUT_PROMPT.format(profile_json=profile_json)

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            raise ExtractionError("Vertex AI returned an empty response")
        return _parse_profile_about_response(raw_text)
    except ExtractionError:
        raise
    except Exception as exc:
        logger.exception("Vertex AI profile about generation failed")
        raise ExtractionError(
            "Vertex AI profile about generation failed",
            detail=str(exc),
        ) from exc


def fallback_profile_about(profile) -> ProfileAboutResult:
    """Build a basic profile card when Gemini is unavailable."""
    parts = []
    if profile.role:
        parts.append(f"{profile.person} is a {profile.role}")
    if profile.skills:
        parts.append(f"with expertise in {', '.join(profile.skills[:4])}")
    description = ". ".join(parts) + "." if parts else f"{profile.person} is part of your org."
    first = profile.person.split()[0]
    skill_hint = profile.skills[0] if profile.skills else "a tricky problem"
    tagline = f"If you ever hit a {skill_hint} snag, don't worry — {first} is your go-to."
    return ProfileAboutResult(description=description, tagline=tagline)
