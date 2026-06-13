"""
TransitGuard AI - LangChain Dispatcher Agent

Takes accident type, severity score, severity level, timestamp, and location,
then returns a structured emergency dispatch decision.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from langchain.agents import create_agent

from pydantic import BaseModel, Field


PipelineSeverityLevel = Literal["MINOR", "LOW", "MODERATE", "MAJOR", "CRITICAL"]
ResponseLevel = Literal["LOW", "MODERATE", "MAJOR", "CRITICAL"]
Responder = Literal["Traffic Control", "Police", "Ambulance", "Hospital"]


class DispatchResponse(BaseModel):
    response_level: ResponseLevel
    notify: list[Responder]
    incident_summary: str


class AccidentDispatchInput(BaseModel):
    accident_type: str = Field(..., description="Accident type from AccidentDetector")
    severity_score: int = Field(..., ge=0, le=100)
    severity_level: PipelineSeverityLevel
    timestamp: str
    location: str


DISPATCH_RULES: dict[ResponseLevel, list[Responder]] = {
    "LOW": ["Traffic Control"],
    "MODERATE": ["Police"],
    "MAJOR": ["Police", "Ambulance"],
    "CRITICAL": ["Police", "Ambulance", "Hospital"],
}


def normalize_severity_level(level: str) -> ResponseLevel:
    """Convert current pipeline labels into dispatcher response levels."""

    normalized = level.upper()
    if normalized == "MINOR":
        return "LOW"
    if normalized in DISPATCH_RULES:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported severity level: {level}")


def dispatch_policy(
    accident_type: str,
    severity_score: int,
    severity_level: PipelineSeverityLevel,
    timestamp: str,
    location: str,
) -> dict:
    """Return the required response_level and notify list for an accident.
    Does NOT include incident_summary - the agent must write that itself."""

    response_level = normalize_severity_level(severity_level)
    notify = DISPATCH_RULES[response_level]

    return {
        "response_level": response_level,
        "notify": notify,
    }


def dispatch_locally(
    accident_type: str,
    severity_score: int,
    severity_level: PipelineSeverityLevel,
    timestamp: str,
    location: str,
) -> DispatchResponse:
    """Return the dispatch decision without calling an LLM."""

    policy = dispatch_policy(
        accident_type=accident_type,
        severity_score=severity_score,
        severity_level=severity_level,
        timestamp=timestamp,
        location=location,
    )

    return DispatchResponse(
        response_level=policy["response_level"],
        notify=policy["notify"],
        incident_summary=(
            f"{policy['response_level']} {accident_type} detected at {location} "
            f"at {timestamp}. Severity score: {severity_score}/100. "
            f"Notify: {', '.join(policy['notify'])}."
        ),
    )


SYSTEM_PROMPT = """
You are the TransitGuard AI Dispatcher Agent - an autonomous emergency
response coordinator analyzing live CCTV accident detections.

Always use the dispatch_policy tool to determine response_level and notify list
- this is the source of truth and must not be overridden.

Dispatch rules:
- LOW -> Traffic Control
- MODERATE -> Police
- MAJOR -> Police + Ambulance
- CRITICAL -> Police + Ambulance + Hospital

If the severity level is MINOR, treat it as LOW.

For the incident_summary field, write a brief (1-2 sentence) natural-language
description of the situation as a human dispatcher would announce it over radio.
Mention what type of accident occurred, its severity, and which responders are
being notified and why. Be concise but make it sound like a real-time alert,
not a robotic log entry.
Return only structured JSON matching the DispatchResponse schema.
"""


def build_dispatcher_agent(model: str | None = None):
    """Build the LangChain dispatcher agent using create_agent."""

    return create_agent(
        model=model or os.getenv("TRANSITGUARD_MODEL", "groq:llama-3.1-8b-instant"),
        tools=[dispatch_policy],
        system_prompt=SYSTEM_PROMPT,
        response_format=DispatchResponse,
    )


def dispatch_incident(
    accident_type: str,
    severity_score: int,
    severity_level: PipelineSeverityLevel,
    timestamp: str,
    location: str,
    model: str | None = None,
) -> DispatchResponse:
    """Dispatch one incident and return a validated structured response."""

    accident = AccidentDispatchInput(
        accident_type=accident_type,
        severity_score=severity_score,
        severity_level=severity_level,
        timestamp=timestamp,
        location=location,
    )

    try:
        agent = build_dispatcher_agent(model=model)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": accident.model_dump_json(),
                    }
                ]
            }
        )
        return result["structured_response"]
    except Exception as exc:
        print(f"LLM unavailable, using local dispatcher fallback: {exc}")
        return dispatch_locally(
            accident_type=accident.accident_type,
            severity_score=accident.severity_score,
            severity_level=accident.severity_level,
            timestamp=accident.timestamp,
            location=accident.location,
        )


if __name__ == "__main__":
    response = dispatch_incident(
        accident_type="vehicle-vehicle",
        severity_score=51,
        severity_level="MAJOR",
        timestamp="frame 48",
        location="Camera-01 / Main Road Junction",
    )
    print(json.dumps(response.model_dump(), indent=2))
