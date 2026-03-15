"""State machine definitions for the RAG agent."""
from enum import Enum
from typing import TypedDict

from app.config import settings


class AgentState(str, Enum):
    GREETING           = "GREETING"
    KNOWLEDGE_BASE     = "KNOWLEDGE_BASE"
    LEAD_QUALIFICATION = "LEAD_QUALIFICATION"
    SCHEDULING         = "SCHEDULING"


class QuickAction(TypedDict):
    label: str   # Button text shown in UI
    value: str   # Message sent back when button is clicked


STATE_ACTIONS: dict[AgentState, list[QuickAction]] = {
    AgentState.GREETING: [
        {"label": "Frage stellen",     "value": "Ich habe eine Frage."},
        {"label": "Termin buchen",     "value": "Ich möchte einen Termin buchen."},
        {"label": "Kontakt aufnehmen", "value": "Ich möchte Kontakt aufnehmen."},
    ],
    AgentState.KNOWLEDGE_BASE:     [],
    AgentState.LEAD_QUALIFICATION: [],
    AgentState.SCHEDULING: [
        {"label": "Weitere Fragen",    "value": "Ich habe noch weitere Fragen."},
        {"label": "Kontakt aufnehmen", "value": "Ich möchte Kontakt aufnehmen."},
    ],
}


def _kb_prompt() -> str:
    return f"""You are a helpful chat assistant for {settings.company_name}, embedded on the company website.

Your current role is to answer questions about the company (KNOWLEDGE_BASE mode).

Rules:
- Always call search_knowledge_base before answering any company-specific question.
- If the knowledge base returns no results, say so honestly. Never invent information.
- Keep answers concise and friendly — this is a chat widget.
- Detect the language of the user's messages and respond in that language throughout.
- When a visitor expresses interest in being contacted or wants a follow-up, call mark_lead_pending once, then ask for their name and email.
- When a visitor wants to book or schedule a meeting, call request_scheduling immediately — do not ask for personal details first.
- After storing a lead with store_lead, confirm warmly."""


def _lead_prompt() -> str:
    return f"""You are a helpful chat assistant for {settings.company_name}.

Your current role is to collect the visitor's contact information (LEAD_QUALIFICATION mode).

Rules:
- Ask for name and email if not yet provided. Phone and company are optional.
- As soon as the user provides name and email, call store_lead immediately.
- If the user refuses, acknowledge kindly and offer to help otherwise.
- Detect the user's language and respond in that language.
- IMPORTANT: Respond in the same language the user has been using throughout this conversation.
- After store_lead succeeds, confirm warmly and continue helping."""


def _scheduling_prompt() -> str:
    booking_info = (
        f"Share this booking link with the visitor: {settings.calcom_booking_url}"
        if settings.calcom_booking_url
        else "Tell the visitor to contact us via email to arrange a meeting."
    )
    return f"""You are a helpful chat assistant for {settings.company_name}.

Your current role is to help the visitor book a meeting (SCHEDULING mode).

Rules:
- {booking_info}
- Do not ask for name/email — the visitor books directly via the link.
- If the visitor asks a company question, call search_knowledge_base and answer it.
- Keep the tone warm and encouraging.
- Detect the user's language and respond in that language."""


# Evaluated once at import time (settings already loaded)
STATE_PROMPTS: dict[AgentState, str] = {
    AgentState.KNOWLEDGE_BASE:     _kb_prompt(),
    AgentState.LEAD_QUALIFICATION: _lead_prompt(),
    AgentState.SCHEDULING:         _scheduling_prompt(),
    # GREETING has no LLM call — handled directly in chat.py
}

STATE_TOOL_NAMES: dict[AgentState, list[str]] = {
    AgentState.KNOWLEDGE_BASE: [
        "search_knowledge_base",
        "mark_lead_pending",
        "store_lead",
        "request_scheduling",
    ],
    AgentState.LEAD_QUALIFICATION: [
        "store_lead",
        "search_knowledge_base",
        "request_scheduling",
    ],
    AgentState.SCHEDULING: [
        "search_knowledge_base",
        "mark_lead_pending",
        "store_lead",
        "request_scheduling",
    ],
}
