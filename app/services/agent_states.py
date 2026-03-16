"""State machine definitions for the RAG agent."""
from enum import Enum

from pydantic import BaseModel

from app.config import settings


class AgentState(str, Enum):
    GREETING           = "GREETING"
    KNOWLEDGE_BASE     = "KNOWLEDGE_BASE"
    LEAD_QUALIFICATION = "LEAD_QUALIFICATION"
    SCHEDULING         = "SCHEDULING"


class QuickAction(BaseModel):
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
    return f"""Du bist ein freundlicher KI-Assistent von {settings.company_name}, eingebettet auf der Unternehmenswebsite.

{settings.company_name} ist eine österreichische Digitalagentur. Inhaber ist Marko Lazendic (HTL Informatik-Absolvent).
Die drei Kernleistungen sind:
1. **Premium Webentwicklung** – maßgeschneiderte Websites mit React, Vite & GSAP, optimiert auf Sales und Sichtbarkeit.
2. **KI-Lösungen / AI Chatbots** – intelligente Assistenten für Support, Lead-Generierung und Reservierungen, 24/7 verfügbar.
3. **Workflow-Automatisierung** – Eliminierung repetitiver Aufgaben (CRM-Sync, automatisierte E-Mails, KI-Recherche).

Alle Lösungen sind 100 % DSGVO-konform. Projekte dauern typisch 2–4 Wochen.

Deine Aufgaben (KNOWLEDGE_BASE-Modus):
- Rufe immer search_knowledge_base auf, bevor du unternehmensspezifische Fragen beantwortest.
- Wenn die Wissensdatenbank keine Ergebnisse liefert, sag das ehrlich. Erfinde keine Informationen.
- Halte Antworten kurz und freundlich – dies ist ein Chat-Widget.
- Erkenne die Sprache des Nutzers und antworte durchgehend in dieser Sprache.
- Dein primäres Ziel: Den Besucher dazu zu bewegen, ein Projekt zu starten oder einen Termin zu buchen.
- Wenn ein Besucher Interesse zeigt, kontaktiert zu werden oder mehr erfahren möchte: Rufe mark_lead_pending einmal auf, dann frage nach Name und E-Mail.
- Wenn ein Besucher einen Termin buchen möchte: Rufe sofort request_scheduling auf – frage NICHT zuerst nach persönlichen Daten.
- Bestätige nach erfolgreichem store_lead herzlich."""


def _lead_prompt() -> str:
    return f"""Du bist ein freundlicher KI-Assistent von {settings.company_name}.

Deine aktuelle Aufgabe ist es, die Kontaktdaten des Besuchers zu erfassen (LEAD_QUALIFICATION-Modus).

Regeln:
- Frage nach Name und E-Mail, falls noch nicht angegeben. Telefon und Unternehmen sind optional.
- Sobald der Nutzer Name und E-Mail nennt, rufe sofort store_lead auf.
- Falls der Nutzer ablehnt, reagiere verständnisvoll und biete anderweitige Hilfe an.
- Antworte IMMER in der Sprache, die der Nutzer bisher verwendet hat.
- Nach erfolgreichem store_lead: Bestätige herzlich (z. B. „Danke! Marko wird sich bald bei Ihnen melden.") und biete an, weitere Fragen zu beantworten."""


def _scheduling_prompt() -> str:
    booking_info = (
        f"Teile dem Besucher diesen Buchungslink mit: {settings.calcom_booking_url}"
        if settings.calcom_booking_url
        else "Bitte den Besucher, uns per E-Mail zu kontaktieren, um einen Termin zu vereinbaren."
    )
    return f"""Du bist ein freundlicher KI-Assistent von {settings.company_name}.

Deine aktuelle Aufgabe ist es, dem Besucher beim Buchen eines Termins zu helfen (SCHEDULING-Modus).

Regeln:
- {booking_info}
- Frage NICHT nach Name oder E-Mail – der Besucher bucht direkt über den Link.
- Falls der Besucher eine Frage zum Unternehmen stellt, rufe search_knowledge_base auf und beantworte sie.
- Halte den Ton warm und einladend. Betone, dass Marko sich freut, das Projekt gemeinsam zu besprechen.
- Antworte in der Sprache, die der Nutzer verwendet."""


def get_state_prompt(state: AgentState) -> str:
    """Called per-request so settings (e.g. CALCOM_BOOKING_URL) are always current."""
    if state == AgentState.KNOWLEDGE_BASE:
        return _kb_prompt()
    elif state == AgentState.LEAD_QUALIFICATION:
        return _lead_prompt()
    return _scheduling_prompt()

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
