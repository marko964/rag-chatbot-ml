import re
import uuid
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.limiter import limiter
from app.services.agent_states import AgentState, QuickAction, STATE_ACTIONS
from app.services.graph import chat_graph
from app.services.session_store import session_store

router = APIRouter()

# Phrases that suggest prompt injection attempts
import re

_INJECTION_PATTERNS = re.compile(
    r"("
    # Englisch: ignore instructions / Deutsch: ignoriere Anweisungen
    r"(ignore|ignoriere|vergiss|forget) (all |alle |previous |vorherige |prior )?(instructions?|anweisungen?|prompts?|regeln?|rules?)|"
    # System-Begriffe
    r"system prompt|systemanweisung|jailbreak|disregard|missachte|"
    # Rollenspiele (Act as / Spiele / Tue so als ob)
    r"act as (if )?you are|tue so als ob du|handle als|du bist jetzt|you are now|"
    r"pretend (you are|to be)|gib vor zu sein|"
    # Alles vergessen
    r"forget (everything|your instructions)|alles vergessen"
    r")",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    message: str | None = Field(None, min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    state: str
    actions: list[QuickAction]


def _greeting_response(session_id: str) -> ChatResponse:
    greeting_text = (
        f"Hallo! 👋 Willkommen bei {settings.company_name}. "
        "Ich bin Ihr KI-Assistent und helfe Ihnen gerne bei Fragen zu unseren Leistungen – "
        "von Premium Webentwicklung über KI-Chatbots bis hin zu Workflow-Automatisierung. "
        "Was kann ich für Sie tun?"
    )
    return ChatResponse(
        session_id=session_id,
        response=greeting_text,
        state=AgentState.GREETING,
        actions=STATE_ACTIONS[AgentState.GREETING],
    )


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    session_id = body.session_id or str(uuid.uuid4())

    # No message → greeting (e.g. widget opened)
    if body.message is None:
        return _greeting_response(session_id)

    session_data = session_store.get(session_id)
    current_state = session_data["state"]

    # GREETING + first real message → transition to KNOWLEDGE_BASE
    if current_state == AgentState.GREETING and not session_data["messages"]:
        current_state = AgentState.KNOWLEDGE_BASE

    if _INJECTION_PATTERNS.search(body.message):
        return ChatResponse(
            session_id=session_id,
            response="I can only answer questions about our company. How can I help you?",
            state=current_state,
            actions=STATE_ACTIONS[current_state],
        )

    history = session_data["messages"]
    pending_action = session_data["pending_action"]
    history.append({"role": "user", "content": body.message})

    result = await chat_graph.ainvoke({
        "messages":       history,
        "current_node":   current_state.value,
        "pending_action": pending_action,
        "response":       "",
        "db":             db,
    })
    new_state = AgentState(result["current_node"])
    session_store.set(session_id, result["messages"],
                      pending_action=result["pending_action"], state=new_state)
    response_text = result["response"]

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        state=new_state,
        actions=STATE_ACTIONS[new_state],
    )
