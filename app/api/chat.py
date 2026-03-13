import re
import uuid
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.main import limiter
from app.services.chat_engine import run_chat
from app.services.session_store import session_store

router = APIRouter()

# Phrases that suggest prompt injection attempts
_INJECTION_PATTERNS = re.compile(
    r"(ignore (all |previous |prior )?(instructions?|prompts?|rules?)|"
    r"system prompt|jailbreak|disregard|act as (if )?you are|"
    r"you are now|pretend (you are|to be)|forget (everything|your instructions))",
    re.IGNORECASE,
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    message: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    session_id: str
    response: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    session_id = body.session_id or str(uuid.uuid4())

    if _INJECTION_PATTERNS.search(body.message):
        return ChatResponse(
            session_id=session_id,
            response="I can only answer questions about our company. How can I help you?",
        )

    history = session_store.get(session_id)
    history.append({"role": "user", "content": body.message})

    response_text, history = await run_chat(history, db)

    session_store.set(session_id, history)

    return ChatResponse(session_id=session_id, response=response_text)
