"""OpenAI chat engine with tool-calling loop."""
import json
from typing import List

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.rag.retriever import retrieve
from app.services import calcom
from app.db.models import Lead

client = AsyncOpenAI(api_key=settings.openai_api_key)

MAX_TOOL_ITERATIONS = 8

# ---------------------------------------------------------------------------
# Tool definitions (JSON schema for OpenAI)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base to answer questions about the company, "
                "its products, services, pricing, team, policies, etc. "
                "Always call this before answering company-specific questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, written as a short question or keywords.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_lead",
            "description": (
                "Save customer contact information when a visitor expresses interest in "
                "being contacted, wants more information, or requests a follow-up. "
                "Collect name and email at minimum before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "company": {"type": "string"},
                    "notes": {
                        "type": "string",
                        "description": "What the customer is interested in or their message.",
                    },
                },
                "required": ["name", "email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Check available meeting time slots for a specific date. "
                "Use this before booking to show the visitor their options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": (
                "Book a meeting with a visitor. Collect their name, email, and preferred "
                "date/time. The start_time must be an ISO 8601 datetime string "
                "(e.g. 2024-06-15T14:00:00Z). Check available slots first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 datetime, e.g. 2024-06-15T14:00:00Z",
                    },
                    "notes": {"type": "string"},
                },
                "required": ["name", "email", "start_time"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    return f"""You are a helpful chat assistant for {settings.company_name}, embedded on the company website.

Your responsibilities:
1. **Answer questions** about the company — always search the knowledge base first.
2. **Capture leads** — when a visitor wants to be contacted or learn more, collect their name and email (phone/company optional), then call store_lead.
3. **Book meetings** — when a visitor wants to schedule a meeting, collect their name, email, and preferred date. Check available slots, then book.

Guidelines:
- Keep responses concise and friendly — this is a chat widget.
- Always search the knowledge base before answering company-specific questions.
- If something is not in the knowledge base and you don't know it, say so honestly.
- Do not make up information about the company.
- After storing a lead or booking a meeting, confirm it warmly to the visitor."""


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _execute_tool(name: str, args: dict, db: Session) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if name == "search_knowledge_base":
            docs = retrieve(args["query"])
            if not docs:
                return "No relevant information found in the knowledge base."
            return "\n\n---\n\n".join(docs)

        elif name == "store_lead":
            lead = Lead(
                name=args["name"],
                email=args["email"],
                phone=args.get("phone"),
                company=args.get("company"),
                notes=args.get("notes"),
            )
            db.add(lead)
            db.commit()
            return f"Lead saved successfully for {args['name']} ({args['email']})."

        elif name == "get_available_slots":
            result = await calcom.get_available_slots(args["date"])
            return json.dumps(result)

        elif name == "book_meeting":
            result = await calcom.create_booking(
                name=args["name"],
                email=args["email"],
                start_time=args["start_time"],
                notes=args.get("notes", ""),
            )
            return json.dumps(result)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error: {str(e)}"


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

async def run_chat(messages: List[dict], db: Session) -> tuple[str, List[dict]]:
    """
    Run one user turn through the tool-calling loop.

    Args:
        messages: Full conversation history (role/content dicts, no system message).
                  The latest user message must already be appended before calling.
        db: SQLAlchemy session for lead storage.

    Returns:
        (assistant_text, updated_messages)
    """
    system = {"role": "system", "content": _system_prompt()}
    iterations = 0

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[system] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final text response
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages

        # Append assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = await _execute_tool(tc.function.name, args, db)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Safety: max iterations reached
    fallback = "I'm sorry, I ran into an issue processing your request. Please try again."
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages
