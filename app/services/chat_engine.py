"""OpenAI chat engine with tool-calling loop."""
import json
from typing import List

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.rag.retriever import retrieve
from app.db.models import Lead

client = AsyncOpenAI(api_key=settings.openai_api_key)

MAX_TOOL_ITERATIONS = 8

_MARK_LEAD_PENDING_SENTINEL = "__MARK_LEAD_PENDING__"

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
            "name": "mark_lead_pending",
            "description": (
                "Call this BEFORE asking the visitor for their name and email. "
                "This signals that you are about to collect lead information. "
                "Do not call it more than once per lead collection flow."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    booking_info = (
        f"Share the booking link: {settings.calcom_booking_url}"
        if settings.calcom_booking_url
        else "ask them to contact us via email"
    )
    return f"""You are a helpful chat assistant for {settings.company_name}, embedded on the company website.

Your responsibilities:
1. **Answer questions** about the company — always call search_knowledge_base before answering company-specific questions.

2. **Capture leads** — when a visitor expresses interest in being contacted or learning more:
   a. Call mark_lead_pending once (this registers your intent server-side).
   b. Ask for their name and email (phone/company optional).
   c. When the visitor provides their contact info, call store_lead immediately.
   d. Confirm warmly after store_lead succeeds.

3. **Book meetings** — when a visitor wants to schedule a meeting:
   - {booking_info}.
   - Do not ask for name/email or try to book manually.

Guidelines:
- Keep responses concise and friendly — this is a chat widget.
- Always search the knowledge base before answering company-specific questions.
- If something is not in the knowledge base, say so honestly. Do not make up information.
- Detect the language of the user's first message and respond in that language for the entire conversation.
- After storing a lead, confirm it warmly to the visitor."""


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

        elif name == "mark_lead_pending":
            return _MARK_LEAD_PENDING_SENTINEL

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error: {str(e)}"


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

async def run_chat(
    messages: List[dict],
    db: Session,
    forced_tool: str | None = None,
) -> tuple[str, List[dict], str | None]:
    """
    Run one user turn through the tool-calling loop.

    Args:
        messages: Full conversation history (role/content dicts, no system message).
                  The latest user message must already be appended before calling.
        db: SQLAlchemy session for lead storage.
        forced_tool: If set, force the model to call this tool on the first iteration.

    Returns:
        (assistant_text, updated_messages, new_pending_action)
    """
    system = {"role": "system", "content": _system_prompt()}
    iterations = 0
    new_pending_action: str | None = None
    first_iteration = True

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1

        # Build the message list for this call
        call_messages = list(messages)

        # On the first iteration with a forced tool, inject a reinforcement hint
        # and force the tool_choice
        if first_iteration and forced_tool:
            # Inject reinforcement just before the last user message
            reinforcement = {
                "role": "system",
                "content": (
                    "The user just provided their contact details. "
                    "Extract name and email from their last message and call store_lead immediately. "
                    "IMPORTANT: Respond in the same language the user has been using throughout this conversation."
                ),
            }
            call_messages = call_messages[:-1] + [reinforcement] + call_messages[-1:]
            tool_choice = {"type": "function", "name": forced_tool}
        else:
            tool_choice = "auto"

        first_iteration = False

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[system] + call_messages,
            tools=TOOLS,
            tool_choice=tool_choice,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final text response
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages, new_pending_action

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

            if result == _MARK_LEAD_PENDING_SENTINEL:
                # Don't expose sentinel to the model; return empty ack
                new_pending_action = "store_lead"
                tool_result = "Noted."
            elif tc.function.name == "store_lead":
                tool_result = result
                if result.startswith("Lead saved successfully"):
                    new_pending_action = None
                elif forced_tool == "store_lead":
                    # store_lead was forced but failed (e.g. missing email) — keep pending
                    new_pending_action = "store_lead"
            else:
                tool_result = result

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # Safety: max iterations reached
    fallback = "I'm sorry, I ran into an issue processing your request. Please try again."
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages, new_pending_action
