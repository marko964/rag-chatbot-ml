"""OpenAI chat engine with tool-calling loop."""
import json
from typing import List

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.rag.retriever import retrieve
from app.db.models import Lead
from app.services.agent_states import AgentState, STATE_PROMPTS, STATE_TOOL_NAMES

client = AsyncOpenAI(api_key=settings.openai_api_key)

MAX_TOOL_ITERATIONS = 8

_MARK_LEAD_PENDING_SENTINEL = "__MARK_LEAD_PENDING__"
_REQUEST_SCHEDULING_SENTINEL = "__REQUEST_SCHEDULING__"

# ---------------------------------------------------------------------------
# Tool definitions (JSON schema for OpenAI)
# ---------------------------------------------------------------------------

_TOOL_SEARCH_KB = {
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
}

_TOOL_STORE_LEAD = {
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
}

_TOOL_MARK_LEAD_PENDING = {
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
}

_TOOL_REQUEST_SCHEDULING = {
    "type": "function",
    "function": {
        "name": "request_scheduling",
        "description": (
            "Call this when a visitor wants to book, schedule, or arrange a meeting. "
            "Do not ask for name/email first — call this immediately."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TOOL_REGISTRY: dict[str, dict] = {
    "search_knowledge_base": _TOOL_SEARCH_KB,
    "store_lead":            _TOOL_STORE_LEAD,
    "mark_lead_pending":     _TOOL_MARK_LEAD_PENDING,
    "request_scheduling":    _TOOL_REQUEST_SCHEDULING,
}


def _tools_for_state(state: AgentState) -> list[dict]:
    names = STATE_TOOL_NAMES.get(state, list(TOOL_REGISTRY.keys()))
    return [TOOL_REGISTRY[n] for n in names if n in TOOL_REGISTRY]


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

        elif name == "request_scheduling":
            return _REQUEST_SCHEDULING_SENTINEL

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
    state: AgentState = AgentState.KNOWLEDGE_BASE,
    forced_tool: str | None = None,
) -> tuple[str, List[dict], AgentState, str | None]:
    """
    Run one user turn through the tool-calling loop.

    Args:
        messages: Full conversation history (role/content dicts, no system message).
                  The latest user message must already be appended before calling.
        db: SQLAlchemy session for lead storage.
        state: Current agent state, determines system prompt and available tools.
        forced_tool: If set, force the model to call this tool on the first iteration.

    Returns:
        (assistant_text, updated_messages, new_state, new_pending_action)
    """
    system_content = STATE_PROMPTS.get(state, STATE_PROMPTS[AgentState.KNOWLEDGE_BASE])
    system = {"role": "system", "content": system_content}
    tools = _tools_for_state(state)

    iterations = 0
    new_pending_action: str | None = None
    new_state: AgentState = state
    first_iteration = True

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1

        # Build the message list for this call
        call_messages = list(messages)

        # On the first iteration with a forced tool, inject a reinforcement hint
        # and force the tool_choice
        if first_iteration and forced_tool:
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
            tools=tools,
            tool_choice=tool_choice,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final text response
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages, new_state, new_pending_action

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
                new_pending_action = "store_lead"
                new_state = AgentState.LEAD_QUALIFICATION
                tool_result = "Noted. You may now ask for the visitor's contact details."

            elif result == _REQUEST_SCHEDULING_SENTINEL:
                new_state = AgentState.SCHEDULING
                tool_result = "Scheduling mode activated."

            elif tc.function.name == "store_lead":
                tool_result = result
                if result.startswith("Lead saved successfully"):
                    new_pending_action = None
                    new_state = AgentState.KNOWLEDGE_BASE
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
    return fallback, messages, new_state, new_pending_action
