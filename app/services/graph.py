"""LangGraph-powered RAG agent — nodes + routing + compiled graph."""
import json
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Lead
from app.rag.retriever import retrieve
from app.services.agent_states import AgentState, STATE_TOOL_NAMES, get_state_prompt

# ---------------------------------------------------------------------------
# Shared OpenAI client
# ---------------------------------------------------------------------------

client = AsyncOpenAI(api_key=settings.openai_api_key)

MAX_TOOL_ITERATIONS = 8
_CALCOM_FALLBACK = "https://cal.com/ml-solutions-at/website-termin"

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ChatState(TypedDict):
    messages: list[dict]        # conversation history (role/content dicts)
    current_node: str           # AgentState value, e.g. "KNOWLEDGE_BASE"
    pending_action: str | None  # "store_lead" if mid-lead-collection, else None
    response: str               # text reply for this turn (ephemeral, not stored)
    db: Any                     # SQLAlchemy Session, injected per request


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
# Shared LLM loop
# ---------------------------------------------------------------------------

async def _run_llm_loop(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
    db: Any,
    forced_tool: str | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> tuple[str, list[dict], str | None, str | None]:
    """
    Run the OpenAI tool-calling loop.

    Returns:
        (response_text, updated_messages, signal_node, new_pending_action)
        signal_node: "LEAD_QUALIFICATION" | "SCHEDULING" | None
    """
    system = {"role": "system", "content": system_prompt}
    iterations = 0
    signal_node: str | None = None
    new_pending_action: str | None = None
    first_iteration = True

    while iterations < max_iterations:
        iterations += 1

        call_messages = list(messages)

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
            tool_choice: Any = {"type": "function", "name": forced_tool}
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
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages, signal_node, new_pending_action

        # Append assistant message with tool calls
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

        # Execute each tool call
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            tool_result = await _execute_tool(tc.function.name, args, db)

            # Check tool name directly — no sentinel strings
            if tc.function.name == "mark_lead_pending":
                signal_node = "LEAD_QUALIFICATION"
                new_pending_action = "store_lead"
                tool_result = "Noted. Bitte frage jetzt nach Name und E-Mail."

            elif tc.function.name == "request_scheduling":
                signal_node = "SCHEDULING"
                tool_result = "Terminbuchungs-Modus aktiviert."

            elif tc.function.name == "store_lead":
                if tool_result.startswith("Lead saved successfully"):
                    new_pending_action = None
                    signal_node = None  # route_after_lead will return "kb"
                # else: keep pending — signal_node stays None

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # Safety: max iterations reached
    fallback = "I'm sorry, I ran into an issue processing your request. Please try again."
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages, signal_node, new_pending_action


async def _execute_tool(name: str, args: dict, db: Any) -> str:
    """Execute a named tool and return its result as a string."""
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

        elif name in ("mark_lead_pending", "request_scheduling"):
            # Handled by caller — return placeholder
            return "ok"

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error: {str(e)}"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def greeting_node(state: ChatState) -> dict:
    """Static greeting — no LLM call."""
    text = (
        f"Hallo! 👋 Willkommen bei {settings.company_name}. "
        "Ich bin Ihr KI-Assistent und helfe Ihnen gerne bei Fragen zu unseren Leistungen – "
        "von Premium Webentwicklung über KI-Chatbots bis hin zu Workflow-Automatisierung. "
        "Was kann ich für Sie tun?"
    )
    return {"response": text, "current_node": "GREETING"}


async def kb_node(state: ChatState) -> dict:
    prompt = get_state_prompt(AgentState.KNOWLEDGE_BASE)
    tools = _tools_for_state(AgentState.KNOWLEDGE_BASE)
    text, msgs, signal, pending = await _run_llm_loop(
        state["messages"], prompt, tools, state["db"],
    )
    new_node = signal or "KNOWLEDGE_BASE"
    return {
        "response": text,
        "messages": msgs,
        "current_node": new_node,
        "pending_action": pending,
    }


async def lead_node(state: ChatState) -> dict:
    prompt = get_state_prompt(AgentState.LEAD_QUALIFICATION)
    tools = _tools_for_state(AgentState.LEAD_QUALIFICATION)
    text, msgs, signal, pending = await _run_llm_loop(
        state["messages"], prompt, tools, state["db"],
        forced_tool=state["pending_action"],
    )
    new_node = signal or "LEAD_QUALIFICATION"
    return {
        "response": text,
        "messages": msgs,
        "current_node": new_node,
        "pending_action": pending,
    }


async def scheduling_node(state: ChatState) -> dict:
    # Ensure Cal.com URL is always current (never baked in at import time)
    url = settings.calcom_booking_url or _CALCOM_FALLBACK
    prompt = get_state_prompt(AgentState.SCHEDULING).replace(
        "Bitte den Besucher, uns per E-Mail zu kontaktieren, um einen Termin zu vereinbaren.",
        f"Teile dem Besucher diesen Buchungslink mit: {url}",
    ) if not settings.calcom_booking_url else get_state_prompt(AgentState.SCHEDULING)

    tools = _tools_for_state(AgentState.SCHEDULING)
    text, msgs, _, pending = await _run_llm_loop(
        state["messages"], prompt, tools, state["db"],
    )
    return {
        "response": text,
        "messages": msgs,
        "current_node": "SCHEDULING",
        "pending_action": pending,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_entry(state: ChatState) -> str:
    node = state["current_node"]
    if node == "GREETING":           return "greeting"
    if node == "LEAD_QUALIFICATION": return "lead"
    if node == "SCHEDULING":         return "scheduling"
    return "kb"  # KNOWLEDGE_BASE or any fallback


def route_after_kb(state: ChatState) -> str:
    node = state["current_node"]
    if node == "LEAD_QUALIFICATION": return "lead"
    if node == "SCHEDULING":         return "scheduling"
    return END


def route_after_lead(state: ChatState) -> str:
    # Lead saved (pending cleared) → offer further help via KB
    if state["pending_action"] is None and state["current_node"] != "LEAD_QUALIFICATION":
        return "kb"
    return END  # still collecting → stay in LEAD_QUALIFICATION


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(ChatState)

    g.add_node("entry",      lambda s: s)   # passthrough router
    g.add_node("greeting",   greeting_node)
    g.add_node("kb",         kb_node)
    g.add_node("lead",       lead_node)
    g.add_node("scheduling", scheduling_node)

    g.set_entry_point("entry")
    g.add_conditional_edges(
        "entry", route_entry,
        {"greeting": "greeting", "kb": "kb", "lead": "lead", "scheduling": "scheduling"},
    )

    g.add_edge("greeting", END)
    g.add_conditional_edges(
        "kb", route_after_kb,
        {"lead": "lead", "scheduling": "scheduling", END: END},
    )
    g.add_conditional_edges(
        "lead", route_after_lead,
        {"kb": "kb", END: END},
    )
    g.add_edge("scheduling", END)

    return g.compile()


chat_graph = _build_graph()   # module-level singleton
