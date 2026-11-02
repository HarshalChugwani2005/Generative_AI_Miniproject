"""
agent.py - Tool (function) calling: the LLM decides which action to take.

The LLM is given three tools. Instead of keyword rules, the model itself
decides when to call them:
  * lookup_order          - live details of an order (status, tracking, ...)
  * cancel_order          - cancel an order that has not shipped yet
  * create_support_ticket - hand the conversation over to a human agent

How it works (see llm.FallbackStream._agent_stream):
  1. The tool descriptions below are sent to the LLM with the question.
  2. The LLM replies either with text (the answer) or with a tool call such
     as {"name": "lookup_order", "args": {"order_id": "SE12345"}}.
  3. AgentSession.run() executes the call and the result is sent back to the
     LLM, which can call another tool or write the final answer.

AgentSession also remembers what happened during one chat turn, so the UI
can show order cards, ticket cards and a "Tools used" line.
"""

import json
import logging

import tools

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tool schemas (exactly what the LLM sees, in the OpenAI "function" format
# that LangChain converts for Gemini, Groq and Ollama)
# --------------------------------------------------------------------------
ORDER_ID_PARAM = {
    "type": "string",
    "description": "The order ID: 'SE' followed by 5 digits, e.g. SE12345.",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Get the live details of a ShopEasy order: status, items, amount, payment "
                "method, courier, tracking number and delivery dates. Call this whenever the "
                "customer asks about a specific order. Never guess order details."
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": ORDER_ID_PARAM},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Cancel a customer's order. Only orders with status 'Placed' or 'Processing' "
                "can be cancelled. Call this ONLY when the customer has clearly asked or "
                "confirmed that they want this specific order cancelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": ORDER_ID_PARAM,
                    "reason": {
                        "type": "string",
                        "description": "Short reason for cancelling, in English, if the customer gave one.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a support ticket so a human agent contacts the customer. Call this "
                "when the customer asks for a human, manager or call-back, or when they agree "
                "to escalate a problem you cannot solve (damaged or missing item, payment "
                "debited but no order, repeated delays)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One-sentence summary of the problem, in English.",
                    },
                    "order_id": {**ORDER_ID_PARAM, "description": "Related order ID, if any."},
                },
                "required": ["reason"],
            },
        },
    },
]


def describe_call(call: dict) -> str:
    """Short progress text shown while a tool runs, e.g. 'Looking up order SE12345'."""
    args = call.get("args") or {}
    order_id = str(args.get("order_id") or "").upper()
    return {
        "lookup_order": f"Looking up order {order_id}",
        "cancel_order": f"Cancelling order {order_id}",
        "create_support_ticket": "Creating a support ticket",
    }.get(call.get("name"), f"Running {call.get('name')}")


# --------------------------------------------------------------------------
# Tool execution for one chat turn
# --------------------------------------------------------------------------
class AgentSession:
    """
    Executes the LLM's tool calls for one user message and records the results.

    After the turn:
      * order_ids - orders that were looked up or cancelled (for order cards)
      * tickets   - tickets created (for ticket cards)
      * actions   - human-readable log, e.g. ["Looked up SE12345"]

    Results are cached per (tool, arguments). If a model fails and the
    fallback model repeats the same call, the order is not cancelled twice and
    no duplicate ticket is created.
    """

    def __init__(self, conversation: list[dict], frustrated: bool = False):
        self.conversation = conversation
        self.frustrated = frustrated
        self.order_ids: list[str] = []
        self.tickets: list[dict] = []
        self.actions: list[str] = []
        self._cache: dict[str, str] = {}

    def run(self, call: dict) -> str:
        """Execute one tool call and return the result text for the LLM."""
        name, args = call.get("name", ""), call.get("args") or {}
        key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if key in self._cache:
            return self._cache[key]
        try:
            result = self._dispatch(name, args)
        except Exception as exc:  # a broken tool must not crash the chat
            logger.exception("Tool %s failed", name, exc_info=exc)
            result = (f"ERROR: the {name} tool failed. Apologise to the customer and offer to "
                      "connect them to a human agent.")
        logger.info("Tool %s(%s) -> %s", name, args, result[:120].replace("\n", " "))
        self._cache[key] = result
        return result

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "lookup_order":
            order_id = str(args.get("order_id", "")).strip().upper()
            order = tools.lookup_order(order_id)
            self._note_order(order_id)
            self.actions.append(f"Looked up {order_id}" + ("" if order else " (not found)"))
            return tools.format_order_for_prompt(order_id, order)

        if name == "cancel_order":
            order_id = str(args.get("order_id", "")).strip().upper()
            outcome = tools.cancel_order(order_id, str(args.get("reason") or ""))
            self._note_order(order_id)
            self.actions.append(f"Cancelled {order_id}" if outcome["success"]
                                else f"Cancel {order_id} refused")
            return outcome["message"]

        if name == "create_support_ticket":
            if self.tickets:  # at most one ticket per message
                t = self.tickets[0]
                return f"Ticket {t['ticket_id']} was already created for this request."
            reason = str(args.get("reason") or "Customer requested a human agent").strip()
            if self.frustrated:
                reason = "Customer appears frustrated: " + reason  # -> High priority
            order_id = str(args.get("order_id") or "").strip().upper() or None
            ticket = tools.create_ticket(reason, self.conversation, order_id)
            self.tickets.append(ticket)
            self.actions.append(f"Created ticket {ticket['ticket_id']}")
            return (f"Support ticket {ticket['ticket_id']} created with {ticket['priority']} "
                    f"priority. A human agent will contact the customer {ticket['eta']}.")

        return f"ERROR: unknown tool '{name}'."

    def _note_order(self, order_id: str) -> None:
        if order_id and order_id not in self.order_ids:
            self.order_ids.append(order_id)
