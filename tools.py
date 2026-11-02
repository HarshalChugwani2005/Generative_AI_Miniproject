"""
tools.py - Helper "tools" used by the chatbot alongside the LLM.

  * Order lookup     - finds order IDs such as SE12345 in a message and reads
                       their details from data/orders.json (mock database).
  * Order cancel     - cancels orders that have not shipped yet.
  * Sentiment check  - detects angry / frustrated messages (keyword rules,
                       optionally confirmed by the LLM).
  * Human handoff    - detects requests for a human agent.
  * Ticket creation  - creates a support ticket and stores it in
                       data/tickets.json (via storage.py).
"""

import json
import logging
import re
import uuid
from datetime import datetime
from functools import lru_cache

import config
import storage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Order lookup
# --------------------------------------------------------------------------
# "SE" followed by exactly 5 digits, case-insensitive (se12345 also matches).
ORDER_ID_PATTERN = re.compile(r"\bSE\d{5}\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_orders() -> dict[str, dict]:
    """Load the mock order database once and index it by order ID."""
    try:
        with open(config.ORDERS_PATH, encoding="utf-8") as f:
            orders = json.load(f)
    except FileNotFoundError:
        logger.error("Orders file not found at %s", config.ORDERS_PATH)
        return {}
    except json.JSONDecodeError as exc:
        logger.error("Orders file is not valid JSON: %s", exc)
        return {}
    return {order["order_id"].upper(): order for order in orders}


def extract_order_ids(text: str) -> list[str]:
    """Return all unique order IDs mentioned in the text, upper-cased, in order."""
    seen: list[str] = []
    for match in ORDER_ID_PATTERN.findall(text):
        oid = match.upper()
        if oid not in seen:
            seen.append(oid)
    return seen


def lookup_order(order_id: str) -> dict | None:
    """
    Return the order record for an ID, or None if it doesn't exist.

    Changes made by the chatbot (data/order_updates.json, e.g. a cancellation)
    are applied on top of the original record.
    """
    order_id = order_id.strip().upper()
    order = load_orders().get(order_id)
    if order is None:
        return None
    return {**order, **storage.load_order_updates().get(order_id, {})}


# --------------------------------------------------------------------------
# Order cancellation
# --------------------------------------------------------------------------
# Same rule as the FAQ: orders can be cancelled only before they ship.
CANCELLABLE_STATUSES = ("Placed", "Processing")


def cancel_order(order_id: str, reason: str = "") -> dict:
    """
    Cancel an order if it has not shipped yet.

    Returns {"success": bool, "message": str}. The message is written for the
    LLM, which explains the result to the customer.
    """
    order_id = order_id.strip().upper()
    order = lookup_order(order_id)
    if order is None:
        return {"success": False, "message": (
            f"Order {order_id} was NOT FOUND. Nothing was cancelled. Ask the customer to "
            "re-check the order ID (SE followed by 5 digits).")}

    status = order["status"]
    if status == "Cancelled":
        return {"success": False, "message": f"Order {order_id} is already cancelled."}
    if status in ("Shipped", "Out for Delivery"):
        return {"success": False, "message": (
            f"Order {order_id} cannot be cancelled because its status is '{status}'. "
            "The customer can refuse the delivery, or return the item after delivery.")}
    if status not in CANCELLABLE_STATUSES:
        return {"success": False, "message": (
            f"Order {order_id} cannot be cancelled because its status is '{status}'. "
            "If it was delivered, the customer can request a return under the return policy.")}

    reason = reason.strip() or "Not given"
    saved = storage.save_order_update(order_id, {
        "status": "Cancelled",
        "expected_delivery": None,
        "notes": f"Cancelled by the customer via chat on {datetime.now():%d %b %Y}. Reason: {reason}.",
    })
    if not saved:
        return {"success": False, "message": (
            f"A system error stopped order {order_id} from being cancelled. Apologise and "
            "offer to create a support ticket.")}

    if order["payment_method"] == "Cash on Delivery":
        refund = "It was Cash on Delivery, so no payment was taken and no refund is needed."
    else:
        refund = (f"A full refund of ₹{order['total_amount']:,} will go back to the original "
                  f"payment method ({order['payment_method']}) as per the refund timelines.")
    logger.info("Cancelled order %s (reason: %s)", order_id, reason)
    return {"success": True, "message": f"Order {order_id} has been CANCELLED. {refund}"}


def format_order_for_prompt(order_id: str, order: dict | None) -> str:
    """
    Turn an order lookup result into plain text for the LLM prompt.

    The LLM is instructed to use ONLY this data for order questions, so it
    cannot invent statuses or dates.
    """
    if order is None:
        return (
            f"ORDER LOOKUP for {order_id}: NOT FOUND. No order with this ID exists in "
            "our system. Ask the customer to double-check the order ID (format: SE "
            "followed by 5 digits) or find it under My Account > My Orders."
        )

    items = "; ".join(
        f"{item['name']} x{item['quantity']} (₹{item['price']:,} each)" for item in order["items"]
    )
    lines = [
        f"ORDER LOOKUP for {order['order_id']}: FOUND",
        f"- Customer name: {order['customer_name']}",
        f"- Order date: {order['order_date']}",
        f"- Current status: {order['status']}",
        f"- Items: {items}",
        f"- Total amount: ₹{order['total_amount']:,}",
        f"- Payment method: {order['payment_method']}",
        f"- Delivery city: {order['delivery_city']}",
        f"- Courier: {order['courier'] or 'Not assigned yet'}",
        f"- Tracking number: {order['tracking_number'] or 'Not available yet'}",
        f"- Expected delivery: {order['expected_delivery'] or 'Not applicable'}",
        f"- Delivered on: {order['delivered_on'] or 'Not delivered yet'}",
        f"- Notes: {order['notes']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Sentiment check
# --------------------------------------------------------------------------
# Words/phrases that usually signal anger or frustration.
FRUSTRATION_KEYWORDS = [
    "angry", "furious", "frustrated", "frustrating", "annoyed", "annoying",
    "upset", "disappointed", "disappointing", "unacceptable", "ridiculous",
    "pathetic", "terrible", "horrible", "worst", "awful", "useless", "rubbish",
    "disgusting", "fed up", "sick of", "tired of", "fraud", "scam", "cheated",
    "cheat", "liar", "lied", "waste of", "never again", "hate", "complaint",
    "complain", "consumer court", "legal action", "sue", "nonsense",
    "how many times", "still not", "still haven't", "still havent",
    "no one is", "nobody is", "not happy", "very bad", "so bad",
    "what the hell", "wtf", "damn", "stupid", "incompetent",
    # Common Hinglish (Hindi typed in English letters)
    "bakwas", "bekaar", "bekar", "ghatiya", "faltu", "pareshan",
]
_KEYWORD_PATTERNS = [
    re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in FRUSTRATION_KEYWORDS
]

# Phrases that mean the user explicitly wants a person.
HUMAN_REQUEST_PATTERN = re.compile(
    r"\b(human|real person|live agent|support agent|representative|customer care|"
    r"talk to (someone|somebody|a person)|speak to (someone|somebody|a person)|"
    r"escalate|manager|supervisor)\b",
    re.IGNORECASE,
)


def detect_sentiment_keywords(text: str) -> dict:
    """
    Simple, fast, offline frustration detector.

    Scores one point per matched keyword, plus extra points for "shouting"
    (lots of capital letters) and repeated exclamation/question marks.
    """
    matched = [kw for kw, pat in zip(FRUSTRATION_KEYWORDS, _KEYWORD_PATTERNS) if pat.search(text)]
    score = len(matched)

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        score += 1
        matched.append("ALL CAPS")
    if re.search(r"[!?]{3,}", text):
        score += 1
        matched.append("!!!")

    return {
        "label": "frustrated" if score >= 1 else "neutral",
        "score": score,
        "matched": matched,
        "method": "keyword",
    }


def detect_sentiment_llm(text: str, llm) -> str | None:
    """
    Ask the LLM to classify the message. Returns "frustrated", "neutral",
    or None if the call fails (the caller then falls back to keywords).
    """
    prompt = (
        "Classify the emotional tone of this customer support message. "
        "Reply with exactly one word: FRUSTRATED (angry, upset, annoyed, "
        "complaining) or NEUTRAL (anything else).\n\n"
        f"Message: {text}"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):  # some providers return content parts
            content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        answer = str(content).strip().upper()
        if "FRUSTRATED" in answer:
            return "frustrated"
        if "NEUTRAL" in answer:
            return "neutral"
    except Exception as exc:  # network/rate-limit issues must not break the chat
        logger.warning("LLM sentiment classification failed: %s", exc)
    return None


def detect_sentiment(text: str, llm=None, mode: str = config.SENTIMENT_MODE) -> dict:
    """
    Combined sentiment check.

    mode="keyword": keyword rules only (default, no extra API call).
    mode="llm":     keyword rules, and if they find nothing, ask the LLM too.
    """
    result = detect_sentiment_keywords(text)
    if mode == "llm" and llm is not None and result["label"] == "neutral":
        llm_label = detect_sentiment_llm(text, llm)
        if llm_label == "frustrated":
            result.update(label="frustrated", method="llm")
    return result


def wants_human_agent(text: str) -> bool:
    """True if the user is explicitly asking for a human / escalation."""
    return bool(HUMAN_REQUEST_PATTERN.search(text))


# --------------------------------------------------------------------------
# Support tickets
# --------------------------------------------------------------------------
def create_ticket(
    reason: str,
    conversation: list[dict] | None = None,
    order_id: str | None = None,
) -> dict:
    """
    Create a support ticket, save it to data/tickets.json and return it.

    Ticket ID format: TKT-YYYYMMDD-XXXXXX (e.g. TKT-20261102-A1B2C3).
    Frustrated customers get "High" priority and a faster target response time.
    """
    now = datetime.now()
    ticket_id = f"TKT-{now:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    high_priority = "frustrat" in reason.lower()

    # Keep the last few real chat messages so a human agent has context.
    transcript = [
        {"role": m["role"], "content": m["content"]}
        for m in (conversation or [])
        if not m.get("is_error") and m.get("kind", "text") == "text"
    ][-config.MEMORY_WINDOW:]

    ticket = {
        "ticket_id": ticket_id,
        "created_at": now.isoformat(timespec="seconds"),
        "status": "Open",
        "priority": "High" if high_priority else "Normal",
        "eta": "within 4 hours" if high_priority else config.HUMAN_RESPONSE_TIME,
        "reason": reason,
        "order_id": order_id,
        "transcript": transcript,
    }
    if not storage.append_ticket(ticket):
        # The ticket is still returned so the user isn't blocked.
        logger.error("Ticket %s could not be saved to disk", ticket_id)
    logger.info("Created support ticket %s", ticket_id)
    return ticket


if __name__ == "__main__":
    # Quick manual test:  python tools.py
    print(extract_order_ids("Where is my order se12345 and SE99999?"))
    print(format_order_for_prompt("SE12345", lookup_order("SE12345")))
    print(detect_sentiment("This is RIDICULOUS, my order is still not here!!!"))
    print(detect_sentiment("How do I apply a coupon?"))
    print(wants_human_agent("Can I talk to a human please"))
