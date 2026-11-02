"""
storage.py - Tiny JSON "database" used by the chatbot and the Admin Dashboard.

Files (all in data/, created automatically):
  * feedback.json   - 👍/👎 ratings on bot replies
  * tickets.json    - escalated support tickets
  * chat_log.jsonl  - one JSON line per user question (for analytics)
  * order_updates.json - order changes made by the chatbot (e.g. cancellations);
                         data/orders.json itself is never modified

JSON files are fine for a demo / college project. A real product would use a
database such as PostgreSQL or Firestore instead.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Streamlit runs each browser session in its own thread, so file writes are
# protected with a lock to avoid two sessions writing at the same time.
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def read_json_list(path: Path) -> list:
    """Read a JSON array from disk. Missing or corrupt files return []."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return []


def write_json_list(path: Path, items: list) -> bool:
    """Write a JSON array to disk. Returns False (and logs) on failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic on Windows and Linux
        return True
    except OSError as exc:
        logger.error("Could not write %s: %s", path.name, exc)
        return False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------
def save_feedback(
    message_id: str,
    rating: str | None,
    question: str,
    answer: str,
    provider: str,
    model: str,
) -> None:
    """
    Save or update the rating ("up" / "down") for one bot reply.

    If the user changes their mind, the existing entry is updated; if they
    clear the rating (rating=None) the entry is removed.
    """
    with _lock:
        items = [f for f in read_json_list(config.FEEDBACK_PATH) if f.get("message_id") != message_id]
        if rating is not None:
            items.append(
                {
                    "message_id": message_id,
                    "rating": rating,
                    "question": question,
                    "answer": answer,
                    "provider": provider,
                    "model": model,
                    "timestamp": now_iso(),
                }
            )
        write_json_list(config.FEEDBACK_PATH, items)


def load_feedback() -> list[dict]:
    return read_json_list(config.FEEDBACK_PATH)


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------
def append_ticket(ticket: dict) -> bool:
    with _lock:
        items = read_json_list(config.TICKETS_PATH)
        items.append(ticket)
        return write_json_list(config.TICKETS_PATH, items)


def load_tickets() -> list[dict]:
    return read_json_list(config.TICKETS_PATH)


# --------------------------------------------------------------------------
# Order updates (applied on top of the mock data/orders.json)
# --------------------------------------------------------------------------
def save_order_update(order_id: str, changes: dict) -> bool:
    """Record a change to an order, e.g. {"status": "Cancelled"}."""
    with _lock:
        items = read_json_list(config.ORDER_UPDATES_PATH)
        items.append({"order_id": order_id, "updated_at": now_iso(), **changes})
        return write_json_list(config.ORDER_UPDATES_PATH, items)


def load_order_updates() -> dict[str, dict]:
    """Return {order_id: merged changes}; later updates win."""
    merged: dict[str, dict] = {}
    for update in read_json_list(config.ORDER_UPDATES_PATH):
        changes = {k: v for k, v in update.items() if k not in ("order_id", "updated_at")}
        merged.setdefault(update.get("order_id", ""), {}).update(changes)
    return merged


# --------------------------------------------------------------------------
# Chat log (for "most common questions")
# --------------------------------------------------------------------------
def log_question(entry: dict) -> None:
    """Append one question record as a JSON line (cheap, append-only)."""
    entry = {"timestamp": now_iso(), **entry}
    with _lock:
        try:
            config.CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(config.CHAT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Could not write chat log: %s", exc)


def load_chat_log() -> list[dict]:
    if not config.CHAT_LOG_PATH.exists():
        return []
    rows = []
    try:
        with open(config.CHAT_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a damaged line instead of failing
    except OSError as exc:
        logger.warning("Could not read chat log: %s", exc)
    return rows
