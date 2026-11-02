"""
config.py - Central settings for the ShopEasy support chatbot.

Values are read (in this order of priority) from:
  1. Real environment variables (used by Hugging Face Spaces secrets)
  2. The `.env` file in the project folder (local development)
  3. Streamlit secrets (`.streamlit/secrets.toml` or Streamlit Community Cloud)
  4. The defaults below

Nothing secret is hard-coded here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths (built with pathlib so they work on both Windows and Linux)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.md"
ORDERS_PATH = DATA_DIR / "orders.json"
TICKETS_PATH = DATA_DIR / "tickets.json"      # created on first escalation
FEEDBACK_PATH = DATA_DIR / "feedback.json"    # created on first 👍/👎
CHAT_LOG_PATH = DATA_DIR / "chat_log.jsonl"   # one line per user question (for analytics)
ORDER_UPDATES_PATH = DATA_DIR / "order_updates.json"  # changes made by tools (e.g. cancellations)
VECTORSTORE_DIR = BASE_DIR / "vectorstore"    # FAISS index cache

# Load .env from the project folder (override=False: real env vars win).
load_dotenv(BASE_DIR / ".env", override=False)

# Quieten noisy-but-harmless library warnings (esp. on Windows).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_secret(name: str, default: str = "") -> str:
    """
    Read a setting from the environment, falling back to Streamlit secrets.

    Streamlit secrets are only available when running inside Streamlit, and
    accessing them without a secrets.toml raises an error, so every failure
    simply falls back to the default.
    """
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


def _get_float(name: str, default: float) -> float:
    """Read a float setting, falling back to `default` if missing or invalid."""
    try:
        return float(_get_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    """Read an int setting, falling back to `default` if missing or invalid."""
    try:
        return int(_get_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# LLM provider selection: "gemini" (default), "groq" or "ollama"
# The provider can also be changed at runtime from the sidebar.
# --------------------------------------------------------------------------
LLM_PROVIDER = _get_secret("LLM_PROVIDER", "gemini").lower()
SUPPORTED_PROVIDERS = ("gemini", "groq", "ollama")
PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "groq": "Groq (Llama)",
    "ollama": "Ollama (local)",
}

# Google Gemini
GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")
GEMINI_MODEL = _get_secret("GEMINI_MODEL", "gemini-3.5-flash")

# Groq (hosted Llama models)
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_MODEL = _get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

# Ollama (local models, no API key needed)
OLLAMA_MODEL = _get_secret("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = _get_secret("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation settings
TEMPERATURE = _get_float("TEMPERATURE", 0.2)       # low = factual, consistent answers
REQUEST_TIMEOUT = _get_int("REQUEST_TIMEOUT", 30)  # seconds per model attempt
MAX_RETRIES = _get_int("MAX_RETRIES", 1)           # retries on the LAST model in the fallback chain

# Automatic model fallback: if the selected model is overloaded, times out or
# fails, the next model in the chain is tried automatically:
#   selected model -> GEMINI_FALLBACK_MODELS -> other configured providers -> Ollama
ENABLE_FALLBACK = _get_secret("ENABLE_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in _get_secret("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",")
    if m.strip()
]

# --------------------------------------------------------------------------
# RAG settings
# --------------------------------------------------------------------------
EMBEDDING_MODEL = _get_secret("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# --------------------------------------------------------------------------
# Conversation / tools
# --------------------------------------------------------------------------
MEMORY_WINDOW = 6  # number of previous chat messages sent to the LLM

# Agent mode: the LLM decides itself when to call lookup_order, cancel_order
# and create_support_ticket (tool / function calling). When off, the older
# keyword rules (regex for order IDs, "human" keywords) are used instead.
# Can also be toggled at runtime from the sidebar.
USE_TOOL_CALLING = _get_secret("USE_TOOL_CALLING", "true").lower() in ("1", "true", "yes", "on")
MAX_TOOL_STEPS = 4  # max rounds of tool calls per answer (stops endless loops)

# Reply language: "auto" = answer in the customer's language, or force one.
REPLY_LANGUAGE = _get_secret("REPLY_LANGUAGE", "auto").lower()
LANGUAGES = {
    "auto": "Auto (match customer)",
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
}

# "keyword" = fast, offline keyword rules (default)
# "llm"     = keyword rules + an extra LLM classification call
SENTIMENT_MODE = _get_secret("SENTIMENT_MODE", "keyword").lower()

# Optional password for the Admin Dashboard page (empty = no password).
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD")

BOT_NAME = "Ava"
COMPANY_NAME = "ShopEasy"
HUMAN_RESPONSE_TIME = "within 24 hours"
SUPPORT_PHONE = "1800-123-4567"


def model_name_for(provider: str) -> str:
    """Return the configured model name for a provider."""
    return {
        "gemini": GEMINI_MODEL,
        "groq": GROQ_MODEL,
        "ollama": OLLAMA_MODEL,
    }.get(provider, "unknown")


def active_model_name() -> str:
    """Return the model name for the default provider (kept for compatibility)."""
    return model_name_for(LLM_PROVIDER)


def _is_missing(key: str) -> bool:
    """True if a key is empty or still the placeholder text from .env.example."""
    return not key or key.lower().startswith("your_")


def validate_config(provider: str | None = None) -> str | None:
    """
    Check that the selected provider is usable.

    Returns None when everything looks fine, otherwise a friendly,
    human-readable message explaining how to fix the problem.
    """
    provider = (provider or LLM_PROVIDER).lower()
    if provider not in SUPPORTED_PROVIDERS:
        return (
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Set LLM_PROVIDER in your .env file to one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    if provider == "gemini" and _is_missing(GOOGLE_API_KEY):
        return (
            "No Google Gemini API key found. Create a free key at "
            "https://aistudio.google.com/app/apikey, then add the line "
            "`GOOGLE_API_KEY=your_key_here` to the `.env` file in the project "
            "folder (or to your Streamlit / Hugging Face secrets) and restart the app."
        )
    if provider == "groq" and _is_missing(GROQ_API_KEY):
        return (
            "No Groq API key found. Create a free key at https://console.groq.com/keys, "
            "then add `GROQ_API_KEY=your_key_here` to the `.env` file (or your app "
            "secrets) and restart the app."
        )
    return None
