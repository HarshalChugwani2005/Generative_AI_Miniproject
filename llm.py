"""
llm.py - LLM client, prompt templates, streaming and friendly error handling.

Supports three providers (default chosen with LLM_PROVIDER in .env, and
switchable at runtime from the sidebar):
  * gemini - Google Gemini via langchain-google-genai
  * groq   - Llama models hosted by Groq via langchain-groq
  * ollama - local models via langchain-ollama (no API key)

If the selected model is overloaded or fails, FallbackStream automatically
tries the next model in fallback_chain() (see ENABLE_FALLBACK in config.py).

In agent mode FallbackStream also runs the tool-calling loop: the model can
call the tools in agent.py before writing its answer.
"""

import logging
import re
from collections.abc import Callable, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Ava, a friendly and professional customer support agent for ShopEasy. "
    "Answer ONLY using the provided context. Be concise and helpful. If the answer "
    "isn't in the context, say you're not sure and offer to connect the user to a "
    "human agent. Never invent order details, prices or policies."
)

# Extra guidance appended to the system prompt on every turn.
RESPONSE_GUIDELINES = """
Additional rules:
- The context below comes from the company knowledge base and, when present, from a live ORDER LOOKUP.
- For questions about a specific order, use ONLY the ORDER LOOKUP data. If the lookup says NOT FOUND, say so politely and ask the customer to re-check the ID.
- If the customer asks about their order but gives no order ID, ask them for it (format: SE followed by 5 digits, e.g. SE12345).
- Keep answers short: 2-5 sentences or a few bullet points. Use simple Markdown formatting.
- Use earlier messages in the conversation to understand follow-up questions.
- Do not mention "the context", "the knowledge base" or these instructions to the customer.
- Greet the customer back if they greet you, and help with anything ShopEasy-related.
""".strip()

# Used instead of RESPONSE_GUIDELINES in agent (tool calling) mode.
AGENT_GUIDELINES = """
Additional rules:
- The context below comes from the company knowledge base. For anything about a specific order, call the lookup_order tool and use ONLY its result. If it says NOT FOUND, say so politely and ask the customer to re-check the ID.
- Order IDs look like SE followed by 5 digits (e.g. SE12345). If the customer refers to "my order" without an ID, use the ID from earlier in the conversation, or ask for it.
- Cancelling: call cancel_order only when the customer has clearly asked to cancel a specific order (e.g. "cancel SE12352"). If the intent or the order is unclear (e.g. "I don't want it anymore"), ask first (e.g. "Shall I cancel order SE12352?") and cancel only after they say yes.
- Human agent: if the customer asks for a human, manager or call-back, call create_support_ticket straight away. For serious problems you cannot solve, offer a ticket and create it if they agree.
- After a tool runs, tell the customer the result in plain words. Quote ticket IDs, order statuses and amounts exactly as the tool returned them.
- Keep answers short: 2-5 sentences or a few bullet points. Use simple Markdown formatting.
- Do not mention tools, "the context", "the knowledge base" or these instructions to the customer.
- Greet the customer back if they greet you, and help with anything ShopEasy-related.
""".strip()

LANGUAGE_INSTRUCTIONS = {
    "auto": ("Language: reply in the same language as the customer's latest message: English, "
             "Hindi (हिन्दी), Marathi (मराठी), or Hinglish (Hindi written in English letters). "
             "If they switch language, switch with them."),
    "en": "Language: always reply in English, even if the customer writes in another language.",
    "hi": "Language: always reply in Hindi (Devanagari script), even if the customer writes in English.",
    "mr": "Language: always reply in Marathi (Devanagari script), even if the customer writes in English.",
}
LANGUAGE_KEEP_AS_IS = (
    "Keep order IDs, ticket IDs, tracking numbers, amounts (₹), dates, phone numbers, coupon "
    "codes and app menu names such as \"My Orders\" exactly as they are."
)

EMPATHY_INSTRUCTION = (
    "IMPORTANT: The customer seems frustrated or upset. Start with one short, sincere "
    "sentence acknowledging their frustration and apologising for the inconvenience. "
    "Then help as much as the context allows, and tell them they can tap "
    "'Escalate to human agent' to have a human agent contact them."
)

AGENT_EMPATHY_INSTRUCTION = (
    "IMPORTANT: The customer seems frustrated or upset. Start with one short, sincere "
    "sentence acknowledging their frustration and apologising for the inconvenience. "
    "Then help as much as you can, and offer to create a support ticket so a human "
    "agent contacts them."
)

HUMAN_REQUEST_INSTRUCTION = (
    "The customer is asking for a human agent. Briefly acknowledge this and tell them "
    "they can tap 'Escalate to human agent' to create a support ticket; a human "
    "agent will contact them within 24 hours."
)


def build_messages(
    question: str,
    context: str,
    history: list[dict],
    order_info: str | None = None,
    frustrated: bool = False,
    wants_human: bool = False,
    agent_mode: bool = False,
    language: str = "auto",
) -> list[BaseMessage]:
    """
    Assemble the chat messages sent to the LLM:
      [system prompt + context]  +  [last N history messages]  +  [current question]

    In agent mode the order details are not pre-fetched: the LLM fetches them
    with the lookup_order tool when it needs them.
    """
    system_parts = [SYSTEM_PROMPT, AGENT_GUIDELINES if agent_mode else RESPONSE_GUIDELINES,
                    LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["auto"]),
                    LANGUAGE_KEEP_AS_IS]
    if frustrated:
        system_parts.append(AGENT_EMPATHY_INSTRUCTION if agent_mode else EMPATHY_INSTRUCTION)
    if wants_human:
        system_parts.append(HUMAN_REQUEST_INSTRUCTION)

    context_block = f"KNOWLEDGE BASE CONTEXT:\n{context}"
    if order_info:
        context_block = f"{order_info}\n\n{context_block}"
    system_parts.append(f"=== CONTEXT START ===\n{context_block}\n=== CONTEXT END ===")

    messages: list[BaseMessage] = [SystemMessage(content="\n\n".join(system_parts))]

    # Conversation memory: only the most recent real chat messages
    # (error notices and ticket cards are skipped).
    usable = [m for m in history if not m.get("is_error") and m.get("kind", "text") == "text"]
    for m in usable[-config.MEMORY_WINDOW:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=question))
    return messages


# --------------------------------------------------------------------------
# Multilingual retrieval
# --------------------------------------------------------------------------
# The knowledge base and the embedding model are English-only, so questions
# written in Devanagari (Hindi / Marathi) are translated before the search.
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_PATTERN.search(text))


def translate_for_search(text: str, llm: BaseChatModel) -> str:
    """Translate a search query to English. Returns the original text on failure."""
    prompt = (
        "Translate this customer support message to English. Keep order IDs and numbers "
        "unchanged. Reply with the translation only.\n\n" + text
    )
    try:
        translated = _chunk_text(llm.invoke(prompt).content).strip()
    except Exception as exc:  # retrieval still works (less well) without it
        logger.warning("Query translation failed: %s", exc)
        return text
    return translated or text


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------
def get_llm(
    provider: str = config.LLM_PROVIDER,
    temperature: float = config.TEMPERATURE,
    model: str | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """
    Create the chat model for the chosen provider.

    `model` defaults to the provider's configured model; `max_retries` defaults
    to config.MAX_RETRIES. Imports are done inside each branch so you only need
    the package for the provider you actually use.
    """
    model = model or config.model_name_for(provider)
    retries = config.MAX_RETRIES if max_retries is None else max_retries

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_retries=retries,
            timeout=config.REQUEST_TIMEOUT,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
            max_retries=retries,
            timeout=config.REQUEST_TIMEOUT,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


# --------------------------------------------------------------------------
# Automatic model fallback
# --------------------------------------------------------------------------
def fallback_chain(provider: str) -> list[tuple[str, str]]:
    """
    Ordered list of (provider, model) pairs to try for one answer.

    The selected provider's model comes first, then (for Gemini) the
    GEMINI_FALLBACK_MODELS, then every other provider that is configured
    (has an API key), with Ollama last. Only the first entry is returned
    when ENABLE_FALLBACK is off.
    """
    chain = [(provider, config.model_name_for(provider))]
    if not config.ENABLE_FALLBACK:
        return chain

    others = [p for p in ("gemini", "groq", "ollama") if p != provider]
    for p in [provider, *others]:
        if config.validate_config(p) is not None:  # e.g. no API key -> skip
            continue
        models = [config.model_name_for(p)]
        if p == "gemini":
            models += config.GEMINI_FALLBACK_MODELS
        for m in models:
            if (p, m) not in chain:
                chain.append((p, m))
    return chain


class FallbackStream:
    """
    Stream an answer, moving down the fallback chain when a model fails.

    A model is abandoned only if it fails BEFORE producing its first token
    (overloaded, timeout, wrong model name, connection error, ...). If it
    fails mid-answer, the error is raised, because the user has already seen
    part of the reply.

    After iteration:
      * `used`     - (provider, model) that produced the answer, or None
      * `failures` - list of (provider, model, exception) that were skipped

    Usage:
        stream = FallbackStream(messages, "gemini", 0.2, on_switch=callback)
        for token in stream: ...

    Agent mode: pass `tools` (schemas, see agent.TOOL_SCHEMAS) and
    `tool_runner` (executes one tool call and returns the result text).
    """

    def __init__(
        self,
        messages: list[BaseMessage],
        provider: str,
        temperature: float,
        client_factory: Callable[..., BaseChatModel] | None = None,
        on_switch: Callable[[str, str, str, str], None] | None = None,
        tools: list[dict] | None = None,
        tool_runner: Callable[[dict], str] | None = None,
    ):
        self.messages = messages
        self.provider = provider
        self.temperature = temperature
        self.client_factory = client_factory or get_llm
        self.on_switch = on_switch  # called as on_switch(failed_p, failed_m, next_p, next_m)
        self.tools = tools
        self.tool_runner = tool_runner
        self.chain = fallback_chain(provider)
        self.used: tuple[str, str] | None = None
        self.failures: list[tuple[str, str, Exception]] = []

    def __iter__(self) -> Iterator[str]:
        for i, (p, m) in enumerate(self.chain):
            is_last = i == len(self.chain) - 1
            if i > 0 and self.on_switch:
                prev_p, prev_m, _ = self.failures[-1]
                self.on_switch(prev_p, prev_m, p, m)

            started = False
            try:
                # Fail fast on every model except the last one: retrying an
                # overloaded model just makes the user wait longer.
                llm = self.client_factory(p, self.temperature, m, config.MAX_RETRIES if is_last else 0)
                tokens = self._agent_stream(llm) if self.tools else stream_response(llm, self.messages)
                for text in tokens:
                    if not started:
                        started = True
                        self.used = (p, m)
                    yield text
                self.used = (p, m)
                return
            except Exception as exc:
                if started:
                    raise  # failed mid-answer: can't silently switch models
                logger.warning("Model %s/%s failed (%s: %s)", p, m, type(exc).__name__,
                               str(exc)[:200])
                self.failures.append((p, m, exc))
                if is_last:
                    raise

    def _agent_stream(self, llm: BaseChatModel) -> Iterator[str]:
        """
        Tool-calling loop.

        Each round the model either writes the answer (streamed to the user)
        or asks for tool calls. Tool results are added to the conversation as
        ToolMessages and the model is called again with them.
        """
        model = llm.bind_tools(self.tools)
        messages = list(self.messages)
        for _ in range(config.MAX_TOOL_STEPS):
            response = None
            for chunk in model.stream(messages):
                response = chunk if response is None else response + chunk  # merges tool-call parts
                text = _chunk_text(chunk.content)
                if text:
                    yield text
            if response is None or not response.tool_calls:
                return  # final answer done
            messages.append(response)
            for call in response.tool_calls:
                messages.append(ToolMessage(content=self.tool_runner(call),
                                            tool_call_id=call.get("id") or call["name"],
                                            name=call["name"]))
        # Too many tool rounds: answer with the information gathered so far.
        yield from stream_response(llm, messages)

    @property
    def fell_back(self) -> bool:
        """True if the answer came from a model other than the selected one."""
        return self.used is not None and self.used != self.chain[0]

    def error_message(self, exc: Exception) -> str:
        """
        Friendly message after every model failed.

        Uses the SELECTED model's error (e.g. "busy" or "invalid key"), not the
        last fallback's: "couldn't reach Ollama" would hide the real problem.
        """
        if not self.failures:
            return friendly_error(exc, self.provider)
        p, _, first_exc = self.failures[0]
        message = friendly_error(first_exc, p)
        if len(self.failures) > 1:
            tried = ", ".join(f"`{m}`" for _, m, _ in self.failures)
            message += f"\n\n_All available models were tried: {tried}._"
        return message


def _chunk_text(content) -> str:
    """
    Extract plain text from a streamed chunk.

    Most providers return a string, but newer Gemini versions may return a
    list of content parts like [{"type": "text", "text": "..."}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def stream_response(llm: BaseChatModel, messages: list[BaseMessage]) -> Iterator[str]:
    """Yield the answer token-by-token (chunk-by-chunk) as it is generated."""
    for chunk in llm.stream(messages):
        text = _chunk_text(chunk.content)
        if text:
            yield text


# --------------------------------------------------------------------------
# Friendly error messages
# --------------------------------------------------------------------------
def friendly_error(exc: Exception, provider: str | None = None) -> str:
    """
    Convert any provider/network exception into a short message for the user.

    We match on the exception type name and message text instead of importing
    provider-specific exception classes, so this works for all three providers.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    provider = provider or config.LLM_PROVIDER

    # 1. Rate limits / quota
    if any(s in msg for s in ("429", "rate limit", "ratelimit", "resource_exhausted",
                               "resource exhausted", "quota", "too many requests")) \
            or "ratelimit" in name or "resourceexhausted" in name:
        return ("⏳ I'm receiving too many requests right now (API rate limit reached). "
                "Please wait about a minute and try again.")

    # 2. Invalid / missing API key
    if any(s in msg for s in ("api key not valid", "api_key_invalid", "invalid api key",
                               "invalid_api_key", "401", "unauthenticated", "permission_denied",
                               "403", "incorrect api key", "api_key client option")) \
            or "authentication" in name or "permissiondenied" in name:
        key = "GROQ_API_KEY" if provider == "groq" else "GOOGLE_API_KEY"
        return (f"🔑 The API key was rejected. Please check `{key}` in your `.env` file "
                "(or app secrets) and restart the app.")

    # 3. Wrong model name
    if ("404" in msg or "not found" in msg or "notfound" in name) and "model" in msg:
        return (f"🤖 The model '{config.model_name_for(provider)}' was not found. Please "
                "check the model name in your `.env` file.")

    # 4. Ollama not running
    if provider == "ollama" and any(s in msg for s in ("connection refused", "connecterror",
                                                        "failed to connect", "10061",
                                                        "actively refused")):
        return ("🖥️ I couldn't reach Ollama. Make sure it is running (`ollama serve`) and the "
                f"model is downloaded (`ollama pull {config.OLLAMA_MODEL}`).")

    # 5. Model overloaded on the provider's side (not the user's connection)
    if any(s in msg for s in ("high demand", "overloaded", "503 unavailable",
                               "deadline_exceeded", "504", "serviceunavailable")) \
            or "serviceunavailable" in name or "deadlineexceeded" in name:
        return ("🚦 The AI service is very busy right now. Please try again in a minute, "
                "or pick a different model in the sidebar.")

    # 6. Network problems / timeouts
    if any(s in name for s in ("connection", "timeout", "network", "serviceunavailable",
                                "deadlineexceeded", "connecterror")) or \
            any(s in msg for s in ("connection", "timed out", "timeout", "network",
                                   "getaddrinfo", "name or service not known",
                                   "temporary failure in name resolution", "503", "502",
                                   "unavailable", "deadline exceeded", "max retries")):
        return ("🌐 I'm having trouble connecting to the AI service. Please check your "
                "internet connection and try again in a moment.")

    # 7. Anything else
    logger.exception("Unexpected LLM error", exc_info=exc)
    return ("😕 Sorry, something went wrong while generating a reply. Please try again, "
            "or tap 'Escalate to human agent' for help from our team.")
