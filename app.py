"""
app.py - Streamlit user interface for "Ava", the ShopEasy support chatbot.

Run with:   streamlit run app.py

Each user message goes through this pipeline:
  1. Order tool      - keyword mode only: look up order IDs (e.g. SE12345) found by regex
  2. Sentiment check - detect frustration (and, in keyword mode, requests for a human)
  3. Retrieval       - fetch the top-3 most relevant chunks from FAISS
                       (Hindi / Marathi questions are translated to English first)
  4. Prompt          - system prompt + context + last 6 messages + question
  5. LLM             - stream the answer token by token into the chat bubble.
                       In agent mode the LLM may first call lookup_order,
                       cancel_order or create_support_ticket (see agent.py)
  6. Escalation      - offer an "Escalate to human agent" card if needed
  7. Analytics       - log the question; 👍/👎 ratings go to data/feedback.json

The look & feel (CSS, header, cards) lives in ui.py. The Admin Dashboard is
in pages/Admin_Dashboard.py; both pages are registered with st.navigation
at the bottom of this file.
"""

import logging
import time
import uuid
from datetime import datetime

import streamlit as st

import agent
import config
import llm as llm_module
import rag
import storage
import tools
import ui

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

BOT_AVATAR = ":material/support_agent:"
USER_AVATAR = ":material/person:"

# Quick-reply chips shown on the welcome screen: label -> message sent.
QUICK_REPLIES = {
    "📦 Track my order": "Track my order",
    "↩️ Return policy": "What is your return policy?",
    "💸 Refund status": "How long does a refund take?",
    "💳 Payment options": "What payment options do you accept?",
    "🎟️ Coupons": "Why is my coupon not working?",
    "🧑‍💼 Talk to a human": "I want to talk to a human agent",
}

st.set_page_config(
    page_title="Ava · ShopEasy Support",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="auto",
)


# --------------------------------------------------------------------------
# Cached resources (created once per server process, shared by all sessions)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the ShopEasy knowledge base (first run downloads the embedding model)...")
def load_default_vectorstore():
    """Build or load the default FAISS index (cached on disk in vectorstore/)."""
    return rag.build_or_load_vectorstore()


@st.cache_resource(show_spinner=False)
def load_llm(provider: str, temperature: float, model: str | None = None,
             max_retries: int | None = None):
    """Create the LLM client. Arguments form the cache key."""
    return llm_module.get_llm(provider, temperature, model, max_retries)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def now_label() -> str:
    """Current time like '4:05 PM' (works on Windows and Linux)."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def find_message(message_id: str) -> dict | None:
    return next((m for m in st.session_state.messages if m.get("id") == message_id), None)


def chat_messages() -> list[dict]:
    """Real chat messages (no error notices or ticket cards)."""
    return [m for m in st.session_state.messages if m.get("kind", "text") == "text" and not m.get("is_error")]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "session_id": new_id(),
        "messages": [],             # chat history (dicts, see handle_user_message)
        "offer_escalation": False,  # show the escalation card?
        "escalation_reason": "",
        "tickets": [],              # ticket IDs created in this session
        "last_order_id": None,      # remembered for follow-up questions
        "pending_prompt": None,     # set by quick-reply chips / sample buttons
        "feedback": {},             # message_id -> "up" / "down"
        "provider": config.LLM_PROVIDER if config.LLM_PROVIDER in config.SUPPORTED_PROVIDERS else "gemini",
        "temperature": float(config.TEMPERATURE),
        "agent_mode": config.USE_TOOL_CALLING,
        "language": config.REPLY_LANGUAGE if config.REPLY_LANGUAGE in config.LANGUAGES else "auto",
        # Dark mode can be preset with ?theme=dark in the URL.
        "dark_mode": st.query_params.get("theme", "light") == "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# Callbacks (run before the script reruns)
# --------------------------------------------------------------------------
def on_theme_change() -> None:
    st.query_params["theme"] = "dark" if st.session_state.dark_mode else "light"


def on_quick_reply() -> None:
    label = st.session_state.get("quick_reply")
    if label:
        st.session_state.pending_prompt = QUICK_REPLIES[label]
    st.session_state.quick_reply = None  # reset the chip selection


def on_feedback(message_id: str) -> None:
    """Save a 👍/👎 rating for one bot reply to data/feedback.json."""
    value = st.session_state.get(f"fb_{message_id}")
    rating = {1: "up", 0: "down"}.get(value)  # st.feedback("thumbs"): 1 = 👍, 0 = 👎
    message = find_message(message_id)
    if message is None:
        return
    if rating:
        st.session_state.feedback[message_id] = rating
    else:
        st.session_state.feedback.pop(message_id, None)
    storage.save_feedback(
        message_id=message_id,
        rating=rating,
        question=message.get("question", ""),
        answer=message.get("content", ""),
        provider=message.get("provider", st.session_state.provider),
        model=message.get("model", config.model_name_for(st.session_state.provider)),
    )
    st.toast("Thanks for your feedback!" if rating else "Feedback removed", icon="💬")


def escalate_to_human(reason: str) -> None:
    """Create a support ticket and add a ticket card to the chat."""
    ticket = tools.create_ticket(
        reason=reason or "Customer requested a human agent",
        conversation=st.session_state.messages,
        order_id=st.session_state.last_order_id,
    )
    add_ticket_card(ticket)


def add_ticket_card(ticket: dict) -> None:
    """Show a created ticket in the chat (used by the button and by the agent)."""
    st.session_state.tickets.append(ticket["ticket_id"])
    st.session_state.messages.append(
        {
            "id": new_id(),
            "role": "assistant",
            "kind": "ticket",
            "ticket": ticket,
            "content": (
                f"Support ticket {ticket['ticket_id']} created. A human agent will contact "
                f"you {ticket['eta']}."
            ),
            "time": now_label(),
            "dismissed": False,
        }
    )
    st.session_state.offer_escalation = False
    st.session_state.escalation_reason = ""


def dismiss_escalation() -> None:
    st.session_state.offer_escalation = False


def continue_chatting(message_id: str) -> None:
    """'Continue chatting' on a ticket card: hide the button, Ava says she's still here."""
    message = find_message(message_id)
    if message:
        message["dismissed"] = True
    st.session_state.messages.append(
        {
            "id": new_id(),
            "role": "assistant",
            "kind": "text",
            "content": ("Of course! I'm still here while our team looks into your ticket. "
                        "What else can I help you with?"),
            "time": now_label(),
            "question": "",
            "sources": [],
            "order_ids": [],
        }
    )


def new_chat() -> None:
    """Start a fresh conversation (tickets are kept)."""
    for m in st.session_state.messages:
        st.session_state.pop(f"fb_{m.get('id')}", None)
    st.session_state.messages = []
    st.session_state.offer_escalation = False
    st.session_state.escalation_reason = ""
    st.session_state.last_order_id = None
    st.session_state.feedback = {}
    st.session_state.session_id = new_id()


def build_transcript() -> str:
    """Plain-text transcript for the 'Download chat transcript' button."""
    lines = [
        f"{config.COMPANY_NAME} Customer Support - chat transcript",
        f"Exported: {datetime.now():%Y-%m-%d %H:%M}",
        f"Assistant: {config.BOT_NAME} ({config.PROVIDER_LABELS.get(st.session_state.provider)})",
        "-" * 60,
    ]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else config.BOT_NAME
        lines.append(f"[{m.get('time', '')}] {who}: {m['content']}")
        lines.append("")
    if st.session_state.tickets:
        lines.append("Tickets: " + ", ".join(st.session_state.tickets))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_order_card(order_id: str) -> None:
    """Compact expander with the order's details below the answer."""
    order = tools.lookup_order(order_id)
    if order is None:
        st.caption(f"📦 Order **{order_id}** was not found in our system.")
        return
    with st.expander(f"📦 Order {order['order_id']} · {order['status']}"):
        items = "\n".join(
            f"- {i['name']} × {i['quantity']} — ₹{i['price']:,}" for i in order["items"]
        )
        st.markdown(
            f"**Ordered on:** {order['order_date']}  \n"
            f"**Status:** {order['status']}  \n"
            f"**Expected delivery:** {order['expected_delivery'] or '—'}  \n"
            f"**Delivered on:** {order['delivered_on'] or '—'}  \n"
            f"**Courier / tracking:** {order['courier'] or '—'} / {order['tracking_number'] or '—'}  \n"
            f"**Total:** ₹{order['total_amount']:,} ({order['payment_method']})\n\n"
            f"**Items:**\n{items}\n\n"
            f"_{order['notes']}_"
        )


def render_sources(sources: list[dict]) -> None:
    """Collapsible list of the knowledge-base chunks used for an answer."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            snippet = src["content"].replace("\n", " ").strip()
            if len(snippet) > 260:
                snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
            st.markdown(f"**{i}. {src['title']}**  \n<small>{src['source']}</small>", unsafe_allow_html=True)
            st.caption(snippet)


def render_user_message(m: dict) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        with st.container(key=f"user_bubble_{m['id']}"):
            st.markdown(m["content"])
        st.markdown(ui.meta_html(f"You · {m.get('time', '')}", align_right=True), unsafe_allow_html=True)


def render_bot_message(m: dict) -> None:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.container(key=f"bot_bubble_{m['id']}"):
            st.markdown(m["content"])
        with st.container(horizontal=True, vertical_alignment="center", key=f"meta_{m['id']}"):
            meta = f"{config.BOT_NAME} · {m.get('time', '')}"
            if m.get("response_time"):
                meta += f" · {m['response_time']:.1f}s"
            if m.get("fallback_from"):
                meta += f" · ⚡ answered by {m.get('model')} ({m['fallback_from']} was busy)"
            st.markdown(ui.meta_html(meta), unsafe_allow_html=True)
            if not m.get("is_error") and m.get("question"):
                st.feedback("thumbs", key=f"fb_{m['id']}", on_change=on_feedback, args=(m["id"],))
        if m.get("tool_calls"):
            st.caption("🛠️ " + " · ".join(m["tool_calls"]))
        for order_id in m.get("order_ids", []):
            render_order_card(order_id)
        render_sources(m.get("sources", []))


def render_ticket_message(m: dict) -> None:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(ui.ticket_card_html(m["ticket"], config.SUPPORT_PHONE), unsafe_allow_html=True)
        if not m.get("dismissed"):
            st.button("💬 Continue chatting", key=f"cont_{m['id']}", on_click=continue_chatting,
                      args=(m["id"],), type="primary")
        st.markdown(ui.meta_html(f"{config.BOT_NAME} · {m.get('time', '')}"), unsafe_allow_html=True)


def render_message(m: dict) -> None:
    if m["role"] == "user":
        render_user_message(m)
    elif m.get("kind") == "ticket":
        render_ticket_message(m)
    else:
        render_bot_message(m)


def render_welcome(disabled: bool) -> None:
    st.markdown(ui.welcome_html(config.BOT_NAME, config.COMPANY_NAME), unsafe_allow_html=True)
    st.markdown('<div class="chips-label">Quick replies</div>', unsafe_allow_html=True)
    st.pills(
        "Quick replies",
        options=list(QUICK_REPLIES.keys()),
        key="quick_reply",
        on_change=on_quick_reply,
        label_visibility="collapsed",
        disabled=disabled,
    )


def render_escalation_offer() -> None:
    with st.container(key="escalation_offer"):
        st.markdown(ui.escalation_offer_html(), unsafe_allow_html=True)
        with st.container(horizontal=True):
            st.button("🧑‍💼 Escalate to human agent", key="escalate_main", type="primary",
                      on_click=escalate_to_human, args=(st.session_state.escalation_reason,))
            st.button("No thanks", key="escalate_dismiss", on_click=dismiss_escalation)


def render_sidebar(config_error: str | None, default_chunks: int | None) -> None:
    with st.sidebar:
        st.markdown(ui.sidebar_brand_html(config.COMPANY_NAME), unsafe_allow_html=True)

        # --- Appearance ---
        st.toggle("🌙 Dark mode", key="dark_mode", on_change=on_theme_change)

        # --- Model ---
        st.markdown(ui.section_html("Model"), unsafe_allow_html=True)
        st.selectbox(
            "LLM provider",
            options=list(config.SUPPORTED_PROVIDERS),
            format_func=lambda p: config.PROVIDER_LABELS[p],
            key="provider",
        )
        st.caption(f"Model: `{config.model_name_for(st.session_state.provider)}`")
        backups = llm_module.fallback_chain(st.session_state.provider)[1:]
        if backups:
            st.caption("Fallback if busy: " + " → ".join(f"`{m}`" for _, m in backups))
        st.slider("Temperature", 0.0, 1.0, step=0.05, key="temperature",
                  help="Lower = more factual and consistent, higher = more creative.")
        st.toggle("🛠️ Agent mode (tool calling)", key="agent_mode",
                  help="On: the LLM decides when to look up or cancel an order or create a "
                       "ticket. Off: keyword rules decide.")
        st.selectbox("Reply language", options=list(config.LANGUAGES),
                     format_func=lambda code: config.LANGUAGES[code], key="language")
        if config_error:
            st.error("This provider isn't configured yet. See the setup message.", icon="🔑")

        # --- Knowledge base ---
        st.markdown(ui.section_html("Knowledge base"), unsafe_allow_html=True)
        if default_chunks is not None:
            st.markdown(f'<span class="kb-pill">ShopEasy FAQ · {default_chunks} chunks</span>', unsafe_allow_html=True)

        # --- Conversation ---
        st.markdown(ui.section_html("Conversation"), unsafe_allow_html=True)
        with st.container(horizontal=True):
            st.button("✨ New chat", on_click=new_chat, key="new_chat")
            st.download_button(
                "⬇️ Transcript",
                data=build_transcript(),
                file_name=f"shopeasy_chat_{datetime.now():%Y%m%d_%H%M}.txt",
                mime="text/plain",
                disabled=not st.session_state.messages,
                key="download_transcript",
                on_click="ignore",
            )
        st.button("🧑‍💼 Escalate to human agent", key="sidebar_escalate", width="stretch",
                  on_click=escalate_to_human, args=("Customer requested a human agent (sidebar)",))

        # --- Live stats ---
        st.markdown(ui.section_html("Session stats"), unsafe_allow_html=True)
        times = [m["response_time"] for m in st.session_state.messages if m.get("response_time")]
        avg = f"{sum(times) / len(times):.1f}s" if times else "—"
        st.markdown(ui.stats_html(len(chat_messages()), avg, len(st.session_state.feedback)),
                    unsafe_allow_html=True)
        if st.session_state.tickets:
            st.caption("Tickets: " + ", ".join(f"`{t}`" for t in st.session_state.tickets))


# --------------------------------------------------------------------------
# Core chat turn
# --------------------------------------------------------------------------
def handle_user_message(prompt: str, vectorstore, chat_llm) -> None:
    """Process one user message end-to-end and stream the assistant reply."""
    provider = st.session_state.provider
    agent_mode = st.session_state.agent_mode
    history = list(st.session_state.messages)  # messages BEFORE this question
    previous_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    started = time.perf_counter()

    user_msg = {"id": new_id(), "role": "user", "kind": "text", "content": prompt, "time": now_label()}
    st.session_state.messages.append(user_msg)
    render_user_message(user_msg)

    # 1. Order lookup tool (keyword mode only; in agent mode the LLM calls it) --
    order_ids: list[str] = [] if agent_mode else tools.extract_order_ids(prompt)
    order_info = None
    if order_ids:
        st.session_state.last_order_id = order_ids[0]
        order_info = "\n\n".join(
            tools.format_order_for_prompt(oid, tools.lookup_order(oid)) for oid in order_ids
        )
    elif st.session_state.last_order_id and not agent_mode:
        # Allows follow-ups like "when will it arrive?" after sharing an order ID.
        oid = st.session_state.last_order_id
        order_info = (
            "(This is the order the customer mentioned earlier in the conversation. "
            "Use it only if the question is about that order.)\n"
            + tools.format_order_for_prompt(oid, tools.lookup_order(oid))
        )

    # 2. Sentiment & human-request check ------------------------------------
    sentiment = tools.detect_sentiment(prompt, llm=chat_llm)
    frustrated = sentiment["label"] == "frustrated"
    wants_human = not agent_mode and tools.wants_human_agent(prompt)

    # 3. Retrieval ----------------------------------------------------------
    query = rag.build_search_query(prompt, previous_user)
    if llm_module.has_devanagari(query):  # FAQ + embeddings are English-only
        query = llm_module.translate_for_search(query, chat_llm)
    try:
        docs = rag.retrieve(vectorstore, query)
    except Exception as exc:
        logger.exception("Retrieval failed", exc_info=exc)
        docs = []
    context = rag.format_context(docs)
    sources = [
        {"title": d.metadata.get("title", "Excerpt"), "source": d.metadata.get("source", ""),
         "content": d.page_content}
        for d in docs
    ]

    # 4. Prompt -------------------------------------------------------------
    messages = llm_module.build_messages(
        question=prompt, context=context, history=history, order_info=order_info,
        frustrated=frustrated, wants_human=wants_human,
        agent_mode=agent_mode, language=st.session_state.language,
    )
    session = agent.AgentSession(st.session_state.messages, frustrated) if agent_mode else None

    # 5. Stream the answer into a bot bubble --------------------------------
    bot_msg: dict = {
        "id": new_id(), "role": "assistant", "kind": "text", "time": now_label(),
        "question": prompt, "provider": provider, "model": config.model_name_for(provider),
    }
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.container(key=f"bot_bubble_{bot_msg['id']}"):
            body = st.empty()
            body.markdown(ui.typing_html(config.BOT_NAME), unsafe_allow_html=True)
        status = st.empty()
        answer = ""

        def on_switch(failed_p: str, failed_m: str, next_p: str, next_m: str) -> None:
            status.caption(f"⚡ `{failed_m}` is busy or unavailable, trying `{next_m}`…")

        def run_tool(call: dict) -> str:
            status.caption(f"🛠️ {agent.describe_call(call)}…")
            return session.run(call)

        stream = llm_module.FallbackStream(
            messages, provider, float(st.session_state.temperature),
            client_factory=load_llm, on_switch=on_switch,
            tools=agent.TOOL_SCHEMAS if agent_mode else None,
            tool_runner=run_tool if agent_mode else None,
        )
        try:
            for token in stream:
                answer += token
                body.markdown(answer + " ▌")
                status.markdown(ui.typing_html(config.BOT_NAME), unsafe_allow_html=True)
            if not answer.strip():
                answer = ("I'm sorry, I couldn't generate a response. Could you rephrase your "
                          "question, or would you like me to connect you to a human agent?")
            body.markdown(answer)
            if session:
                order_ids = session.order_ids
            bot_msg.update(content=answer, sources=sources, order_ids=order_ids)
        except Exception as exc:
            error_text = stream.error_message(exc)
            body.markdown(error_text)
            bot_msg.update(content=error_text, is_error=True, sources=[], order_ids=[])
        status.empty()

        # Tools may have run even if the final answer failed, so record them anyway.
        if session:
            bot_msg["tool_calls"] = session.actions
            if session.order_ids:
                st.session_state.last_order_id = session.order_ids[-1]

        # Record the model that actually answered (used by feedback + analytics).
        if stream.used:
            bot_msg["provider"], bot_msg["model"] = stream.used
        if stream.fell_back:
            bot_msg["fallback_from"] = stream.chain[0][1]

    bot_msg["response_time"] = round(time.perf_counter() - started, 2)
    st.session_state.messages.append(bot_msg)
    agent_tickets = session.tickets if session else []
    for ticket in agent_tickets:
        add_ticket_card(ticket)

    # 6. Escalation offer (not needed if the agent already created a ticket) --
    if agent_tickets:
        st.session_state.offer_escalation = False
    elif frustrated or wants_human or bot_msg.get("is_error"):
        st.session_state.offer_escalation = True
        if frustrated:
            st.session_state.escalation_reason = "Customer appears frustrated: " + prompt[:200]
        elif wants_human:
            st.session_state.escalation_reason = "Customer requested a human agent: " + prompt[:200]
        else:
            st.session_state.escalation_reason = "Chatbot error during conversation"
    else:
        st.session_state.offer_escalation = False

    # 7. Analytics log (used by the Admin Dashboard) -------------------------
    storage.log_question(
        {
            "session_id": st.session_state.session_id,
            "question": prompt,
            "top_source": sources[0]["title"] if sources else None,
            "sources": [s["title"] for s in sources],
            "sentiment": sentiment["label"],
            "wants_human": wants_human,
            "order_ids": order_ids,
            "agent_mode": agent_mode,
            "tool_calls": bot_msg.get("tool_calls", []),
            "language": st.session_state.language,
            "provider": bot_msg["provider"],
            "model": bot_msg["model"],
            "fallback_from": bot_msg.get("fallback_from"),
            "response_time": bot_msg["response_time"],
            "error": bool(bot_msg.get("is_error")),
        }
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    init_state()
    ui.inject_css("dark" if st.session_state.dark_mode else "light")

    provider = st.session_state.provider
    config_error = config.validate_config(provider)

    # Knowledge base (FAISS index of data/faq.md).
    vectorstore = None
    kb_error = None
    try:
        vectorstore = load_default_vectorstore()
    except Exception as exc:
        logger.exception("Failed to load knowledge base", exc_info=exc)
        kb_error = exc

    # LLM client.
    chat_llm = None
    if not config_error:
        try:
            chat_llm = load_llm(provider, float(st.session_state.temperature))
        except Exception as exc:
            logger.exception("Failed to create LLM client", exc_info=exc)
            config_error = llm_module.friendly_error(exc, provider)

    render_sidebar(config_error, rag.count_chunks(vectorstore) if vectorstore else None)

    # Header.
    badge = f"{config.PROVIDER_LABELS[provider]} · {config.model_name_for(provider)}"
    st.markdown(ui.header_html(config.BOT_NAME, config.COMPANY_NAME, badge), unsafe_allow_html=True)

    if kb_error:
        st.error(
            "⚠️ Could not load the knowledge base. On the first run the app downloads the "
            f"embedding model (`{config.EMBEDDING_MODEL}`, ~90 MB), so an internet connection "
            "is required. Check your connection and that `data/faq.md` exists, then refresh."
            f"\n\nDetails: `{type(kb_error).__name__}: {kb_error}`"
        )
    if config_error:
        st.warning(f"**Setup needed:** {config_error}", icon="🔑")

    ready = vectorstore is not None and chat_llm is not None

    # Conversation.
    if not st.session_state.messages:
        render_welcome(disabled=not ready)
    for message in st.session_state.messages:
        render_message(message)
    if st.session_state.offer_escalation:
        render_escalation_offer()

    # New input: typed message or a clicked quick reply.
    typed = st.chat_input(
        f"Message {config.BOT_NAME}..." if ready else "Chat is unavailable until setup is complete",
        disabled=not ready,
    )
    prompt = typed or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if prompt and prompt.strip() and ready:
        handle_user_message(prompt.strip(), vectorstore, chat_llm)
        st.rerun()  # redraw so feedback buttons, sources and escalation card appear in order


# --------------------------------------------------------------------------
# Multipage navigation: the chat (this file) + the Admin Dashboard page.
# --------------------------------------------------------------------------
navigation = st.navigation(
    [
        st.Page(main, title="Chat with Ava", icon=":material/chat:", default=True),
        st.Page(
            config.BASE_DIR / "pages" / "Admin_Dashboard.py",
            title="Admin Dashboard",
            icon=":material/monitoring:",
        ),
    ]
)
navigation.run()
