"""
ui.py - Look & feel for the ShopEasy support app.

Everything visual lives here so app.py can focus on the chat logic:
  * inject_css()        - Inter font, light/dark colour tokens, chat bubbles,
                          header, cards, chips, responsive (mobile) rules, and
                          hiding Streamlit's default menu/footer/Deploy button
  * header_html()       - gradient header bar with avatar + "Online" dot
  * welcome_html()      - greeting card shown when the chat is empty
  * typing_html()       - animated "Ava is typing…" indicator
  * ticket_card_html()  - escalation card with ticket ID and response time
  * ...and a few small helpers (sidebar brand, stats grid, timestamps)

How the chat bubbles work: every message body is placed inside an
st.container(key="user_bubble_<id>") or st.container(key="bot_bubble_<id>").
Streamlit adds the CSS class "st-key-<key>" to keyed containers, so the CSS
below can target [class*="st-key-user_bubble"] to style user bubbles and use
:has() to flip the whole user row to the right-hand side.
"""

import html

import streamlit as st

BRAND = "#4F46E5"
BRAND_DARK = "#4338CA"
BRAND_2 = "#7C3AED"

# Colour tokens for each theme. All custom CSS uses these variables, so the
# light/dark toggle only has to swap this one block.
THEMES = {
    "light": {
        "bg": "#F5F6FA",
        "surface": "#FFFFFF",
        "surface-2": "#F1F2F7",
        "text": "#111827",
        "muted": "#6B7280",
        "border": "#E4E7EE",
        "bot-bubble": "#EEF0F4",
        "bot-text": "#111827",
        "card-shadow": "0 1px 2px rgba(16,24,40,.04), 0 4px 16px rgba(16,24,40,.06)",
        "bubble-shadow": "0 1px 2px rgba(16,24,40,.06)",
        "header-grad": f"linear-gradient(135deg, {BRAND} 0%, {BRAND_2} 100%)",
        "chip-bg": "#FFFFFF",
        "chip-hover": "#EEF0FF",
        "success": "#16A34A",
        "warn-bg": "#FFF7ED",
        "warn-border": "#FED7AA",
        "ok-bg": "#F0FDF4",
        "ok-border": "#BBF7D0",
    },
    "dark": {
        "bg": "#0B1020",
        "surface": "#121A2E",
        "surface-2": "#18213A",
        "text": "#E6E8EF",
        "muted": "#9AA3B5",
        "border": "#253050",
        "bot-bubble": "#1C2541",
        "bot-text": "#E6E8EF",
        "card-shadow": "0 1px 2px rgba(0,0,0,.3), 0 6px 20px rgba(0,0,0,.35)",
        "bubble-shadow": "0 1px 2px rgba(0,0,0,.35)",
        "header-grad": "linear-gradient(135deg, #4338CA 0%, #6D28D9 100%)",
        "chip-bg": "#18213A",
        "chip-hover": "#232E52",
        "success": "#22C55E",
        "warn-bg": "#2A1E12",
        "warn-border": "#7C4A1E",
        "ok-bg": "#0F2A1C",
        "ok-border": "#1F6B43",
    },
}

AVATAR_SVG = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" '
    'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M6 7h12l-1 13H7L6 7z"/><path d="M9 7a3 3 0 0 1 6 0"/></svg>'
)


def _theme_vars(mode: str) -> str:
    tokens = THEMES.get(mode, THEMES["light"])
    body = "".join(f"--{k}:{v};" for k, v in tokens.items())
    return f":root{{--brand:{BRAND};--brand-dark:{BRAND_DARK};--brand-2:{BRAND_2};{body}}}"


BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Font ---------- */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown, [data-testid="stSidebar"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}

/* ---------- Hide default Streamlit chrome ---------- */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
.stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbarActions"] { display: none !important; visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; height: 2.5rem; }

/* ---------- Page surfaces ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important; color: var(--text);
}
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
  color: var(--text);
}
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 860px; padding-top: 1rem !important; padding-bottom: 1rem !important;
}
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 860px; padding-bottom: 1.25rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--surface) !important; border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: var(--text); }
[data-testid="stSidebarNav"] a span { color: var(--text) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--chip-hover) !important; }
.sb-section {
  font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); margin: 1.1rem 0 .35rem;
}

/* ---------- Widgets (so they follow light/dark mode) ---------- */
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"], .stApp input, .stApp textarea {
  background: var(--surface-2) !important; color: var(--text) !important;
  border-color: var(--border) !important;
}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {
  background: var(--surface) !important; color: var(--text) !important;
}
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  font-weight: 500 !important; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--brand) !important; color: var(--brand) !important;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--brand) !important; color: #fff !important; border-color: var(--brand) !important;
}
.stButton > button[kind="primary"]:hover { background: var(--brand-dark) !important; color: #fff !important; }
.stButton > button[kind="primary"] * { color: #fff !important; }
[data-testid="stExpander"] details, [data-testid="stExpander"] > details {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--muted) !important; font-size: .85rem; }
[data-testid="stExpander"] summary:hover * { color: var(--brand) !important; }
[data-testid="stAlert"] { border-radius: 12px; }
.stApp code { color: var(--brand) !important; background: var(--surface-2) !important; border-radius: 6px; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] code { font-size: .75rem; }

/* Chat input */
[data-testid="stChatInput"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 16px !important; box-shadow: var(--card-shadow);
}
[data-testid="stChatInput"] > div { background: transparent !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--text) !important; }
[data-testid="stSelectbox"] > div > div {
  background: var(--surface-2) !important; border-color: var(--border) !important;
}
[data-testid="stSelectbox"] svg { color: var(--muted) !important; fill: var(--muted) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--brand) !important; color: #fff !important; border-radius: 12px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #fff !important; color: #fff !important; }
[data-testid="stChatInputSubmitButton"]:disabled { opacity: .45; }

/* ---------- Header bar ---------- */
.ava-header {
  display: flex; align-items: center; gap: .85rem; padding: .9rem 1.1rem;
  background: var(--header-grad); color: #fff; border-radius: 18px;
  box-shadow: 0 8px 24px rgba(79,70,229,.25); margin-bottom: 1rem;
}
/* Keep the header visible while the conversation scrolls */
[data-testid="stElementContainer"]:has(.ava-header) { position: sticky; top: .4rem; z-index: 50; }
[data-testid="stChatMessage"] [data-testid="stExpander"] { max-width: min(82%, 600px); }
.ava-avatar {
  position: relative; flex: 0 0 auto; width: 46px; height: 46px; border-radius: 50%;
  background: rgba(255,255,255,.18); border: 2px solid rgba(255,255,255,.35);
  display: flex; align-items: center; justify-content: center; color: #fff;
}
.ava-avatar .dot {
  position: absolute; right: 0; bottom: 1px; width: 12px; height: 12px; border-radius: 50%;
  background: #22C55E; border: 2px solid #fff;
}
.ava-title { font-weight: 700; font-size: 1.06rem; line-height: 1.25; color: #fff; }
.ava-status { font-size: .8rem; opacity: .92; display: flex; align-items: center; gap: .4rem; color: #fff; }
.ava-status .pulse {
  width: 8px; height: 8px; border-radius: 50%; background: #22C55E;
  box-shadow: 0 0 0 0 rgba(34,197,94,.7); animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(34,197,94,.6); }
  70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
.ava-badge {
  margin-left: auto; font-size: .72rem; font-weight: 600; padding: .3rem .65rem;
  border-radius: 999px; background: rgba(255,255,255,.18); color: #fff; white-space: nowrap;
}

/* ---------- Chat messages & bubbles ---------- */
[data-testid="stChatMessage"] {
  background: transparent !important; padding: .2rem 0 !important; gap: .6rem;
}
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--header-grad) !important; color: #fff;
  display: flex; align-items: center; justify-content: center;
}
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] * { color: #fff !important; fill: #fff; }
/* User rows: avatar hidden, everything aligned to the right */
[data-testid="stChatMessage"]:has([class*="st-key-user_bubble"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([class*="st-key-user_bubble"]) [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessage"]:has([class*="st-key-user_bubble"]) [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] { align-items: flex-end; }

[class*="st-key-user_bubble"], [class*="st-key-bot_bubble"] {
  width: fit-content !important; max-width: min(82%, 600px); padding: .62rem .95rem;
  box-shadow: var(--bubble-shadow); gap: .25rem !important;
  animation: bubbleIn .18s ease-out;
}
[class*="st-key-user_bubble"] {
  margin-left: auto; background: var(--brand); border-radius: 18px 18px 4px 18px;
}
[class*="st-key-user_bubble"] *, [class*="st-key-user_bubble"] a { color: #fff !important; }
[class*="st-key-bot_bubble"] {
  background: var(--bot-bubble); border-radius: 18px 18px 18px 4px;
  border: 1px solid var(--border);
}
[class*="st-key-bot_bubble"] * { color: var(--bot-text); }
[class*="st-key-bot_bubble"] a { color: var(--brand) !important; }
/* Streamlit gives markdown blocks margin-bottom:-1rem (to cancel the last <p> margin).
   Inside bubbles and for our own HTML snippets that would clip text, so reset it. */
[class*="_bubble"] [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"]:has(.ui-card, .msg-meta, .ava-header, .brand, .stats-grid,
  .sb-section, .chips-label, .kpi, .kb-pill, .typing) { margin-bottom: 0 !important; }
[class*="_bubble"] p { margin-bottom: .35rem; line-height: 1.55; font-size: .95rem; }
[class*="_bubble"] p:last-child { margin-bottom: 0; }
[class*="_bubble"] ul, [class*="_bubble"] ol { margin: .2rem 0 .3rem 1rem; padding-left: .4rem; }
[class*="_bubble"] code { background: rgba(127,127,127,.15); padding: .05rem .3rem; border-radius: 5px; }
@keyframes bubbleIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
.bubble-error { color: #DC2626 !important; font-weight: 500; }

/* Timestamp + feedback row */
[class*="st-key-meta_"] { gap: .4rem !important; min-height: 1.6rem; }
.msg-meta { font-size: .72rem; color: var(--muted) !important; margin: 0; white-space: nowrap; }
.msg-meta.right { text-align: right; }
[class*="st-key-meta_"] [data-testid="stMarkdownContainer"] p { margin: 0; }
[class*="st-key-meta_"] button { padding: .1rem .3rem !important; min-height: 1.6rem !important; }
[class*="st-key-meta_"] button span { color: var(--muted); }

/* ---------- Cards ---------- */
.ui-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.2rem 1.3rem; box-shadow: var(--card-shadow); margin: .2rem 0 .8rem;
}
.welcome-title { font-size: 1.25rem; font-weight: 700; margin: 0 0 .3rem; color: var(--text); }
.welcome-sub { color: var(--muted); font-size: .93rem; margin: 0 0 .9rem; line-height: 1.55; }
.welcome-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; }
.welcome-item {
  background: var(--surface-2); border-radius: 12px; padding: .6rem .7rem;
  font-size: .8rem; color: var(--muted); line-height: 1.35;
}
.welcome-item b { display: block; color: var(--text); font-size: .85rem; margin-bottom: .1rem; }
.chips-label { font-size: .78rem; font-weight: 600; color: var(--muted); margin: .4rem 0 .2rem; }

/* Quick-reply chips (st.pills) */
[data-testid="stButtonGroup"] button {
  background: var(--chip-bg) !important; color: var(--brand) !important;
  border: 1px solid var(--brand) !important; border-radius: 999px !important;
  font-weight: 500 !important; padding: .35rem .9rem !important;
}
[data-testid="stButtonGroup"] button:hover { background: var(--chip-hover) !important; }
[data-testid="stButtonGroup"] button * { color: var(--brand) !important; }

/* Escalation & ticket cards */
.esc-card { border-left: 4px solid #F59E0B; background: var(--warn-bg); border-color: var(--warn-border); }
.ticket-card { border-left: 4px solid var(--success); background: var(--ok-bg); border-color: var(--ok-border); }
.card-title { font-weight: 700; font-size: 1rem; margin: 0 0 .25rem; color: var(--text); }
.card-text { font-size: .88rem; color: var(--muted); margin: 0; line-height: 1.5; }
.ticket-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: .8rem 0 .2rem; }
.ticket-cell { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .55rem .7rem; }
.ticket-cell .k { font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
.ticket-cell .v { font-size: .92rem; font-weight: 700; color: var(--text); word-break: break-all; }
.prio-high { color: #DC2626 !important; }

/* Typing indicator */
.typing { display: inline-flex; align-items: center; gap: .55rem; color: var(--muted); font-size: .85rem; padding: .15rem 0; }
.typing .dots { display: inline-flex; gap: 4px; }
.typing .dots span {
  width: 7px; height: 7px; border-radius: 50%; background: var(--muted);
  animation: blink 1.2s infinite ease-in-out both;
}
.typing .dots span:nth-child(2) { animation-delay: .15s; }
.typing .dots span:nth-child(3) { animation-delay: .3s; }
@keyframes blink { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }

/* Sidebar brand + stats */
.brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .6rem; }
.brand-logo {
  width: 42px; height: 42px; border-radius: 12px; background: var(--header-grad);
  display: flex; align-items: center; justify-content: center; color: #fff;
  box-shadow: 0 6px 16px rgba(79,70,229,.3);
}
.brand-name { font-weight: 700; font-size: 1.1rem; color: var(--text); line-height: 1.1; }
.brand-tag { font-size: .75rem; color: var(--muted); }
.brand-desc { font-size: .82rem; color: var(--muted); line-height: 1.5; margin: 0 0 .4rem; }
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; }
.stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: .55rem .4rem; text-align: center; }
.stat .v { font-size: 1.05rem; font-weight: 700; color: var(--text); }
.stat .k { font-size: .66rem; color: var(--muted); line-height: 1.2; margin-top: .1rem; }
.kb-pill {
  display: inline-block; font-size: .72rem; padding: .2rem .55rem; border-radius: 999px;
  background: var(--chip-hover); color: var(--brand); font-weight: 600; margin-bottom: .4rem;
}

/* Admin dashboard */
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.1rem; box-shadow: var(--card-shadow); }
.kpi .k { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.kpi .v { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: .15rem; }
.kpi .s { font-size: .78rem; color: var(--muted); }

/* ---------- Responsive (phones) ---------- */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container { padding-left: .8rem !important; padding-right: .8rem !important; }
  .ava-header { padding: .7rem .8rem; border-radius: 14px; gap: .6rem; }
  .ava-avatar { width: 38px; height: 38px; }
  .ava-title { font-size: .95rem; }
  .ava-badge { display: none; }
  [class*="st-key-user_bubble"], [class*="st-key-bot_bubble"] { max-width: 92%; }
  .welcome-grid, .ticket-grid { grid-template-columns: 1fr; }
  .ui-card { padding: 1rem; border-radius: 14px; }
  [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { width: 28px; height: 28px; }
}
"""


def inject_css(mode: str = "light") -> None:
    """Inject the theme tokens + all custom CSS into the page."""
    st.markdown(f"<style>{_theme_vars(mode)}{BASE_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTML snippets (kept on single lines: indented lines would be treated as
# Markdown code blocks by st.markdown)
# --------------------------------------------------------------------------
def header_html(bot_name: str, company: str, badge: str) -> str:
    return (
        '<div class="ava-header">'
        f'<div class="ava-avatar">{AVATAR_SVG}<span class="dot"></span></div>'
        '<div>'
        f'<div class="ava-title">{html.escape(bot_name)} · {html.escape(company)} Support</div>'
        '<div class="ava-status"><span class="pulse"></span>Online · replies instantly</div>'
        '</div>'
        f'<div class="ava-badge">{html.escape(badge)}</div>'
        '</div>'
    )


def welcome_html(bot_name: str, company: str) -> str:
    return (
        '<div class="ui-card">'
        f'<div class="welcome-title">Hi there 👋 I\'m {html.escape(bot_name)}</div>'
        f'<div class="welcome-sub">Your {html.escape(company)} support assistant. Ask me anything about '
        'your orders, deliveries, returns, refunds, payments or account. To track an order, '
        'just share your order ID (for example <b>SE12345</b>).</div>'
        '<div class="welcome-grid">'
        '<div class="welcome-item"><b>📦 Orders</b>Live status for any order ID</div>'
        '<div class="welcome-item"><b>↩️ Returns</b>Policies, refunds &amp; pickups</div>'
        '<div class="welcome-item"><b>🧑‍💼 Human help</b>Escalate with a ticket anytime</div>'
        '</div>'
        '</div>'
    )


def typing_html(bot_name: str) -> str:
    return (
        '<div class="typing"><span class="dots"><span></span><span></span><span></span></span>'
        f'{html.escape(bot_name)} is typing…</div>'
    )


def meta_html(text: str, align_right: bool = False) -> str:
    cls = "msg-meta right" if align_right else "msg-meta"
    return f'<div class="{cls}">{html.escape(text)}</div>'


def escalation_offer_html() -> str:
    return (
        '<div class="ui-card esc-card">'
        '<div class="card-title">Need a human to take a look?</div>'
        '<div class="card-text">I can hand this conversation to our support team. They\'ll see '
        'everything we discussed, so you won\'t have to repeat yourself.</div>'
        '</div>'
    )


def ticket_card_html(ticket: dict, phone: str) -> str:
    prio = ticket.get("priority", "Normal")
    prio_cls = "v prio-high" if prio == "High" else "v"
    return (
        '<div class="ui-card ticket-card">'
        '<div class="card-title">✅ Your request has been escalated</div>'
        '<div class="card-text">A human support agent will review this conversation and contact you '
        'on your registered phone number or email.</div>'
        '<div class="ticket-grid">'
        f'<div class="ticket-cell"><div class="k">Ticket ID</div><div class="v">{html.escape(ticket["ticket_id"])}</div></div>'
        f'<div class="ticket-cell"><div class="k">Est. response</div><div class="v">{html.escape(ticket.get("eta", ""))}</div></div>'
        f'<div class="ticket-cell"><div class="k">Priority</div><div class="{prio_cls}">{html.escape(prio)}</div></div>'
        '</div>'
        f'<div class="card-text">Urgent? Call <b>{html.escape(phone)}</b> (9 AM–9 PM IST) and quote your ticket ID.</div>'
        '</div>'
    )


def sidebar_brand_html(company: str) -> str:
    return (
        '<div class="brand">'
        f'<div class="brand-logo">{AVATAR_SVG}</div>'
        f'<div><div class="brand-name">{html.escape(company)}</div>'
        '<div class="brand-tag">Customer Support</div></div>'
        '</div>'
        '<div class="brand-desc">AI support assistant powered by a pre-trained LLM with '
        'Retrieval-Augmented Generation over the ShopEasy help centre.</div>'
    )


def section_html(title: str) -> str:
    return f'<div class="sb-section">{html.escape(title)}</div>'


def stats_html(messages: int, avg_response: str, feedback: int) -> str:
    def cell(v, k):
        return f'<div class="stat"><div class="v">{html.escape(str(v))}</div><div class="k">{k}</div></div>'

    return (
        '<div class="stats-grid">'
        + cell(messages, "Messages")
        + cell(avg_response, "Avg reply")
        + cell(feedback, "Ratings")
        + "</div>"
    )


def kpi_html(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi"><div class="k">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="s">{html.escape(sub)}</div></div>'
    )
