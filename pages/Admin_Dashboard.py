"""
pages/Admin_Dashboard.py - Analytics for the support team.

app.py registers this file with st.navigation, so it appears in the sidebar as
"Admin Dashboard" next to "Chat with Ava".

Shows:
  * KPIs: questions asked, customer satisfaction, tickets, average reply time
  * Feedback: 👍/👎 counts, satisfaction by model, recent negative answers
  * Questions: most common FAQ topics, most frequent questions, volume per day
  * Tickets: escalated tickets with priority, reason and full transcript

Data comes from data/feedback.json, data/chat_log.jsonl and data/tickets.json.
Set ADMIN_PASSWORD in .env / secrets to protect this page.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this page.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import storage  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Admin Dashboard · ShopEasy", page_icon="📊", layout="wide")

# Streamlit forgets widget values when you switch pages. Re-assigning them here
# keeps the chat page's settings (theme, model, temperature) when you go back.
for _key in ("dark_mode", "provider", "temperature", "agent_mode", "language"):
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = st.query_params.get("theme", "light") == "dark"

ui.inject_css("dark" if st.session_state.dark_mode else "light")
BRAND = ui.BRAND


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(ui.sidebar_brand_html(config.COMPANY_NAME), unsafe_allow_html=True)
    st.toggle("🌙 Dark mode", key="dark_mode")
    if st.button("🔄 Refresh data", width="stretch"):
        st.rerun()


# --------------------------------------------------------------------------
# Optional password protection
# --------------------------------------------------------------------------
def check_password() -> bool:
    if not config.ADMIN_PASSWORD:
        return True
    if st.session_state.get("admin_ok"):
        return True
    st.markdown(
        '<div class="ui-card"><div class="card-title">🔒 Admin login</div>'
        '<div class="card-text">Enter the admin password to view support analytics.</div></div>',
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Password", type="password", key="admin_pwd")
    if st.button("Log in", type="primary"):
        if pwd == config.ADMIN_PASSWORD:
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


st.markdown(
    '<div class="ava-header"><div class="ava-avatar">📊</div><div>'
    '<div class="ava-title">Admin Dashboard</div>'
    '<div class="ava-status">Feedback, common questions and escalated tickets</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

if not check_password():
    st.stop()


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
feedback = pd.DataFrame(storage.load_feedback())
log = pd.DataFrame(storage.load_chat_log())
tickets = pd.DataFrame(storage.load_tickets())

for df, col in ((feedback, "timestamp"), (log, "timestamp"), (tickets, "created_at")):
    if not df.empty and col in df:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if feedback.empty and log.empty and tickets.empty:
    st.info(
        "No data yet. Chat with Ava on the main page, rate a few answers with 👍/👎 and "
        "escalate a conversation. Then come back here and press **Refresh data**.",
        icon="💡",
    )
    st.stop()


# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
n_questions = len(log)
n_sessions = log["session_id"].nunique() if "session_id" in log else 0
ups = int((feedback["rating"] == "up").sum()) if not feedback.empty else 0
downs = int((feedback["rating"] == "down").sum()) if not feedback.empty else 0
satisfaction = f"{ups / (ups + downs) * 100:.0f}%" if (ups + downs) else "—"
n_tickets = len(tickets)
high = int((tickets["priority"] == "High").sum()) if not tickets.empty else 0
avg_rt = (
    f"{log.loc[~log['error'], 'response_time'].mean():.1f}s"
    if not log.empty and "response_time" in log and (~log["error"]).any()
    else "—"
)
frustrated = int((log["sentiment"] == "frustrated").sum()) if "sentiment" in log else 0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(ui.kpi_html("Questions", str(n_questions), f"{n_sessions} chat sessions"), unsafe_allow_html=True)
c2.markdown(ui.kpi_html("Satisfaction", satisfaction, f"👍 {ups} · 👎 {downs}"), unsafe_allow_html=True)
c3.markdown(ui.kpi_html("Escalated tickets", str(n_tickets), f"{high} high priority"), unsafe_allow_html=True)
c4.markdown(ui.kpi_html("Avg reply time", avg_rt, f"{frustrated} frustrated messages"), unsafe_allow_html=True)
st.write("")

tab_fb, tab_q, tab_t = st.tabs(["👍 Feedback", "❓ Common questions", "🎫 Escalated tickets"])


# --------------------------------------------------------------------------
# Feedback tab
# --------------------------------------------------------------------------
with tab_fb:
    if feedback.empty:
        st.info("No ratings yet. Use the 👍/👎 buttons under Ava's replies.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Ratings")
            counts = pd.DataFrame({"Rating": ["Helpful", "Not helpful"], "Count": [ups, downs]})
            st.bar_chart(counts, x="Rating", y="Count", color=BRAND, horizontal=True, height=200)
        with right:
            st.subheader("Satisfaction by model")
            by_model = (
                feedback.assign(helpful=feedback["rating"].eq("up"))
                .groupby("model")["helpful"].mean().mul(100).round(0)
                .rename("Satisfaction %").reset_index()
            )
            st.bar_chart(by_model, x="model", y="Satisfaction %", color=BRAND, horizontal=True, height=200)

        st.subheader("Answers rated 'Not helpful' (most recent first)")
        negative = feedback[feedback["rating"] == "down"].sort_values("timestamp", ascending=False)
        if negative.empty:
            st.success("No negative feedback so far. 🎉")
        else:
            st.dataframe(
                negative[["timestamp", "question", "answer", "model"]],
                hide_index=True, width="stretch",
                column_config={"timestamp": st.column_config.DatetimeColumn("When", format="DD MMM, HH:mm")},
            )
        st.download_button("⬇️ Download feedback (CSV)", feedback.to_csv(index=False),
                           "feedback.csv", "text/csv")


# --------------------------------------------------------------------------
# Common questions tab
# --------------------------------------------------------------------------
def normalise(q: str) -> str:
    """Lower-case, mask order IDs and strip punctuation so similar questions group."""
    q = re.sub(r"\bse\d{5}\b", "<order id>", q.lower())
    q = re.sub(r"[^\w\s<>]", "", q)
    return re.sub(r"\s+", " ", q).strip()


with tab_q:
    if log.empty:
        st.info("No questions logged yet.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Top FAQ topics")
            st.caption("Which knowledge-base entry best matched each question")
            topics = log["top_source"].dropna().value_counts().head(10)
            topics_df = topics.rename_axis("Topic").reset_index(name="Questions")
            st.bar_chart(topics_df, x="Topic", y="Questions", color=BRAND, horizontal=True, height=340)
        with right:
            st.subheader("Questions per day")
            st.caption("Daily chat volume")
            per_day = log.set_index("timestamp").resample("D").size().rename("Questions")
            st.line_chart(per_day, color=BRAND, height=340)

        st.subheader("Most frequent questions")
        freq = (
            log.assign(normalised=log["question"].astype(str).map(normalise))
            .groupby("normalised")
            .agg(Asked=("question", "size"), Example=("question", "first"),
                 Topic=("top_source", "first"))
            .sort_values("Asked", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        st.dataframe(freq[["Asked", "Example", "Topic"]], hide_index=True, width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Customer sentiment")
            sentiment = log["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Messages")
            st.bar_chart(sentiment, x="Sentiment", y="Messages", color=BRAND, horizontal=True, height=200)
        with right:
            st.subheader("Model usage")
            usage = log["model"].value_counts().rename_axis("Model").reset_index(name="Questions")
            st.bar_chart(usage, x="Model", y="Questions", color=BRAND, horizontal=True, height=200)
        st.download_button("⬇️ Download chat log (CSV)", log.to_csv(index=False),
                           "chat_log.csv", "text/csv")


# --------------------------------------------------------------------------
# Tickets tab
# --------------------------------------------------------------------------
with tab_t:
    if tickets.empty:
        st.info("No tickets yet. Escalate a conversation from the chat to create one.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Tickets by priority")
            prio = tickets["priority"].value_counts().rename_axis("Priority").reset_index(name="Tickets")
            st.bar_chart(prio, x="Priority", y="Tickets", color=BRAND, horizontal=True, height=200)
        with right:
            st.subheader("Tickets per day")
            per_day = tickets.set_index("created_at").resample("D").size().rename("Tickets")
            st.bar_chart(per_day, color=BRAND, height=260)

        st.subheader("All tickets")
        view = tickets.sort_values("created_at", ascending=False)
        st.dataframe(
            view[["ticket_id", "created_at", "priority", "status", "order_id", "reason"]],
            hide_index=True, width="stretch",
            column_config={
                "ticket_id": "Ticket ID",
                "created_at": st.column_config.DatetimeColumn("Created", format="DD MMM, HH:mm"),
                "priority": "Priority",
                "status": "Status",
                "order_id": "Order",
                "reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )

        st.subheader("Ticket transcript")
        chosen = st.selectbox("Choose a ticket", view["ticket_id"].tolist())
        row = view[view["ticket_id"] == chosen].iloc[0]
        st.caption(f"Priority **{row['priority']}** · created {row['created_at']:%d %b %Y, %H:%M} · {row['reason']}")
        transcript = row.get("transcript") or []
        if not transcript:
            st.write("No messages were recorded for this ticket.")
        for m in transcript:
            who = "🧑 Customer" if m["role"] == "user" else f"🤖 {config.BOT_NAME}"
            st.markdown(f"**{who}:** {m['content']}")
        st.download_button("⬇️ Download tickets (CSV)",
                           tickets.drop(columns=["transcript"], errors="ignore").to_csv(index=False),
                           "tickets.csv", "text/csv")
