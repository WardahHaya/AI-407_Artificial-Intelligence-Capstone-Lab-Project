from __future__ import annotations

import html
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from runtime_services import ensure_runtime_dirs, init_schedule_db, list_scheduled_emails, list_stored_files, save_uploaded_bytes

load_dotenv()


def _get_setting(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value in {None, ""}:
        return default
    return str(secret_value)


def _load_runtime_secrets_into_env() -> None:
    for name in ["GROQ_API_KEY", "GOOGLE_CLIENT_SECRET_FILE", "CHROMA_HOST", "CHROMA_PORT", "CHROMA_SSL"]:
        value = _get_setting(name)
        if value and not os.getenv(name):
            os.environ[name] = value


DB_PATH = Path(_get_setting("FEEDBACK_DB_PATH", "feedback_log.db") or "feedback_log.db")


class StreamlitDemoModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        latest_user_message = next(
            (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
            "",
        )
        if "repeat that in one sentence" in latest_user_message.lower():
            return AIMessage(
                content=(
                    "Talent Team asked for your updated resume before May 6 and wants your interview availability."
                )
            )

        last_tool_message = next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)
        if last_tool_message is None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {
                            "query": "updated resume interview",
                            "department": "careers",
                            "top_k": 1,
                        },
                        "id": "streamlit_demo_tool_call",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content=(
                "I found the recruiter request in grounded memory. "
                "Talent Team asked for the updated resume before May 6 and requested interview availability."
            )
        )


def _get_api_base_url() -> str | None:
    api_base_url = _get_setting("AGENT_API_BASE_URL")
    if api_base_url:
        return api_base_url.rstrip("/")
    return None


def _history_to_messages(chat_history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in chat_history:
        messages.append(HumanMessage(content=turn["user_input"]))
        messages.append(AIMessage(content=turn["agent_response"]))
    return messages


def _graph_input(messages: list[BaseMessage]) -> dict[str, object]:
    return {
        "messages": messages,
        "safety_status": "safe",
        "guardrail_reason": "",
        "sanitized_output": "",
    }


def _status_from_result(result: dict[str, object]) -> str:
    if result.get("safety_status") == "unsafe":
        return "blocked"
    return "completed"


@st.cache_resource(show_spinner=False)
def _get_local_runtime() -> dict[str, object]:
    _load_runtime_secrets_into_env()

    from ingest_data import get_collection, ingest_chunks, load_project_chunks
    from secured_graph import build_secured_graph
    from tools import deliver_email_message
    from runtime_services import start_scheduler_thread

    chosen_model = None
    mode = "live"
    if not os.getenv("GROQ_API_KEY"):
        chosen_model = StreamlitDemoModel()
        mode = "demo"

    ensure_runtime_dirs()
    init_schedule_db()
    collection = get_collection()
    if collection.count() == 0:
        ingest_chunks(load_project_chunks())

    stop_event, thread = start_scheduler_thread(deliver_email_message)

    return {
        "graph": build_secured_graph(model=chosen_model),
        "mode": mode,
        "scheduler_stop_event": stop_event,
        "scheduler_thread": thread,
    }


def init_feedback_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                feedback_score INTEGER NOT NULL,
                optional_comment TEXT
            )
            """
        )
        conn.commit()


def log_feedback(
    thread_id: str,
    message_id: str,
    user_input: str,
    agent_response: str,
    feedback_score: int,
    optional_comment: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback_log (
                timestamp,
                thread_id,
                message_id,
                user_input,
                agent_response,
                feedback_score,
                optional_comment
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                thread_id,
                message_id,
                user_input,
                agent_response,
                feedback_score,
                optional_comment.strip() or None,
            ),
        )
        conn.commit()


def _send_remote_chat_request(message: str, thread_id: str) -> dict[str, str]:
    api_base_url = _get_api_base_url()
    if not api_base_url:
        raise RuntimeError("AGENT_API_BASE_URL is not configured.")

    response = requests.post(
        f"{api_base_url}/chat",
        json={"message": message, "thread_id": thread_id},
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    payload["mode"] = "remote"
    return payload


def _send_local_chat_request(message: str, thread_id: str, chat_history: list[dict[str, str]]) -> dict[str, str]:
    runtime = _get_local_runtime()
    history = _history_to_messages(chat_history)
    result = runtime["graph"].invoke(_graph_input([*history, HumanMessage(content=message)]))
    final_answer = str(result.get("sanitized_output") or result["messages"][-1].content)
    return {
        "thread_id": str(thread_id),
        "message_id": str(uuid4()),
        "final_answer": final_answer,
        "status": _status_from_result(result),
        "mode": str(runtime["mode"]),
    }


def send_chat_request(message: str, thread_id: str, chat_history: list[dict[str, str]]) -> dict[str, str]:
    if _get_api_base_url():
        return _send_remote_chat_request(message, thread_id)
    return _send_local_chat_request(message, thread_id, chat_history)


def _upload_remote_file(uploaded_file) -> dict[str, object]:
    api_base_url = _get_api_base_url()
    if not api_base_url:
        raise RuntimeError("AGENT_API_BASE_URL is not configured.")

    response = requests.post(
        f"{api_base_url}/upload",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def _upload_local_file(uploaded_file) -> dict[str, object]:
    ensure_runtime_dirs()
    return save_uploaded_bytes(uploaded_file.name, uploaded_file.getvalue(), area="uploads")


def store_uploaded_file(uploaded_file) -> dict[str, object]:
    if _get_api_base_url():
        return _upload_remote_file(uploaded_file)
    return _upload_local_file(uploaded_file)


def get_stored_files() -> list[dict[str, object]]:
    if _get_api_base_url():
        api_base_url = _get_api_base_url()
        if not api_base_url:
            return []
        response = requests.get(f"{api_base_url}/uploads", timeout=30)
        response.raise_for_status()
        return list(response.json().get("files", []))

    ensure_runtime_dirs()
    return list_stored_files()


def get_scheduled_items() -> list[dict[str, object]]:
    if _get_api_base_url():
        api_base_url = _get_api_base_url()
        if not api_base_url:
            return []
        response = requests.get(f"{api_base_url}/scheduled", timeout=30)
        response.raise_for_status()
        return list(response.json().get("items", []))

    init_schedule_db()
    return list_scheduled_emails()


def _connection_label() -> str:
    api_base_url = _get_api_base_url()
    if api_base_url:
        return f"Remote API ({api_base_url})"
    if _get_setting("GROQ_API_KEY"):
        return "Local agent (live model)"
    return "Local agent (demo model)"


def _gmail_status_label() -> str:
    creds_path = Path(_get_setting("GOOGLE_CLIENT_SECRET_FILE", "credentials.json") or "credentials.json")
    token_path = Path("token.pickle")
    if token_path.exists():
        return "Gmail connected"
    if creds_path.exists():
        return "Credentials found, OAuth still pending"
    return "No Gmail credentials file found"


def _safe_html(value: object) -> str:
    return html.escape(str(value))


def _text_block_html(text: str) -> str:
    return "<br>".join(_safe_html(line) for line in str(text).splitlines())


def _status_badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="status-pill status-{tone}">{_safe_html(text)}</span>'


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #fbf8f3;
            --bg-strong: #f4eadc;
            --panel: rgba(255, 252, 247, 0.92);
            --panel-strong: #fffaf2;
            --ink: #1f2933;
            --muted: #6b7280;
            --accent: #d97745;
            --accent-strong: #b85b2f;
            --accent-soft: #f7dfd0;
            --teal: #2f5d62;
            --teal-soft: #dce9e8;
            --gold: #c08a22;
            --gold-soft: #f7ecd2;
            --success: #2f6b4f;
            --success-soft: #dcefe5;
            --danger: #b24136;
            --danger-soft: #f6ddd8;
            --line: #eadfce;
            --shadow: 0 24px 60px rgba(31, 41, 51, 0.08);
            --radius-lg: 26px;
            --radius-md: 20px;
            --radius-sm: 14px;
        }

        html, body, [class*="css"]  {
            font-family: "Manrope", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top right, rgba(217, 119, 69, 0.18), transparent 28%),
                radial-gradient(circle at top left, rgba(47, 93, 98, 0.10), transparent 24%),
                linear-gradient(180deg, #f5ede1 0%, #faf7f1 38%, #fdfbf7 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 249, 241, 0.97), rgba(246, 238, 227, 0.97));
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-shell {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.8rem 1.9rem;
            border-radius: var(--radius-lg);
            background:
                linear-gradient(135deg, rgba(255, 253, 249, 0.96), rgba(248, 240, 230, 0.96)),
                radial-gradient(circle at right top, rgba(217, 119, 69, 0.18), transparent 36%);
            border: 1px solid rgba(234, 223, 206, 0.92);
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            margin: 0 0 0.45rem 0;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            font-weight: 800;
            color: var(--teal);
        }

        .hero-shell h1 {
            margin: 0;
            font-size: 2.9rem;
            line-height: 1;
            letter-spacing: -0.04em;
            color: var(--ink);
        }

        .hero-text {
            margin: 0.85rem 0 1rem 0;
            max-width: 48rem;
            font-size: 1rem;
            line-height: 1.65;
            color: #415164;
        }

        .hero-badges, .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .hero-aside {
            min-width: 14rem;
            border-radius: var(--radius-md);
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(234, 223, 206, 0.85);
            align-self: stretch;
        }

        .aside-label {
            margin: 0;
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
        }

        .aside-value {
            margin: 0.2rem 0 1rem 0;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.44rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
            border: 1px solid transparent;
        }

        .status-accent {
            color: var(--accent-strong);
            background: var(--accent-soft);
            border-color: rgba(217, 119, 69, 0.22);
        }

        .status-teal {
            color: var(--teal);
            background: var(--teal-soft);
            border-color: rgba(47, 93, 98, 0.18);
        }

        .status-success {
            color: var(--success);
            background: var(--success-soft);
            border-color: rgba(47, 107, 79, 0.18);
        }

        .status-warning {
            color: #8a5a00;
            background: var(--gold-soft);
            border-color: rgba(192, 138, 34, 0.22);
        }

        .status-danger {
            color: var(--danger);
            background: var(--danger-soft);
            border-color: rgba(178, 65, 54, 0.18);
        }

        .status-neutral {
            color: #475467;
            background: rgba(241, 238, 233, 0.96);
            border-color: rgba(71, 84, 103, 0.12);
        }

        .metric-card {
            min-height: 9.8rem;
            padding: 1.1rem 1.1rem 1rem 1.1rem;
            border-radius: var(--radius-md);
            border: 1px solid rgba(234, 223, 206, 0.92);
            background: var(--panel);
            box-shadow: var(--shadow);
        }

        .metric-label {
            margin: 0;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
        }

        .metric-value {
            margin: 0.55rem 0 0.35rem 0;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.05em;
        }

        .metric-caption {
            margin: 0;
            font-size: 0.93rem;
            line-height: 1.55;
            color: #4b5563;
        }

        .mail-card, .surface-card {
            padding: 1rem 1.05rem;
            border-radius: var(--radius-md);
            border: 1px solid rgba(234, 223, 206, 0.94);
            background: var(--panel);
            box-shadow: var(--shadow);
            margin-bottom: 0.9rem;
        }

        .mail-meta {
            margin: 0;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 800;
        }

        .mail-title {
            margin: 0.35rem 0 0.2rem 0;
            font-size: 1.15rem;
            line-height: 1.35;
            font-weight: 800;
            color: var(--ink);
        }

        .mail-from {
            margin: 0 0 0.75rem 0;
            font-size: 0.96rem;
            color: #465464;
        }

        .mail-snippet {
            margin: 0.8rem 0 0 0;
            font-size: 0.96rem;
            line-height: 1.6;
            color: #334155;
        }

        .section-title {
            margin: 0 0 0.2rem 0;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .section-copy {
            margin: 0 0 1rem 0;
            color: #5b6472;
            line-height: 1.65;
        }

        .mono-note {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.8rem;
            color: #5c6470;
        }

        .tiny-stack {
            display: grid;
            gap: 0.45rem;
        }

        .sidebar-panel {
            padding: 1rem;
            border: 1px solid rgba(234, 223, 206, 0.92);
            border-radius: var(--radius-md);
            background: rgba(255, 252, 247, 0.75);
            margin-bottom: 1rem;
        }

        .sidebar-panel p {
            margin: 0.15rem 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.8rem;
            padding: 0 1rem;
            border-radius: 999px;
            background: rgba(255, 252, 247, 0.84);
            border: 1px solid rgba(234, 223, 206, 0.95);
            color: #536173;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(217, 119, 69, 0.12);
            color: var(--accent-strong);
            border-color: rgba(217, 119, 69, 0.3);
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stBaseButton-secondary"] {
            border-radius: 999px;
            border: 1px solid rgba(217, 119, 69, 0.22);
            background: linear-gradient(180deg, #fffaf5, #f7ede2);
            color: var(--ink);
            font-weight: 700;
            min-height: 2.75rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(217, 119, 69, 0.48);
            color: var(--accent-strong);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader section {
            border-radius: var(--radius-sm);
            background: rgba(255, 252, 247, 0.94);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stDateInput input,
        .stTimeInput input {
            border: 1px solid rgba(234, 223, 206, 0.95);
        }

        .stChatMessage {
            border-radius: var(--radius-md);
            border: 1px solid rgba(234, 223, 206, 0.92);
            background: rgba(255, 252, 247, 0.84);
            box-shadow: var(--shadow);
        }

        .stExpander {
            border: 1px solid rgba(234, 223, 206, 0.92);
            border-radius: var(--radius-md);
            background: rgba(255, 252, 247, 0.82);
        }

        .stDataFrame, div[data-testid="stTable"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid rgba(234, 223, 206, 0.92);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("saved_feedback_ids", set())
    st.session_state.setdefault("last_uploaded_ref", "")
    st.session_state.setdefault("dashboard_refresh_key", 0)
    st.session_state.setdefault("queued_prompt", None)


def _truncate(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _extract_spam_count(spam_text: str) -> int:
    match = re.search(r"(\d+)\s+spam email", spam_text.lower())
    if match:
        return int(match.group(1))
    if "no spam" in spam_text.lower():
        return 0
    return 0


@st.cache_data(ttl=75, show_spinner=False)
def load_dashboard_snapshot(refresh_key: int) -> dict[str, object]:
    del refresh_key
    _load_runtime_secrets_into_env()

    from tools import _load_available_inbox_rows, _load_deadline_rows, check_important_alerts, check_spam, daily_email_summary

    rows, source = _load_available_inbox_rows(max_results=12)
    urgent_rows = [
        row
        for row in rows
        if str(row.get("priority", "")).lower() == "high" or str(row.get("action_required", "")).lower() == "yes"
    ]
    reply_rows = [
        row
        for row in rows
        if str(row.get("subject", "")).lower().startswith("re:") or "reply" in str(row.get("snippet", "")).lower()
    ]
    upcoming_deadlines = [row for row in _load_deadline_rows() if row.get("status") != "completed"][:5]
    spam_text = check_spam.invoke({"max_results": 10})
    alerts_text = check_important_alerts.invoke({"max_results": 6})
    summary_text = daily_email_summary.invoke({"date": "today"})

    return {
        "source": source,
        "rows": rows,
        "urgent_rows": urgent_rows[:6],
        "reply_rows": reply_rows[:6],
        "deadline_rows": upcoming_deadlines,
        "spam_text": spam_text,
        "spam_count": _extract_spam_count(spam_text),
        "alerts_text": alerts_text,
        "summary_text": summary_text,
        "last_refreshed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _load_feedback_snapshot(limit: int = 20) -> dict[str, object]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total_items,
                SUM(CASE WHEN feedback_score = 1 THEN 1 ELSE 0 END) AS positive_items,
                SUM(CASE WHEN feedback_score = -1 THEN 1 ELSE 0 END) AS negative_items
            FROM feedback_log
            """
        ).fetchone()
        recent_rows = conn.execute(
            """
            SELECT timestamp, user_input, feedback_score, optional_comment
            FROM feedback_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "total_items": int(totals["total_items"] or 0),
        "positive_items": int(totals["positive_items"] or 0),
        "negative_items": int(totals["negative_items"] or 0),
        "recent_rows": [dict(row) for row in recent_rows],
    }


def render_feedback_controls(turn: dict[str, str]) -> None:
    message_id = turn["message_id"]
    if message_id in st.session_state["saved_feedback_ids"]:
        st.caption("Feedback saved for this response.")
        return

    selected_score = st.session_state.get(f"feedback-score-{message_id}")
    selected_label = "Not rated yet"
    if selected_score == 1:
        selected_label = "Thumbs Up"
    elif selected_score == -1:
        selected_label = "Thumbs Down"

    st.caption(f"Status: {selected_label}")

    up_col, down_col = st.columns(2)
    if up_col.button("Thumbs Up", key=f"up-{message_id}", use_container_width=True):
        st.session_state[f"feedback-score-{message_id}"] = 1
    if down_col.button("Thumbs Down", key=f"down-{message_id}", use_container_width=True):
        st.session_state[f"feedback-score-{message_id}"] = -1

    comment = st.text_area(
        "Optional comment",
        key=f"comment-{message_id}",
        placeholder="Why did this response work or fail?",
    )

    if st.button("Save Feedback", key=f"save-{message_id}", use_container_width=True):
        feedback_score = st.session_state.get(f"feedback-score-{message_id}")
        if feedback_score not in {-1, 1}:
            st.warning("Choose Thumbs Up or Thumbs Down before saving.")
            return

        log_feedback(
            thread_id=st.session_state["thread_id"],
            message_id=message_id,
            user_input=turn["user_input"],
            agent_response=turn["agent_response"],
            feedback_score=feedback_score,
            optional_comment=comment,
        )
        st.session_state["saved_feedback_ids"].add(message_id)
        st.success("Feedback saved.")


def _render_multiline_markdown(text: str) -> None:
    st.markdown(text.replace("\n", "  \n"))


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-label">{_safe_html(title)}</p>
            <p class="metric-value">{_safe_html(value)}</p>
            <p class="metric-caption">{_safe_html(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mail_card(row: dict[str, str]) -> None:
    priority = str(row.get("priority", "unknown")).lower()
    action_required = str(row.get("action_required", "unknown")).lower()
    priority_tone = "danger" if priority == "high" else "warning" if priority == "medium" else "neutral"
    action_tone = "accent" if action_required == "yes" else "neutral"
    category = str(row.get("category", "general")).title()
    snippet = _truncate(str(row.get("snippet", "")), limit=240)

    st.markdown(
        f"""
        <div class="mail-card">
            <p class="mail-meta">{_safe_html(row.get("date", "Unknown time"))}</p>
            <p class="mail-title">{_safe_html(row.get("subject", "No Subject"))}</p>
            <p class="mail-from">From {_safe_html(row.get("from", "Unknown"))}</p>
            <div class="chip-row">
                {_status_badge(f"Priority: {priority.title()}", priority_tone)}
                {_status_badge(f"Action: {action_required.title()}", action_tone)}
                {_status_badge(category, "teal")}
            </div>
            <p class="mail-snippet">{_safe_html(snippet)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div class="surface-card">
                <p class="section-title">No conversation yet</p>
                <p class="section-copy">
                    Start with a quick action or ask Buraq to inspect Gmail, summarize your day, draft a response,
                    or search the grounded knowledge base.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["user_input"])

        with st.chat_message("assistant"):
            st.write(turn["agent_response"])
            tone = "success" if turn["status"] == "completed" else "danger"
            st.markdown(_status_badge(f"Status: {turn['status']}", tone), unsafe_allow_html=True)
            with st.expander("Rate this response", expanded=False):
                render_feedback_controls(turn)


def _send_direct_email(to: str, subject: str, body: str, attachment_ref: str | None = None) -> tuple[bool, str]:
    _load_runtime_secrets_into_env()
    from tools import deliver_email_message

    return deliver_email_message(to=to, subject=subject, body=body, attachment_ref=attachment_ref)


def _schedule_direct_email(to: str, subject: str, body: str, send_at: str) -> str:
    _load_runtime_secrets_into_env()
    from tools import schedule_email

    return str(schedule_email.invoke({"to": to, "subject": subject, "body": body, "send_at": send_at}))


def _sync_live_emails() -> str:
    _load_runtime_secrets_into_env()
    from tools import ingest_emails_to_vector_store

    return str(ingest_emails_to_vector_store.invoke({"max_results": 25, "include_sent_style": True}))


def _download_recent_attachments(query: str, max_results: int = 5) -> str:
    _load_runtime_secrets_into_env()
    from tools import download_attachments

    payload = {"max_results": max_results}
    if query.strip():
        payload["query"] = query.strip()
    return str(download_attachments.invoke(payload))


def render_sidebar(snapshot: dict[str, object]) -> None:
    with st.sidebar:
        st.markdown("### Mission Control")
        st.markdown(
            f"""
            <div class="sidebar-panel tiny-stack">
                <p><strong>Mode</strong></p>
                <p>{_safe_html(_connection_label())}</p>
                <p><strong>Gmail</strong></p>
                <p>{_safe_html(_gmail_status_label())}</p>
                <p><strong>Source</strong></p>
                <p>{'Live Gmail' if snapshot.get('source') == 'gmail' else 'Grounded sample inbox'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh inbox view", use_container_width=True):
            st.session_state["dashboard_refresh_key"] += 1
            st.rerun()

        if st.button("Start new conversation", use_container_width=True):
            st.session_state["thread_id"] = str(uuid4())
            st.session_state["chat_history"] = []
            st.session_state["queued_prompt"] = None
            st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-panel tiny-stack">
                <p><strong>Thread ID</strong></p>
                <p class="mono-note">{_safe_html(st.session_state['thread_id'])}</p>
                <p><strong>Last refresh</strong></p>
                <p>{_safe_html(snapshot.get('last_refreshed', ''))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("last_uploaded_ref"):
            st.markdown(
                f"""
                <div class="sidebar-panel tiny-stack">
                    <p><strong>Latest stored file</strong></p>
                    <p class="mono-note">{_safe_html(st.session_state['last_uploaded_ref'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not _get_api_base_url() and not _get_setting("GROQ_API_KEY"):
            st.info(
                "Chat is using the built-in demo model right now. The inbox widgets can still use live Gmail if OAuth is connected."
            )


def render_header(snapshot: dict[str, object]) -> None:
    source_badge = _status_badge("Live Gmail" if snapshot.get("source") == "gmail" else "Sample inbox", "teal")
    connection_badge = _status_badge(_connection_label(), "accent")
    gmail_badge = _status_badge(_gmail_status_label(), "success" if "connected" in _gmail_status_label().lower() else "warning")

    st.markdown(
        f"""
        <div class="hero-shell">
            <div>
                <p class="hero-kicker">AI Gmail workspace</p>
                <h1>Buraq</h1>
                <p class="hero-text">
                    A cleaner front end for live inbox triage, grounded search, scheduling, and attachment workflows.
                    Instead of hiding everything behind one chat box, the app now shows what is happening in your mail.
                </p>
                <div class="hero-badges">
                    {connection_badge}
                    {gmail_badge}
                    {source_badge}
                </div>
            </div>
            <div class="hero-aside">
                <p class="aside-label">Active thread</p>
                <p class="aside-value mono-note">{_safe_html(st.session_state["thread_id"])}</p>
                <p class="aside-label">Recent cards</p>
                <p class="aside-value">{len(snapshot.get("rows", []))} inbox items loaded</p>
                <p class="aside-label">Focus</p>
                <p class="aside-value">{len(snapshot.get("urgent_rows", []))} urgent or action-heavy items</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inbox_tab(snapshot: dict[str, object]) -> None:
    st.markdown('<p class="section-title">Inbox Overview</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Recent email cards, urgent alerts, and a grounded day summary all in one place.</p>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Recent emails", str(len(snapshot.get("rows", []))), "Latest inbox items loaded into the workspace.")
    with metric_cols[1]:
        render_metric_card("Urgent focus", str(len(snapshot.get("urgent_rows", []))), "Messages tagged high priority or action-required.")
    with metric_cols[2]:
        render_metric_card("Reply signals", str(len(snapshot.get("reply_rows", []))), "Recent messages that look like replies.")
    with metric_cols[3]:
        render_metric_card("Spam count", str(snapshot.get("spam_count", 0)), "Live Gmail spam signal from the connected account.")

    action_cols = st.columns(4)
    quick_actions = [
        ("Show 5 recent emails", "Fetch my 5 most recent emails and show sender, subject, and what needs action."),
        ("What is urgent?", "Tell me which inbox items and deadlines need my attention right now."),
        ("Summarize today", "Give me a summary of today's emails and highlight anything I should answer."),
        ("Find recruiter emails", "Search my inbox for recruiter or resume-related emails."),
    ]
    for index, (label, prompt) in enumerate(quick_actions):
        if action_cols[index].button(label, use_container_width=True):
            st.session_state["queued_prompt"] = prompt
            st.rerun()

    main_col, side_col = st.columns([1.45, 0.95], gap="large")
    with main_col:
        st.markdown("### Recent Inbox Cards")
        rows = snapshot.get("rows", [])
        if rows:
            for row in rows:
                render_mail_card(row)
        else:
            st.info("No inbox items are available yet.")

    with side_col:
        st.markdown("### Day Summary")
        st.markdown(
            f'<div class="surface-card">{_text_block_html(str(snapshot.get("summary_text", "No summary is available right now.")))}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Important Alerts")
        st.markdown(
            f'<div class="surface-card">{_text_block_html(str(snapshot.get("alerts_text", "No urgent items right now.")))}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Upcoming Deadlines")
        deadlines = snapshot.get("deadline_rows", [])
        if deadlines:
            for row in deadlines:
                st.markdown(
                    f"""
                    <div class="surface-card">
                        <p class="mail-meta">{_safe_html(row.get('due_date', 'Unknown due date'))}</p>
                        <p class="mail-title">{_safe_html(row.get('title', 'Untitled'))}</p>
                        <p class="mail-from">Owner {_safe_html(row.get('owner', 'Unknown'))} | Status {_safe_html(row.get('status', 'unknown'))}</p>
                        <p class="mail-snippet">{_safe_html(_truncate(str(row.get('details', '')), limit=180))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No deadline records are available.")


def render_ask_tab() -> None:
    st.markdown('<p class="section-title">Ask Buraq</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Use chat for open-ended reasoning, drafting, grounded search, or Gmail follow-up questions.</p>',
        unsafe_allow_html=True,
    )
    render_chat_history()


def render_compose_tab() -> None:
    st.markdown('<p class="section-title">Compose, Store, and Schedule</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Direct controls for attachments, sending, queueing, and live inbox maintenance.</p>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.12, 0.88], gap="large")

    with left_col:
        st.markdown("### Send an Email")
        stored_files = get_stored_files()
        file_options = ["No attachment"] + [str(item["ref"]) for item in stored_files]

        with st.form("send-email-form", clear_on_submit=False):
            to_address = st.text_input("To", placeholder="name@example.com")
            subject = st.text_input("Subject", placeholder="What is this email about?")
            body = st.text_area("Message", height=200, placeholder="Write the message body here.")
            attachment_ref = st.selectbox("Attachment", options=file_options)
            send_now = st.form_submit_button("Send Now", use_container_width=True)

        if send_now:
            if not to_address.strip() or not subject.strip() or not body.strip():
                st.warning("To, subject, and message body are all required.")
            else:
                selected_ref = None if attachment_ref == "No attachment" else attachment_ref
                success, detail = _send_direct_email(
                    to=to_address.strip(),
                    subject=subject.strip(),
                    body=body.strip(),
                    attachment_ref=selected_ref,
                )
                if success:
                    st.success(detail)
                else:
                    st.error(detail)

        st.markdown("### Queue for Later")
        default_time = (datetime.now() + timedelta(hours=1)).replace(second=0, microsecond=0)
        with st.form("schedule-email-form", clear_on_submit=False):
            queue_to = st.text_input("Queue to", key="queue-to", placeholder="name@example.com")
            queue_subject = st.text_input("Queue subject", key="queue-subject", placeholder="Scheduled follow-up")
            queue_body = st.text_area(
                "Queue message",
                key="queue-body",
                height=170,
                placeholder="Write the scheduled message body here.",
            )
            send_date = st.date_input("Send date", value=default_time.date())
            send_time = st.time_input("Send time", value=default_time.time())
            queue_submit = st.form_submit_button("Schedule Email", use_container_width=True)

        if queue_submit:
            if not queue_to.strip() or not queue_subject.strip() or not queue_body.strip():
                st.warning("Queue to, subject, and message body are all required.")
            else:
                scheduled_for = datetime.combine(send_date, send_time)
                try:
                    result = _schedule_direct_email(
                        to=queue_to.strip(),
                        subject=queue_subject.strip(),
                        body=queue_body.strip(),
                        send_at=scheduled_for.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                except Exception as exc:
                    st.error(f"Scheduling failed: {exc}")
                else:
                    st.success(result)

        st.markdown("### Inbox Utilities")
        utility_cols = st.columns(2)
        if utility_cols[0].button("Sync live emails to vector store", use_container_width=True):
            try:
                sync_result = _sync_live_emails()
            except Exception as exc:
                st.error(f"Sync failed: {exc}")
            else:
                st.success(sync_result)

        attachment_query = utility_cols[1].text_input(
            "Attachment search query",
            key="download-query",
            placeholder="optional Gmail query",
        )
        if utility_cols[1].button("Download recent attachments", use_container_width=True):
            try:
                download_result = _download_recent_attachments(attachment_query)
            except Exception as exc:
                st.error(f"Attachment download failed: {exc}")
            else:
                st.success(download_result)

    with right_col:
        st.markdown("### File Locker")
        uploaded_file = st.file_uploader("Add a local file to managed storage", key="workspace-uploader")
        if uploaded_file is not None:
            if st.button("Store file", key="store-file-button", use_container_width=True):
                stored = store_uploaded_file(uploaded_file)
                st.session_state["last_uploaded_ref"] = str(stored["ref"])
                st.success(f"Stored {stored['name']} as {stored['ref']}")

        if st.session_state.get("last_uploaded_ref"):
            st.caption(f"Latest reference: {st.session_state['last_uploaded_ref']}")

        files = get_stored_files()
        if files:
            st.dataframe(
                [
                    {
                        "ref": item["ref"],
                        "size_bytes": item["size_bytes"],
                        "modified_at": item["modified_at"],
                    }
                    for item in files[:20]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No stored files yet.")

        st.markdown("### Scheduled Queue")
        scheduled_items = get_scheduled_items()
        if scheduled_items:
            st.dataframe(
                [
                    {
                        "id": item["id"],
                        "status": item["status"],
                        "send_at": item["send_at"],
                        "to": item["to_address"],
                        "subject": item["subject"],
                    }
                    for item in scheduled_items[:20]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No scheduled emails yet.")


def render_feedback_tab() -> None:
    snapshot = _load_feedback_snapshot()
    st.markdown('<p class="section-title">Feedback Signal</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">The UI still keeps the lab feedback loop, but it now lives in its own space instead of taking over every response.</p>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Logged ratings", str(snapshot["total_items"]), "Saved thumbs-up and thumbs-down events.")
    with metric_cols[1]:
        render_metric_card("Positive", str(snapshot["positive_items"]), "Responses that landed well.")
    with metric_cols[2]:
        render_metric_card("Negative", str(snapshot["negative_items"]), "Responses worth revisiting.")

    recent_rows = snapshot["recent_rows"]
    if recent_rows:
        st.dataframe(
            [
                {
                    "timestamp": row["timestamp"],
                    "feedback": "up" if row["feedback_score"] == 1 else "down",
                    "prompt": _truncate(str(row["user_input"]), limit=70),
                    "comment": _truncate(str(row.get("optional_comment") or ""), limit=90),
                }
                for row in recent_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No feedback has been logged yet.")


def _process_prompt(prompt: str) -> None:
    try:
        with st.spinner("Buraq is working..."):
            payload = send_chat_request(prompt, st.session_state["thread_id"], st.session_state["chat_history"])
        turn = {
            "message_id": payload["message_id"],
            "user_input": prompt,
            "agent_response": payload["final_answer"],
            "status": payload["status"],
        }
        st.session_state["chat_history"].append(turn)
        st.rerun()
    except requests.RequestException as exc:
        st.error(
            "Could not reach the configured FastAPI backend. Update `AGENT_API_BASE_URL` or remove it to use the "
            "local in-process agent.\n\n"
            f"Details: {exc}"
        )
    except Exception as exc:
        st.error(
            "The agent could not complete that request. If you want live local responses, make sure `GROQ_API_KEY` "
            "is configured.\n\n"
            f"Details: {exc}"
        )


def main() -> None:
    st.set_page_config(page_title="Buraq Workspace", page_icon=":material/mail:", layout="wide")
    ensure_runtime_dirs()
    init_schedule_db()
    init_feedback_db()
    init_session_state()
    _inject_theme()

    try:
        snapshot = load_dashboard_snapshot(st.session_state["dashboard_refresh_key"])
    except Exception as exc:
        snapshot = {
            "source": "sample",
            "rows": [],
            "urgent_rows": [],
            "reply_rows": [],
            "deadline_rows": [],
            "spam_text": f"Snapshot unavailable: {exc}",
            "spam_count": 0,
            "alerts_text": f"Snapshot unavailable: {exc}",
            "summary_text": f"Snapshot unavailable: {exc}",
            "last_refreshed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.warning(f"Dashboard snapshot could not be loaded: {exc}")

    render_sidebar(snapshot)
    render_header(snapshot)

    tab_inbox, tab_ask, tab_compose, tab_feedback = st.tabs(
        ["Inbox", "Ask Buraq", "Compose", "Feedback"]
    )

    with tab_inbox:
        render_inbox_tab(snapshot)

    with tab_ask:
        render_ask_tab()

    with tab_compose:
        render_compose_tab()

    with tab_feedback:
        render_feedback_tab()

    prompt = st.chat_input("Ask Buraq about Gmail, deadlines, recruiter emails, attachments, or drafts.")
    queued_prompt = st.session_state.pop("queued_prompt", None)
    if prompt:
        _process_prompt(prompt)
    elif queued_prompt:
        _process_prompt(str(queued_prompt))


if __name__ == "__main__":
    main()
