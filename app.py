from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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

    chosen_model = None
    mode = "live"
    if not os.getenv("GROQ_API_KEY"):
        chosen_model = StreamlitDemoModel()
        mode = "demo"

    collection = get_collection()
    if collection.count() == 0:
        ingest_chunks(load_project_chunks())

    return {
        "graph": build_secured_graph(model=chosen_model),
        "mode": mode,
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


def _connection_label() -> str:
    api_base_url = _get_api_base_url()
    if api_base_url:
        return f"Remote API ({api_base_url})"
    if _get_setting("GROQ_API_KEY"):
        return "Local agent (live model)"
    return "Local agent (demo mode)"


def init_session_state() -> None:
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("saved_feedback_ids", set())


def render_feedback_controls(turn: dict[str, str]) -> None:
    message_id = turn["message_id"]
    if message_id in st.session_state["saved_feedback_ids"]:
        st.caption("Feedback saved for this response.")
        return

    selected_score = st.session_state.get(f"feedback-score-{message_id}")
    selected_label = "None"
    if selected_score == 1:
        selected_label = "Thumbs Up"
    elif selected_score == -1:
        selected_label = "Thumbs Down"

    st.caption(
        f"thread_id: {st.session_state['thread_id']} | message_id: {message_id} | current selection: {selected_label}"
    )

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


def render_chat_history() -> None:
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["user_input"])

        with st.chat_message("assistant"):
            st.write(turn["agent_response"])
            st.caption(f"Status: {turn['status']}")
            render_feedback_controls(turn)


def main() -> None:
    st.set_page_config(page_title="Buraq Feedback Console", page_icon="mailbox", layout="wide")
    init_feedback_db()
    init_session_state()

    st.title("Buraq Feedback Console")
    st.write(
        "This Streamlit app can use a deployed FastAPI backend or run the secured LangGraph agent directly."
    )
    st.caption(f"Connection mode: {_connection_label()}")
    st.caption(f"Active thread_id: {st.session_state['thread_id']}")

    if not _get_api_base_url() and not _get_setting("GROQ_API_KEY"):
        st.info("`GROQ_API_KEY` is not configured, so the app is using the built-in demo model for deployment previews.")

    render_chat_history()

    prompt = st.chat_input("Ask Buraq about emails, deadlines, grounded notes, or draft requests.")
    if not prompt:
        return

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
            "The local agent could not start. If you want live model responses on Streamlit Cloud, add "
            "`GROQ_API_KEY` as a Streamlit secret.\n\n"
            f"Details: {exc}"
        )


if __name__ == "__main__":
    main()
