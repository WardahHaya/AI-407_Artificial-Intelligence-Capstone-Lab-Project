from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "feedback_log.db"))
API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://127.0.0.1:8000")


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


def send_chat_request(message: str, thread_id: str) -> dict[str, str]:
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"message": message, "thread_id": thread_id},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


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
    st.write("This Streamlit UI talks to the Lab 8 FastAPI service and records thumbs-based feedback into SQLite.")
    st.caption(f"API base URL: {API_BASE_URL}")
    st.caption(f"Active thread_id: {st.session_state['thread_id']}")

    render_chat_history()

    prompt = st.chat_input("Ask Buraq about emails, deadlines, grounded notes, or draft requests.")
    if not prompt:
        return

    try:
        payload = send_chat_request(prompt, st.session_state["thread_id"])
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
            "Could not reach the FastAPI agent. Start the Lab 8 API first, for example with "
            "`uvicorn main:app --host 0.0.0.0 --port 8000`.\n\n"
            f"Details: {exc}"
        )


if __name__ == "__main__":
    main()
