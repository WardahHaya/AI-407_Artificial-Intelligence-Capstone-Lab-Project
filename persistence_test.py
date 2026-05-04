from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

CHECKPOINT_DB_PATH = Path("checkpoint_db.sqlite")


def reset_checkpoint_db() -> None:
    checkpoint_artifacts = [
        CHECKPOINT_DB_PATH.with_name(f"{CHECKPOINT_DB_PATH.name}-shm"),
        CHECKPOINT_DB_PATH.with_name(f"{CHECKPOINT_DB_PATH.name}-wal"),
        CHECKPOINT_DB_PATH,
    ]

    for _ in range(5):
        try:
            for artifact in checkpoint_artifacts:
                if artifact.exists():
                    artifact.unlink()
            return
        except PermissionError:
            time.sleep(0.2)

    raise RuntimeError(
        "checkpoint_db.sqlite is currently in use by another process. "
        "Close other running demos or use a different thread_id, then try again."
    )


class PersistenceState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    facts: dict[str, str]
    answer: str


def extract_facts_from_text(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}

    patterns = {
        "project": r"my project is(?: called)? (?P<value>.+?)(?=, my |, and my | and my |$)",
        "supervisor": r"my supervisor is (?P<value>.+?)(?=, my |, and my | and my |$)",
        "internship_contact": r"my internship contact is (?P<value>.+?)(?=, my |, and my | and my |$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            facts[key] = match.group("value").strip().rstrip(".,")

    return facts


def answer_from_facts(question: str, facts: dict[str, str]) -> str:
    normalized = question.lower()
    asks_project = "project" in normalized
    asks_supervisor = "supervisor" in normalized
    asks_contact = "internship contact" in normalized or "contact" in normalized

    if asks_project and asks_supervisor:
        project = facts.get("project", "unknown")
        supervisor = facts.get("supervisor", "unknown")
        if project != "unknown" or supervisor != "unknown":
            return f"Your remembered project is {project}, and your remembered supervisor is {supervisor}."
        return "I do not have saved project or supervisor details for this thread yet."

    if asks_project:
        if facts.get("project"):
            return f"Your remembered project is {facts['project']}."
        return "I do not have a saved project name for this thread yet."
    if asks_supervisor:
        if facts.get("supervisor"):
            return f"Your remembered supervisor is {facts['supervisor']}."
        return "I do not have a saved supervisor for this thread yet."
    if asks_contact:
        if facts.get("internship_contact"):
            return f"Your remembered internship contact is {facts['internship_contact']}."
        return "I do not have a saved internship contact for this thread yet."

    if facts:
        remembered = ", ".join(f"{key}={value}" for key, value in facts.items())
        return f"I still remember these facts for this thread: {remembered}."
    return "This thread does not have any saved facts yet."


def persistence_node(state: PersistenceState) -> PersistenceState:
    facts = dict(state.get("facts", {}))
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            facts.update(extract_facts_from_text(message.content))

    latest_user_message = next(
        (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
        "",
    )

    if "remember" in latest_user_message.lower() or extract_facts_from_text(latest_user_message):
        if facts:
            answer = "Stored for this thread: " + ", ".join(f"{key}={value}" for key, value in facts.items()) + "."
        else:
            answer = "I did not detect any structured facts to store."
    else:
        answer = answer_from_facts(latest_user_message, facts)

    return {
        "facts": facts,
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def build_persistence_graph(checkpointer):
    workflow = StateGraph(PersistenceState)
    workflow.add_node("memory_agent", persistence_node)
    workflow.add_edge(START, "memory_agent")
    workflow.add_edge("memory_agent", END)
    return workflow.compile(checkpointer=checkpointer)


def run_single_message(thread_id: str, message: str) -> str:
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_persistence_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({"messages": [HumanMessage(content=message)], "facts": {}, "answer": ""}, config)
        return result["answer"]


def read_thread_state(thread_id: str) -> dict[str, object]:
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_persistence_graph(saver)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        return {"values": snapshot.values, "next": snapshot.next}


def run_demo(thread_id: str, reset: bool = False) -> None:
    if reset:
        reset_checkpoint_db()

    first_message = (
        "Remember that my project is Buraq, my supervisor is Dr. Sana Qureshi, "
        "and my internship contact is Talent Team."
    )
    second_message = "What project am I working on and who is my supervisor?"

    print(f"Demo thread_id: {thread_id}")
    print("\nSession 1")
    print("User:", first_message)
    print("Agent:", run_single_message(thread_id, first_message))

    print("\nSession 2 (simulated restart with same thread_id)")
    print("Recovered state before the second message:")
    print(read_thread_state(thread_id))
    print("User:", second_message)
    print("Agent:", run_single_message(thread_id, second_message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate LangGraph checkpoint persistence with thread_id.")
    parser.add_argument("--thread-id", default="lab5-persistence-demo", help="Conversation thread identifier.")
    parser.add_argument("--message", help="Run a single message against an existing or new thread.")
    parser.add_argument("--show-state", action="store_true", help="Print the current persisted state for the thread.")
    parser.add_argument("--demo", action="store_true", help="Run the full two-session persistence demo.")
    parser.add_argument("--reset", action="store_true", help="Delete checkpoint_db.sqlite before running.")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_DB_PATH.exists():
        reset_checkpoint_db()

    if args.show_state:
        print(read_thread_state(args.thread_id))
        return

    if args.demo or not args.message:
        run_demo(thread_id=args.thread_id, reset=True)
        return

    print(run_single_message(args.thread_id, args.message))


if __name__ == "__main__":
    main()
