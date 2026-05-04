from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools import DRAFT_CACHE_PATH, draft_email, send_reviewed_email

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


class ApprovalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    email_request: dict[str, str]
    pending_email: Optional[dict[str, str]]
    approval_status: Literal["pending", "proceed", "cancel"]
    action_result: str
    final_answer: str


def parse_draft_output(output: str) -> dict[str, str]:
    lines = output.splitlines()
    to_line = next(line for line in lines if line.startswith("To: "))
    subject_line = next(line for line in lines if line.startswith("Subject: "))

    subject_index = lines.index(subject_line)
    separator_index = next(
        index for index in range(subject_index + 1, len(lines)) if lines[index].startswith("=" * 10)
    )
    body_lines = lines[subject_index + 2:separator_index]
    while body_lines and body_lines[0] == "":
        body_lines = body_lines[1:]
    while body_lines and body_lines[-1] == "":
        body_lines = body_lines[:-1]

    return {
        "to": to_line.replace("To: ", "", 1).strip(),
        "subject": subject_line.replace("Subject: ", "", 1).strip(),
        "body": "\n".join(body_lines).strip(),
    }


def build_review_message(pending_email: dict[str, str]) -> str:
    return (
        "Safety pause: the graph stopped before the high-risk send action.\n\n"
        f"To: {pending_email['to']}\n"
        f"Subject: {pending_email['subject']}\n\n"
        f"{pending_email['body']}\n\n"
        "Use Proceed to continue, Cancel to abort, or edit the pending_email body before resuming."
    )


def prepare_email_node(state: ApprovalState) -> ApprovalState:
    request = state["email_request"]
    draft_output = draft_email.invoke(request)
    pending_email = parse_draft_output(draft_output)
    review_message = build_review_message(pending_email)
    return {
        "pending_email": pending_email,
        "approval_status": "pending",
        "action_result": "",
        "final_answer": review_message,
        "messages": [AIMessage(content=review_message)],
    }


def execute_project_send_from_state(pending_email: dict[str, str]) -> str:
    with DRAFT_CACHE_PATH.open("wb") as handle:
        pickle.dump(
            {"to": pending_email["to"], "subject": pending_email["subject"], "body": pending_email["body"]},
            handle,
        )

    tool_result = send_reviewed_email.invoke({"confirmed": True})
    return (
        f"{tool_result}\n\n"
        "Final approved body taken from checkpointed state:\n"
        f"{pending_email['body']}"
    )


def send_email_node(state: ApprovalState) -> ApprovalState:
    pending_email = state.get("pending_email")
    if not pending_email:
        message = "No pending email was available to send."
        return {"action_result": message, "final_answer": message, "messages": [AIMessage(content=message)]}

    if state.get("approval_status") == "cancel":
        message = "The pending email was cancelled by the human reviewer and was not sent."
        return {"action_result": message, "final_answer": message, "messages": [AIMessage(content=message)]}

    if state.get("approval_status") != "proceed":
        message = "Send action is blocked until the reviewer chooses Proceed or Cancel."
        return {"action_result": message, "final_answer": message, "messages": [AIMessage(content=message)]}

    message = execute_project_send_from_state(pending_email)
    return {"action_result": message, "final_answer": message, "messages": [AIMessage(content=message)]}


def finalize_node(state: ApprovalState) -> ApprovalState:
    summary = state.get("final_answer") or state.get("action_result") or "Approval flow finished."
    return {"messages": [AIMessage(content=summary)], "final_answer": summary}


def build_approval_graph(checkpointer):
    workflow = StateGraph(ApprovalState)
    workflow.add_node("prepare_email", prepare_email_node)
    workflow.add_node("send_email", send_email_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_edge(START, "prepare_email")
    workflow.add_edge("prepare_email", "send_email")
    workflow.add_edge("send_email", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["send_email"])


def get_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def start_approval_flow(thread_id: str, request: dict[str, str]) -> dict[str, object]:
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_approval_graph(saver)
        config = get_config(thread_id)
        graph.invoke(
            {
                "messages": [],
                "email_request": request,
                "pending_email": None,
                "approval_status": "pending",
                "action_result": "",
                "final_answer": "",
            },
            config,
        )
        snapshot = graph.get_state(config)
        return {"values": snapshot.values, "next": snapshot.next}


def inspect_pending_state(thread_id: str) -> dict[str, object]:
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_approval_graph(saver)
        snapshot = graph.get_state(get_config(thread_id))
        return {"values": snapshot.values, "next": snapshot.next}


def validate_resumable_state(snapshot) -> str | None:
    values = snapshot.values or {}
    next_nodes = tuple(snapshot.next or ())

    if not values:
        return (
            "No paused approval state was found for this thread. "
            "Start a new approval flow before using Proceed or Cancel."
        )

    if "send_email" not in next_nodes:
        return (
            "This thread is not currently paused at the send approval step. "
            "Use --show-state to inspect the saved checkpoint or start a new approval flow."
        )

    if not values.get("pending_email"):
        return (
            "The paused checkpoint does not contain a pending email draft, "
            "so there is nothing to approve or cancel."
        )

    return None


def apply_human_decision(
    thread_id: str,
    decision: Literal["proceed", "cancel"],
    edited_body: str | None = None,
) -> str:
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        graph = build_approval_graph(saver)
        config = get_config(thread_id)
        snapshot = graph.get_state(config)
        validation_error = validate_resumable_state(snapshot)
        if validation_error:
            return validation_error

        values = snapshot.values
        pending_email = dict(values["pending_email"]) if values.get("pending_email") else None

        if pending_email and edited_body:
            pending_email["body"] = edited_body

        graph.update_state(
            config,
            {
                "approval_status": decision,
                "pending_email": pending_email,
            },
        )
        result = graph.invoke(None, config)
        return result["final_answer"]


def run_demo(thread_id: str, reset: bool = False) -> None:
    if reset:
        reset_checkpoint_db()

    request = {
        "to": "Talent Team <careers@neuralbridge.ai>",
        "subject": "Updated resume for interview",
        "context": (
            "Thank them for the update, confirm that I will send my updated resume tonight, "
            "and mention that I am available for the interview."
        ),
        "tone": "professional",
    }

    print(f"Approval demo thread_id: {thread_id}")
    paused = start_approval_flow(thread_id, request)
    print("\nState at safety breakpoint:")
    print(paused["values"])
    print("Next node:", paused["next"])

    edited_body = (
        "Hi Talent Team,\n\n"
        "Thank you for the update. I will send my updated resume tonight and I am available "
        "for the interview at your convenience.\n\n"
        "Best regards,\nWardah Haya"
    )
    print("\nHuman edit applied to pending_email.body before proceeding.")
    print(apply_human_decision(thread_id, decision="proceed", edited_body=edited_body))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate HITL approval with interrupt_before on send_email.")
    parser.add_argument("--thread-id", default="lab5-approval-demo", help="Approval workflow thread identifier.")
    parser.add_argument("--demo", action="store_true", help="Run the full approval/edit/resume demonstration.")
    parser.add_argument("--reset", action="store_true", help="Delete checkpoint_db.sqlite before running.")
    parser.add_argument("--show-state", action="store_true", help="Show the currently paused state for the thread.")
    parser.add_argument("--start", action="store_true", help="Start a new approval flow with the default request.")
    parser.add_argument("--proceed", action="store_true", help="Approve the paused send action and resume the graph.")
    parser.add_argument("--cancel", action="store_true", help="Cancel the paused send action and resume the graph.")
    parser.add_argument("--edit-body", help="Optional replacement email body to write into state before resuming.")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_DB_PATH.exists():
        reset_checkpoint_db()

    default_request = {
        "to": "Talent Team <careers@neuralbridge.ai>",
        "subject": "Updated resume for interview",
        "context": (
            "Thank them for the update, confirm that I will send my updated resume tonight, "
            "and mention that I am available for the interview."
        ),
        "tone": "professional",
    }

    if args.show_state:
        print(inspect_pending_state(args.thread_id))
        return

    if args.start:
        print(start_approval_flow(args.thread_id, default_request))
        return

    if args.proceed:
        print(apply_human_decision(args.thread_id, decision="proceed", edited_body=args.edit_body))
        return

    if args.cancel:
        print(apply_human_decision(args.thread_id, decision="cancel", edited_body=args.edit_body))
        return

    run_demo(thread_id=args.thread_id, reset=True)


if __name__ == "__main__":
    main()
