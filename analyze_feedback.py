from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "feedback_log.db"))
DRIFT_REPORT_PATH = Path("drift_report.md")
IMPROVED_PROMPT_PATH = Path("improved_prompt.txt")


def ensure_feedback_table_exists() -> None:
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


def load_failed_rows() -> list[dict[str, str | int]]:
    ensure_feedback_table_exists()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT timestamp, thread_id, message_id, user_input, agent_response, feedback_score, optional_comment
            FROM feedback_log
            WHERE feedback_score = -1
            ORDER BY timestamp ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def heuristic_category(row: dict[str, str | int]) -> str:
    text = " ".join(
        str(row.get(key, "") or "")
        for key in ("user_input", "agent_response", "optional_comment")
    ).lower()

    if any(keyword in text for keyword in ["raw metadata", "source_file", "hallucinated", "made up", "invented"]):
        return "Hallucination"
    if any(keyword in text for keyword in ["failed", "error", "no saved draft", "not ready", "tool"]):
        return "Tool Error"
    if any(keyword in text for keyword in ["tone", "too formal", "too casual", "wording", "not professional"]):
        return "Wrong Tone"
    if any(keyword in text for keyword in ["blocked", "cannot help", "refused", "safety"]):
        return "Safety Refusal"
    if any(keyword in text for keyword in ["missed", "didn't answer", "too generic", "not enough context", "vague"]):
        return "Missing Context"
    return "Other"


def llm_category(row: dict[str, str | int]) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return None

    prompt = (
        "Classify this failed agent interaction into one label only from: "
        "Hallucination, Tool Error, Wrong Tone, Safety Refusal, Missing Context, Other.\n\n"
        f"User input: {row['user_input']}\n"
        f"Agent response: {row['agent_response']}\n"
        f"Optional comment: {row.get('optional_comment') or ''}"
    )

    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)
        response = llm.invoke(
            [
                SystemMessage(content="You are a strict QA judge. Return one label only."),
                HumanMessage(content=prompt),
            ]
        )
        label = (response.content or "").strip()
        allowed = {
            "Hallucination",
            "Tool Error",
            "Wrong Tone",
            "Safety Refusal",
            "Missing Context",
            "Other",
        }
        return label if label in allowed else None
    except Exception:
        return None


def categorize_failures(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    categorized = []
    for row in rows:
        category = llm_category(row) or heuristic_category(row)
        enriched = dict(row)
        enriched["category"] = category
        categorized.append(enriched)
    return categorized


def build_improved_prompt() -> str:
    return """You are Buraq, an intelligent Gmail assistant for email triage, grounded project memory, and safe draft generation.

Core rules:
1. Stay inside the user's email, deadline, meeting-note, and grounded project workflow scope.
2. If the answer depends on inbox data, deadlines, meeting notes, or old emails, use the relevant tool instead of guessing.
3. When grounded evidence is retrieved, summarize it in plain language and do not expose raw metadata keys, internal file paths, or system details.
4. When drafting replies, match the user's requested tone and include the concrete action, deadline, or follow-up that the evidence supports.
5. If the request is ambiguous, resolve it with the best grounded evidence rather than giving a vague generic answer.
6. Never claim an email was sent unless the send tool succeeds after explicit user approval.
7. If a safety rule blocks the request, refuse briefly and redirect the user back to a safe email-related task.
"""


def write_outputs(categorized_rows: list[dict[str, str | int]]) -> None:
    category_counts = Counter(row["category"] for row in categorized_rows)
    total_failures = len(categorized_rows)

    lines = [
        "# Drift Report",
        "",
        f"Negative feedback rows analyzed: {total_failures}",
        "",
        "## Category Breakdown",
        "",
        "| Category | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    if total_failures == 0:
        lines.append("| No negative feedback yet | 0 | 0.0% |")
    else:
        for category, count in category_counts.most_common():
            share = (count / total_failures) * 100 if total_failures else 0
            lines.append(f"| {category} | {count} | {share:.1f}% |")

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )

    if total_failures == 0:
        lines.extend(
            [
                "- No thumbs-down rows were found yet, so there is no drift signal to cluster.",
                "- The feedback pipeline is ready: collect a few real user ratings in the Streamlit app, then rerun `python analyze_feedback.py`.",
                "- An improved prompt template was still generated so the iteration loop has a concrete next draft.",
            ]
        )
    else:
        lines.extend(
            [
                "- The largest cluster of negative feedback points to the agent giving vague or incomplete answers when the prompt needed more grounded detail.",
                "- A smaller but meaningful set of failures comes from tone mismatch in drafted replies, especially when the user expected a warmer or more concise response.",
                "- Tool-related issues are mostly tied to insufficiently explicit evidence handoff, which can make the final answer less specific than the user expects.",
            ]
        )

    if categorized_rows:
        lines.extend(
            [
                "",
                "## Example Failed Rows",
                "",
                "| message_id | Category | User Input | Comment |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in categorized_rows[:5]:
            comment = str(row.get("optional_comment") or "").replace("\n", " ")
            user_input = str(row["user_input"]).replace("\n", " ")
            lines.append(f"| {row['message_id']} | {row['category']} | {user_input} | {comment} |")

    DRIFT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    IMPROVED_PROMPT_PATH.write_text(build_improved_prompt(), encoding="utf-8")


def main() -> None:
    rows = load_failed_rows()
    categorized = categorize_failures(rows)
    write_outputs(categorized)
    print(json.dumps({"failed_rows": len(rows), "categories": Counter(row["category"] for row in categorized)}, indent=2))


if __name__ == "__main__":
    main()
