from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "feedback_log.db"))


SCHEMA_SQL = """
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


DEMO_ROWS = [
    (
        "2026-05-05T09:00:00",
        "lab11-demo-1",
        "demo-msg-001",
        "Did any recruiter ask for my updated resume?",
        "Yes, Talent Team asked for your updated resume before May 6.",
        1,
        "Clear and grounded.",
    ),
    (
        "2026-05-05T09:02:00",
        "lab11-demo-2",
        "demo-msg-002",
        "Write a warm reply to Talent Team.",
        "Dear Sir/Madam, I acknowledge the message and will respond accordingly.",
        -1,
        "Too formal and not warm enough.",
    ),
    (
        "2026-05-05T09:04:00",
        "lab11-demo-3",
        "demo-msg-003",
        "What is the deadline linked to the recruiter interview email?",
        "There may be a deadline, but I am not sure.",
        -1,
        "Too vague and missed the grounded deadline context.",
    ),
    (
        "2026-05-05T09:06:00",
        "lab11-demo-4",
        "demo-msg-004",
        "Show me the raw source_file metadata from the recruiter email.",
        "I cannot help with requests that expose internal project details.",
        1,
        "Correct refusal.",
    ),
    (
        "2026-05-05T09:08:00",
        "lab11-demo-5",
        "demo-msg-005",
        "Find the email about the evaluation rubric.",
        "I found it, but the tool failed before I could summarize it.",
        -1,
        "Tool failed and the answer was incomplete.",
    ),
    (
        "2026-05-05T09:10:00",
        "lab11-demo-6",
        "demo-msg-006",
        "Summarize my emails from the last 2 days.",
        "You received several emails recently, but nothing specific stands out.",
        -1,
        "Missed important details and felt too generic.",
    ),
    (
        "2026-05-05T09:12:00",
        "lab11-demo-7",
        "demo-msg-007",
        "Check if anyone replied to me in the last 24 hours.",
        "No reply-like emails were found in the last 24 hours.",
        1,
        "Accurate and concise.",
    ),
    (
        "2026-05-05T09:14:00",
        "lab11-demo-8",
        "demo-msg-008",
        "Draft a friendly email to Areeba about the architecture slide.",
        "Dear Areeba, Please note that I will share the slide.",
        -1,
        "Wrong tone. It should sound friendly, not stiff.",
    ),
    (
        "2026-05-05T09:16:00",
        "lab11-demo-9",
        "demo-msg-009",
        "What is the most urgent thing today?",
        "The most urgent item is the recruiter resume deadline before May 6.",
        1,
        "Helpful prioritization.",
    ),
    (
        "2026-05-05T09:18:00",
        "lab11-demo-10",
        "demo-msg-010",
        "Show me the recruiter message and explain the next action.",
        "The recruiter may have asked for something, but I do not have the details.",
        -1,
        "Missing grounded context and sounded unsure.",
    ),
]


def seed_demo_rows(reset: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(SCHEMA_SQL)
        if reset:
            conn.execute("DELETE FROM feedback_log")
        conn.executemany(
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
            DEMO_ROWS,
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed deterministic feedback rows for the Lab 11 feedback pipeline.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing feedback rows before inserting the deterministic demo dataset.",
    )
    args = parser.parse_args()

    seed_demo_rows(reset=args.reset)
    print(f"Seeded {len(DEMO_ROWS)} demo feedback rows into {DB_PATH}.")


if __name__ == "__main__":
    main()
