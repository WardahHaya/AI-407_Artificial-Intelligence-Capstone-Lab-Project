from __future__ import annotations

import base64
import csv
import os
import pickle
import re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ingest_data import query_chunks

load_dotenv()

DATA_DIR = Path("Initial_Data")
DRAFT_CACHE_PATH = Path(".draft_cache.pkl")
REFERENCE_NOW = datetime(2026, 5, 5, 12, 0, 0)
DEFAULT_SIGNATURE = "Best regards,\nWardah Haya"


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _parse_email_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def _parse_due_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


@lru_cache(maxsize=1)
def _load_inbox_rows() -> list[dict[str, str]]:
    with (DATA_DIR / "inbox_emails_sample.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: _parse_email_timestamp(row["date"]), reverse=True)
    return rows


@lru_cache(maxsize=1)
def _load_deadline_rows() -> list[dict[str, str]]:
    with (DATA_DIR / "project_deadlines.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: _parse_due_timestamp(row["due_date"]))
    return rows


@lru_cache(maxsize=1)
def _load_style_rows() -> list[dict[str, str]]:
    with (DATA_DIR / "sent_emails_style_reference.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_get_gmail_service():
    """
    Returns a Gmail service only when credentials are already configured.
    This avoids forcing a browser auth flow during lab verification.
    """
    token_path = Path("token.pickle")
    creds_path = Path(os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json"))
    if not token_path.exists() or not creds_path.exists():
        return None

    try:
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds = None
        with token_path.open("rb") as handle:
            creds = pickle.load(handle)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with token_path.open("wb") as handle:
                pickle.dump(creds, handle)

        if not creds or not creds.valid:
            return None

        return build("gmail", "v1", credentials=creds)
    except Exception:
        return None


def _format_email_row(row: dict[str, str]) -> str:
    return (
        f"From: {row['from']}\n"
        f"Subject: {row['subject']}\n"
        f"Date: {row['date']}\n"
        f"Category: {row['category']}\n"
        f"Priority: {row['priority']}\n"
        f"Action required: {row['action_required']}\n"
        f"Preview: {row['snippet']}"
    )


def _format_deadline_row(row: dict[str, str]) -> str:
    return (
        f"Title: {row['title']}\n"
        f"Due date: {row['due_date']}\n"
        f"Owner: {row['owner']}\n"
        f"Status: {row['status']}\n"
        f"Details: {row['details']}"
    )


def _match_query(text: str, query: str) -> bool:
    haystack = text.lower()
    query = query.lower().strip()
    if not query:
        return False
    if query in haystack:
        return True
    tokens = [token for token in re.split(r"\s+", query) if token]
    return all(token in haystack for token in tokens)


def _resolve_window(date_text: Optional[str]) -> tuple[datetime, datetime, str]:
    if not date_text or date_text.lower() in {"today", ""}:
        start = REFERENCE_NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, start.strftime("%Y-%m-%d")

    normalized = date_text.lower().strip()
    if normalized == "yesterday":
        end = REFERENCE_NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
        return start, end, start.strftime("%Y-%m-%d")

    if any(keyword in normalized for keyword in ["last", "past", "recent"]):
        match = re.search(r"(\d+)", normalized)
        days = int(match.group(1)) if match else 1
        end = REFERENCE_NOW + timedelta(minutes=1)
        start = (REFERENCE_NOW - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, end, f"last {days} day(s)"

    exact = datetime.strptime(date_text, "%Y-%m-%d")
    return exact, exact + timedelta(days=1), exact.strftime("%Y-%m-%d")


def _retrieve_style_examples(context: str, top_k: int = 2) -> list[str]:
    try:
        matches = query_chunks(
            query=context,
            top_k=top_k,
            where={"doc_type": "sent_style_reference"},
        )
    except Exception:
        return []
    examples: list[str] = []
    for match in matches:
        examples.append(match["document"])
    return examples


def _build_draft_body(to: str, subject: str, context: str, tone: str) -> str:
    style_examples = _retrieve_style_examples(f"{subject}\n{context}")
    recipient_name = to.split("<")[0].strip() if "<" in to else to.split("@")[0].strip()
    greeting = {
        "formal": f"Dear {recipient_name},",
        "friendly": f"Hi {recipient_name},",
        "casual": f"Hello {recipient_name},",
        "professional": f"Hi {recipient_name},",
    }.get(tone, f"Hi {recipient_name},")

    body_intro = {
        "formal": "I hope you are doing well.",
        "friendly": "I hope you're doing well.",
        "casual": "Hope you're doing well.",
        "professional": "I hope you are doing well.",
    }.get(tone, "I hope you are doing well.")

    style_hint = ""
    if style_examples:
        first_example = style_examples[0]
        style_hint = (
            "\n\nStyle reference used:\n"
            f"{first_example.splitlines()[0]}\n"
            f"{first_example.splitlines()[-2] if len(first_example.splitlines()) >= 2 else ''}"
        )

    return (
        f"{greeting}\n\n"
        f"{body_intro}\n\n"
        f"{context.strip()}\n\n"
        f"{DEFAULT_SIGNATURE}"
        f"{style_hint}"
    ).strip()


def _save_draft(draft: dict[str, str]) -> None:
    with DRAFT_CACHE_PATH.open("wb") as handle:
        pickle.dump(draft, handle)


def _load_draft() -> dict[str, str] | None:
    if not DRAFT_CACHE_PATH.exists():
        return None
    with DRAFT_CACHE_PATH.open("rb") as handle:
        return pickle.load(handle)


class ReadInboxInput(ToolInput):
    max_results: int = Field(default=5, ge=1, le=20, description="Number of recent inbox emails to return.")


@tool(args_schema=ReadInboxInput)
def read_inbox(max_results: int = 5) -> str:
    """
    Read the most recent inbox messages.
    Use this when the user asks to check their inbox, see recent mail, or asks what arrived lately.
    """
    rows = _load_inbox_rows()[:max_results]
    if not rows:
        return "No inbox emails are available in the grounded sample data."
    return "\n\n---\n\n".join(_format_email_row(row) for row in rows)


class SearchEmailInput(ToolInput):
    query: str = Field(description="Keyword or phrase to search inside sender, subject, or preview text.")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of matching emails to return.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


@tool(args_schema=SearchEmailInput)
def search_emails(query: str, max_results: int = 5) -> str:
    """
    Search grounded inbox emails by sender, subject, or message preview.
    Use this when the user wants to find a specific email or keyword in their inbox.
    """
    matches = []
    for row in _load_inbox_rows():
        haystack = " ".join([row["from"], row["subject"], row["snippet"], row["category"]])
        if _match_query(haystack, query):
            matches.append(row)
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No inbox emails matched '{query}'."
    return "\n\n---\n\n".join(_format_email_row(row) for row in matches)


class FetchByDateInput(ToolInput):
    date: str = Field(description="Date in YYYY-MM-DD format.")

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


@tool(args_schema=FetchByDateInput)
def fetch_emails_by_date(date: str) -> str:
    """
    Fetch inbox emails received on one specific date.
    Use this when the user asks for emails from a particular day.
    """
    matches = [row for row in _load_inbox_rows() if row["date"].startswith(date)]
    if not matches:
        return f"No inbox emails were found for {date}."
    return "\n\n---\n\n".join(_format_email_row(row) for row in matches)


class DraftEmailInput(ToolInput):
    to: str = Field(description="Recipient name or email address.")
    subject: str = Field(description="Subject line for the email draft.")
    context: str = Field(description="Instructions describing what the email should say.")
    tone: Literal["professional", "friendly", "formal", "casual"] = Field(
        default="professional",
        description="Writing tone for the draft.",
    )


@tool(args_schema=DraftEmailInput)
def draft_email(to: str, subject: str, context: str, tone: str = "professional") -> str:
    """
    Draft a complete email and store it for later approval.
    Use this when the user asks to write or compose an email. Draft first, then wait for explicit send approval.
    """
    body = _build_draft_body(to=to, subject=subject, context=context, tone=tone)
    draft = {"to": to, "subject": subject, "body": body, "tone": tone}
    _save_draft(draft)

    return (
        f"Draft Email\n"
        f"{'=' * 40}\n"
        f"To: {to}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n"
        f"{'=' * 40}\n"
        "Wait for the user's explicit approval before sending."
    )


class SendReviewedInput(ToolInput):
    confirmed: bool = Field(description="Set to true only when the user has explicitly approved sending the draft.")


@tool(args_schema=SendReviewedInput)
def send_reviewed_email(confirmed: bool) -> str:
    """
    Send the previously drafted email only after explicit user approval.
    Use this only when the user clearly says to send the draft.
    """
    if not confirmed:
        return "The draft was not sent because explicit approval was not confirmed."

    draft = _load_draft()
    if not draft:
        return "No saved draft is available. Draft an email first."

    service = _safe_get_gmail_service()
    if service is None:
        return (
            "Gmail credentials are not configured in this lab environment, so the draft was kept for review "
            f"instead of being sent.\nTo: {draft['to']}\nSubject: {draft['subject']}"
        )

    message = MIMEText(draft["body"])
    message["to"] = draft["to"]
    message["subject"] = draft["subject"]
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except Exception as exc:
        return (
            "Gmail authentication exists, but sending failed. "
            "If you authenticated with an older read-only token, delete token.pickle and run "
            "python vector_store\\ingest.py again so Google grants both read and send access.\n"
            f"Details: {exc}"
        )

    DRAFT_CACHE_PATH.unlink(missing_ok=True)
    return f"Email sent to {draft['to']} with subject '{draft['subject']}'."


class DailySummaryInput(ToolInput):
    date: Optional[str] = Field(
        default=None,
        description="One of: today, yesterday, last N days, or a date in YYYY-MM-DD format.",
    )


@tool(args_schema=DailySummaryInput)
def daily_email_summary(date: Optional[str] = None) -> str:
    """
    Summarize grounded inbox emails for a date or short time window.
    Use this when the user asks for a daily digest or summary of recent emails.
    """
    start, end, label = _resolve_window(date)
    matches = [
        row for row in _load_inbox_rows()
        if start <= _parse_email_timestamp(row["date"]) < end
    ]

    if not matches:
        return f"No inbox emails were found for {label}."

    urgent = [row for row in matches if row["priority"] == "high" or row["action_required"] == "yes"]
    summaries = [
        f"- {row['subject']} | from {row['from']} | priority={row['priority']} | action_required={row['action_required']}"
        for row in matches
    ]

    result_lines = [
        f"Email summary for {label}:",
        f"- Total emails: {len(matches)}",
        f"- Urgent or action-required emails: {len(urgent)}",
        "",
        "Highlights:",
        *summaries,
    ]
    return "\n".join(result_lines)


class CheckSpamInput(ToolInput):
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum spam emails to inspect.")


@tool(args_schema=CheckSpamInput)
def check_spam(max_results: int = 10) -> str:
    """
    Check for spam or suspicious emails.
    Use this when the user asks about spam, junk mail, or suspicious inbox items.
    """
    service = _safe_get_gmail_service()
    if service is not None:
        results = service.users().messages().list(userId="me", labelIds=["SPAM"], maxResults=max_results).execute()
        messages = results.get("messages", [])
        if not messages:
            return "No spam emails were found."
        return f"{len(messages)} spam email(s) were found in Gmail."

    return "No spam emails are present in the grounded sample inbox."


class CheckRepliesInput(ToolInput):
    hours_back: int = Field(default=24, ge=1, le=720, description="How many hours back to check for replies.")


@tool(args_schema=CheckRepliesInput)
def check_replies(hours_back: int = 24) -> str:
    """
    Check whether any recent inbox messages look like replies.
    Use this when the user asks if anyone has replied lately.
    """
    cutoff = REFERENCE_NOW - timedelta(hours=hours_back)
    matches = []
    for row in _load_inbox_rows():
        timestamp = _parse_email_timestamp(row["date"])
        if timestamp < cutoff:
            continue
        if row["subject"].lower().startswith("re:") or "reply" in row["snippet"].lower():
            matches.append(row)

    if not matches:
        return f"No reply-like emails were found in the last {hours_back} hours."
    return "\n\n---\n\n".join(_format_email_row(row) for row in matches)


class ImportantAlertsInput(ToolInput):
    max_results: int = Field(default=10, ge=1, le=20, description="Maximum number of urgent items to return.")


@tool(args_schema=ImportantAlertsInput)
def check_important_alerts(max_results: int = 10) -> str:
    """
    Surface urgent inbox items and imminent deadlines that need attention.
    Use this when the user asks what is urgent, what needs attention, or what deadlines are coming up.
    """
    urgent_emails = [
        row for row in _load_inbox_rows()
        if row["priority"] == "high" or row["action_required"] == "yes"
    ]
    upcoming_deadlines = [
        row for row in _load_deadline_rows()
        if _parse_due_timestamp(row["due_date"]) <= REFERENCE_NOW + timedelta(days=3)
        and row["status"] != "completed"
    ]

    lines = ["Important alerts:"]
    for row in urgent_emails[:max_results]:
        lines.append(
            f"- Email | {row['subject']} | from {row['from']} | priority={row['priority']} | action_required={row['action_required']}"
        )
    for row in upcoming_deadlines[:max_results]:
        lines.append(
            f"- Deadline | {row['title']} | due {row['due_date']} | status={row['status']}"
        )

    if len(lines) == 1:
        return "No urgent emails or upcoming deadlines were found."
    return "\n".join(lines)


class SearchKnowledgeBaseInput(ToolInput):
    query: str = Field(description="Natural-language question for the grounded project knowledge base.")
    doc_type: Optional[Literal["incoming_email", "sent_style_reference", "deadline_record", "course_brief", "meeting_notes"]] = Field(
        default=None,
        description="Optional document-type filter for more precise retrieval.",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optional department filter such as careers, project_team, academics, or task_management.",
    )
    priority_level: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Optional priority-level filter.",
    )
    top_k: int = Field(default=3, ge=1, le=5, description="Maximum number of retrieved chunks to return.")

    @field_validator("query")
    @classmethod
    def validate_kb_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


@tool(args_schema=SearchKnowledgeBaseInput)
def search_knowledge_base(
    query: str,
    doc_type: Optional[str] = None,
    department: Optional[str] = None,
    priority_level: Optional[str] = None,
    top_k: int = 3,
) -> str:
    """
    Search the grounded Lab 2 vector database.
    Use this when the user asks about old emails, deadlines, meeting notes, or project facts that live in source memory.
    """
    filters: list[dict[str, str]] = []
    if doc_type:
        filters.append({"doc_type": doc_type})
    if department:
        filters.append({"department": department})
    if priority_level:
        filters.append({"priority_level": priority_level})

    where: dict[str, object] | None
    if not filters:
        where = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {"$and": filters}

    try:
        matches = query_chunks(query=query, top_k=top_k, where=where)
    except Exception as exc:
        return (
            "The grounded knowledge base is not ready yet. "
            "Run ingest_data.py from Lab 2 before using this tool.\n"
            f"Details: {exc}"
        )
    if not matches:
        return f"No grounded knowledge base matches were found for '{query}'."

    formatted_matches = []
    for match in matches:
        metadata = match["metadata"]
        formatted_matches.append(
            f"ID: {match['id']}\n"
            f"doc_type: {metadata['doc_type']}\n"
            f"department: {metadata['department']}\n"
            f"priority_level: {metadata['priority_level']}\n"
            f"source_file: {metadata['source_file']}\n"
            f"content:\n{match['document']}"
        )
    return "\n\n---\n\n".join(formatted_matches)


ALL_TOOLS = [
    read_inbox,
    search_emails,
    fetch_emails_by_date,
    draft_email,
    send_reviewed_email,
    daily_email_summary,
    check_spam,
    check_replies,
    check_important_alerts,
    search_knowledge_base,
]
