from __future__ import annotations

import base64
import csv
import mimetypes
import os
import pickle
import re
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ingest_data import ChunkRecord, ingest_chunks, query_chunks
from runtime_services import copy_local_file_to_storage, queue_scheduled_email, resolve_file_reference, save_uploaded_bytes

load_dotenv()

DATA_DIR = Path("Initial_Data")
DRAFT_CACHE_PATH = Path(".draft_cache.pkl")
REFERENCE_NOW = datetime(2026, 5, 5, 12, 0, 0)
DEFAULT_SIGNATURE = "Best regards,\nWardah Haya"
GMAIL_METADATA_HEADERS = ["Subject", "From", "To", "Date"]


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
    if os.getenv("BURAQ_DISABLE_LIVE_GMAIL", "false").lower() == "true":
        return None

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


def _direct_outbound_allowed() -> bool:
    return os.getenv("BURAQ_ALLOW_DIRECT_OUTBOUND", "false").lower() == "true"


def _normalize_gmail_timestamp(header_date: str, internal_date: str | None) -> str:
    if internal_date:
        try:
            return datetime.fromtimestamp(int(internal_date) / 1000).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    if header_date:
        try:
            parsed = parsedate_to_datetime(header_date)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    return ""


def _infer_live_category(subject: str, sender: str, snippet: str) -> str:
    text = " ".join([subject, sender, snippet]).lower()
    if any(keyword in text for keyword in ["interview", "resume", "recruit", "hiring", "talent team"]):
        return "recruitment"
    if any(keyword in text for keyword in ["meeting", "team", "project", "sprint", "standup"]):
        return "team"
    if any(keyword in text for keyword in ["deadline", "assignment", "submission", "course", "lab"]):
        return "course"
    if any(keyword in text for keyword in ["invoice", "receipt", "billing", "subscription", "admin"]):
        return "admin"
    return "general"


def _infer_live_priority(subject: str, snippet: str, label_ids: set[str]) -> str:
    text = " ".join([subject, snippet]).lower()
    urgent_keywords = [
        "urgent",
        "asap",
        "deadline",
        "due",
        "interview",
        "resume",
        "important",
        "action required",
        "today",
        "tomorrow",
    ]
    if "IMPORTANT" in label_ids or any(keyword in text for keyword in urgent_keywords):
        return "high"
    if "UNREAD" in label_ids:
        return "medium"
    return "low"


def _infer_live_action_required(subject: str, snippet: str) -> str:
    text = " ".join([subject, snippet]).lower()
    action_keywords = ["reply", "respond", "confirm", "send", "review", "submit", "upload", "schedule", "let us know"]
    return "yes" if any(keyword in text for keyword in action_keywords) else "no"


def _fetch_live_gmail_rows(label_ids: list[str], max_results: int = 20, query: str | None = None) -> list[dict[str, str]]:
    service = _safe_get_gmail_service()
    if service is None:
        return []

    try:
        request: dict[str, object] = {
            "userId": "me",
            "labelIds": label_ids,
            "maxResults": min(max_results, 100),
        }
        if query:
            request["q"] = query

        results = service.users().messages().list(**request).execute()
        messages = results.get("messages", [])
        rows: list[dict[str, str]] = []

        for message in messages:
            payload = service.users().messages().get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=GMAIL_METADATA_HEADERS,
            ).execute()
            headers = {header["name"]: header["value"] for header in payload.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "No Subject")
            sender = headers.get("From", "Unknown")
            recipient = headers.get("To", "Unknown")
            snippet = payload.get("snippet", "")
            normalized_date = _normalize_gmail_timestamp(headers.get("Date", ""), payload.get("internalDate"))
            label_set = set(payload.get("labelIds", []))

            rows.append(
                {
                    "email_id": message["id"],
                    "thread_id": payload.get("threadId", ""),
                    "from": sender,
                    "to": recipient,
                    "subject": subject,
                    "date": normalized_date,
                    "snippet": snippet,
                    "category": _infer_live_category(subject, sender, snippet),
                    "priority": _infer_live_priority(subject, snippet, label_set),
                    "action_required": _infer_live_action_required(subject, snippet),
                    "has_attachment": "yes" if "has:attachment" in snippet.lower() else "unknown",
                }
            )

        return rows
    except Exception:
        return []


def _load_available_inbox_rows(max_results: int = 20) -> tuple[list[dict[str, str]], str]:
    live_rows = _fetch_live_gmail_rows(["INBOX"], max_results=max_results)
    if live_rows:
        return live_rows, "gmail"
    return _load_inbox_rows()[:max_results], "sample"


def _format_email_row(row: dict[str, str]) -> str:
    return (
        f"From: {row.get('from', 'Unknown')}\n"
        f"Subject: {row.get('subject', 'No Subject')}\n"
        f"Date: {row.get('date', '')}\n"
        f"Category: {row.get('category', 'general')}\n"
        f"Priority: {row.get('priority', 'unknown')}\n"
        f"Action required: {row.get('action_required', 'unknown')}\n"
        f"Preview: {row.get('snippet', '')}"
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


def export_saved_draft() -> dict[str, str] | None:
    draft = _load_draft()
    if not draft:
        return None
    return dict(draft)


def clear_saved_draft() -> None:
    DRAFT_CACHE_PATH.unlink(missing_ok=True)


def _parse_schedule_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("T", " ")
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


def _build_encoded_email(to: str, subject: str, body: str, attachment_path: Path | None = None) -> str:
    if attachment_path is None:
        message = MIMEText(body)
    else:
        message = MIMEMultipart()
        message.attach(MIMEText(body))

        mime_type, _ = mimetypes.guess_type(str(attachment_path))
        content_type = mime_type or "application/octet-stream"
        maintype, subtype = content_type.split("/", 1)
        part = MIMEBase(maintype, subtype)
        part.set_payload(attachment_path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment_path.name}"')
        message.attach(part)

    message["to"] = to
    message["subject"] = subject
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


def deliver_email_message(
    to: str,
    subject: str,
    body: str,
    attachment_ref: str | None = None,
) -> tuple[bool, str]:
    service = _safe_get_gmail_service()
    if service is None:
        return (
            False,
            "Gmail credentials are not configured in this environment, so the email could not be delivered yet.",
        )

    attachment_path: Path | None = None
    if attachment_ref:
        try:
            attachment_path = resolve_file_reference(attachment_ref)
        except Exception as exc:
            return False, f"Attachment could not be resolved: {exc}"

    raw_message = _build_encoded_email(to=to, subject=subject, body=body, attachment_path=attachment_path)
    try:
        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
    except Exception as exc:
        return False, f"Gmail send failed: {_simplify_gmail_error(exc)}"

    detail = f"Email sent to {to} with subject '{subject}'."
    if attachment_path is not None:
        detail += f" Attachment: {attachment_path.name}."
    return True, detail


def _walk_message_parts(payload: dict[str, object]) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    queue = [payload]
    while queue:
        node = queue.pop()
        parts.append(node)
        for child in node.get("parts", []) or []:
            if isinstance(child, dict):
                queue.append(child)
    return parts


def _simplify_gmail_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "accessnotconfigured" in lowered or "gmail api has not been used" in lowered:
        return "The Gmail API is not enabled for the configured Google Cloud project yet."
    if "insufficient permission" in lowered or "insufficientpermissions" in lowered:
        return "The stored Gmail token does not have the required permission scope."
    return text


def _chunk_live_inbox_rows(rows: list[dict[str, str]]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for row in rows:
        body = "\n".join(
            [
                f"Subject: {row.get('subject', 'No Subject')}",
                f"From: {row.get('from', 'Unknown')}",
                f"To: {row.get('to', 'Unknown')}",
                f"Date: {row.get('date', '')}",
                f"Snippet: {row.get('snippet', '')}",
                f"Category: {row.get('category', 'general')}",
                f"Priority: {row.get('priority', 'medium')}",
                f"Action required: {row.get('action_required', 'unknown')}",
                f"Has attachment: {row.get('has_attachment', 'unknown')}",
            ]
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"gmail-live-{row.get('email_id', normalize_id(row.get('subject', 'email')))}",
                text=body,
                metadata={
                    "doc_type": "incoming_email",
                    "department": infer_email_department(row.get("category", "general")),
                    "priority_level": row.get("priority", "medium").lower(),
                    "last_updated": row.get("date", "")[:10] or REFERENCE_NOW.strftime("%Y-%m-%d"),
                    "source_file": "live_gmail_inbox",
                    "thread_id": row.get("thread_id", ""),
                    "category": row.get("category", "general").lower(),
                    "action_required": row.get("action_required", "unknown").lower(),
                },
            )
        )
    return chunks


def _chunk_live_sent_rows(rows: list[dict[str, str]]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for row in rows:
        body = "\n".join(
            [
                f"Subject: {row.get('subject', 'No Subject')}",
                f"Recipient: {row.get('to', 'Unknown')}",
                f"Date: {row.get('date', '')}",
                "Tone: professional",
                f"Body excerpt: {row.get('snippet', '')}",
                f"Sign off: {DEFAULT_SIGNATURE.splitlines()[0]}",
            ]
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"gmail-sent-{row.get('email_id', normalize_id(row.get('subject', 'sent')))}",
                text=body,
                metadata={
                    "doc_type": "sent_style_reference",
                    "department": "writing_style",
                    "priority_level": "medium",
                    "last_updated": row.get("date", "")[:10] or REFERENCE_NOW.strftime("%Y-%m-%d"),
                    "source_file": "live_gmail_sent",
                    "tone": "professional",
                    "recipient_scope": "external" if "@" in row.get("to", "") else "internal",
                },
            )
        )
    return chunks


class ReadInboxInput(ToolInput):
    max_results: int = Field(default=5, ge=1, le=20, description="Number of recent inbox emails to return.")


@tool(args_schema=ReadInboxInput)
def read_inbox(max_results: int = 5) -> str:
    """
    Read the most recent inbox messages.
    Use this when the user asks to check their inbox, see recent mail, or asks what arrived lately.
    """
    rows, source = _load_available_inbox_rows(max_results=max_results)
    if not rows:
        return "No inbox emails are available."
    prefix = "Showing live Gmail inbox messages.\n\n" if source == "gmail" else "Gmail is not connected, so showing grounded sample inbox data.\n\n"
    return prefix + "\n\n---\n\n".join(_format_email_row(row) for row in rows)


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
    live_rows = _fetch_live_gmail_rows(["INBOX"], max_results=max(max_results * 10, 25), query=query)
    search_space = live_rows
    source = "gmail" if live_rows else "sample"
    if not search_space:
        search_space = _load_inbox_rows()

    matches = []
    for row in search_space:
        haystack = " ".join([row["from"], row["subject"], row["snippet"], row["category"]])
        if _match_query(haystack, query):
            matches.append(row)
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No inbox emails matched '{query}'."
    prefix = "Showing live Gmail matches.\n\n" if source == "gmail" else "Gmail is not connected, so showing grounded sample inbox matches.\n\n"
    return prefix + "\n\n---\n\n".join(_format_email_row(row) for row in matches)


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
    rows, source = _load_available_inbox_rows(max_results=100)
    matches = [row for row in rows if row["date"].startswith(date)]
    if not matches:
        return f"No inbox emails were found for {date}."
    prefix = "Showing live Gmail inbox messages.\n\n" if source == "gmail" else "Gmail is not connected, so showing grounded sample inbox data.\n\n"
    return prefix + "\n\n---\n\n".join(_format_email_row(row) for row in matches)


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
    if not _direct_outbound_allowed():
        return "Direct outbound execution is disabled. Stage the draft through the approval workflow instead."

    draft = _load_draft()
    if not draft:
        return "No saved draft is available. Draft an email first."

    success, detail = deliver_email_message(
        to=draft["to"],
        subject=draft["subject"],
        body=draft["body"],
        attachment_ref=draft.get("attachment_ref"),
    )
    if not success:
        return detail

    clear_saved_draft()
    return detail


class SendWithAttachmentInput(ToolInput):
    to: str = Field(description="Recipient name or email address.")
    subject: str = Field(description="Subject line for the email.")
    body: str = Field(description="Body text of the email.")
    file_url: str = Field(
        description="Managed storage reference such as storage://uploads/file.pdf to attach.",
    )


@tool(args_schema=SendWithAttachmentInput)
def send_email_with_attachment(to: str, subject: str, body: str, file_url: str) -> str:
    """
    Send an email with one stored attachment.
    Use this when the user explicitly wants to send a managed-storage file along with the message.
    """
    if not _direct_outbound_allowed():
        return "Direct outbound execution is disabled. Stage the attachment send through the approval workflow instead."

    success, detail = deliver_email_message(to=to, subject=subject, body=body, attachment_ref=file_url)
    if success:
        return detail
    return detail


class ScheduleEmailInput(ToolInput):
    to: str = Field(description="Recipient name or email address.")
    subject: str = Field(description="Subject line for the email.")
    body: str = Field(description="Body text of the scheduled email.")
    send_at: str = Field(
        description="Future send time in YYYY-MM-DD HH:MM, YYYY-MM-DD HH:MM:SS, or ISO format.",
    )

    @field_validator("send_at")
    @classmethod
    def validate_send_at(cls, value: str) -> str:
        scheduled_for = _parse_schedule_timestamp(value)
        if scheduled_for <= datetime.utcnow():
            raise ValueError("send_at must be in the future")
        return scheduled_for.isoformat(timespec="seconds")


@tool(args_schema=ScheduleEmailInput)
def schedule_email(to: str, subject: str, body: str, send_at: str) -> str:
    """
    Queue an email for future delivery by the scheduler daemon.
    Use this when the user asks to send an email later at a specific time.
    """
    if not _direct_outbound_allowed():
        return "Direct outbound execution is disabled. Stage the scheduled email through the approval workflow instead."

    scheduled_for = _parse_schedule_timestamp(send_at)
    row_id = queue_scheduled_email(
        to_address=to,
        subject=subject,
        body=body,
        send_at=scheduled_for.isoformat(timespec="seconds"),
    )
    return (
        f"Scheduled email #{row_id} for {scheduled_for.isoformat(sep=' ', timespec='minutes')} "
        f"to {to} with subject '{subject}'. The scheduler will attempt delivery at that time and will mark the item "
        "failed if Gmail credentials or API access are unavailable."
    )


class UploadFileToStorageInput(ToolInput):
    file: str = Field(description="Local file path to copy into Buraq storage.")


@tool(args_schema=UploadFileToStorageInput)
def upload_file_to_storage(file: str) -> str:
    """
    Copy a local file into Buraq's managed storage and return a reusable storage reference.
    Use this before sending an attachment when the file lives on disk.
    """
    try:
        stored = copy_local_file_to_storage(file, area="uploads")
    except Exception as exc:
        return f"Upload failed: {exc}"

    return (
        f"Stored file '{stored['name']}' successfully.\n"
        f"Reference: {stored['ref']}\n"
        f"Size: {stored['size_bytes']} bytes"
    )


class IngestEmailsToVectorStoreInput(ToolInput):
    max_results: int = Field(default=25, ge=1, le=200, description="Maximum live inbox and sent emails to ingest.")
    include_sent_style: bool = Field(
        default=True,
        description="Whether to also ingest recent sent emails as writing-style references.",
    )


@tool(args_schema=IngestEmailsToVectorStoreInput)
def ingest_emails_to_vector_store(max_results: int = 25, include_sent_style: bool = True) -> str:
    """
    Refresh the grounded vector store with live Gmail emails when available.
    Use this when the user asks to sync or re-index current email data.
    """
    chunks: list[ChunkRecord] = []
    live_inbox_rows = _fetch_live_gmail_rows(["INBOX"], max_results=max_results)
    if live_inbox_rows:
        chunks.extend(_chunk_live_inbox_rows(live_inbox_rows))

    live_sent_rows: list[dict[str, str]] = []
    if include_sent_style:
        live_sent_rows = _fetch_live_gmail_rows(["SENT"], max_results=max_results)
        if live_sent_rows:
            chunks.extend(_chunk_live_sent_rows(live_sent_rows))

    if not chunks:
        try:
            from ingest_data import load_project_chunks

            ingest_chunks(load_project_chunks())
        except Exception as exc:
            return f"Vector-store refresh failed: {exc}"
        return "Live Gmail was unavailable, so the grounded sample knowledge base was re-ingested successfully."

    try:
        ingest_chunks(chunks)
    except Exception as exc:
        return f"Vector-store refresh failed: {exc}"

    return (
        f"Ingested {len(chunks)} live email chunks into the grounded vector store "
        f"({len(live_inbox_rows)} inbox, {len(live_sent_rows)} sent-style)."
    )


class DownloadAttachmentsInput(ToolInput):
    query: Optional[str] = Field(
        default=None,
        description="Optional Gmail search query to narrow which attachment emails to inspect.",
    )
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of matching emails to inspect.")


@tool(args_schema=DownloadAttachmentsInput)
def download_attachments(query: Optional[str] = None, max_results: int = 5) -> str:
    """
    Download attachments from recent Gmail messages into managed storage.
    Use this when the user asks to fetch files from their inbox.
    """
    service = _safe_get_gmail_service()
    if service is None:
        return (
            "Gmail is not connected in this environment, so live attachment download is unavailable. "
            "Use upload_file_to_storage for local files instead."
        )

    gmail_query = "has:attachment"
    if query:
        gmail_query = f"{query} has:attachment"

    try:
        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            q=gmail_query,
            maxResults=max_results,
        ).execute()
        messages = results.get("messages", [])
    except Exception as exc:
        return (
            "Attachment search could not run against live Gmail. "
            f"Reason: {_simplify_gmail_error(exc)}"
        )

    downloaded: list[str] = []
    for message in messages:
        try:
            payload = service.users().messages().get(userId="me", id=message["id"], format="full").execute()
        except Exception:
            continue

        for part in _walk_message_parts(payload.get("payload", {})):
            filename = str(part.get("filename") or "").strip()
            body = part.get("body", {}) or {}
            attachment_id = body.get("attachmentId")
            inline_data = body.get("data")

            if not filename or (not attachment_id and not inline_data):
                continue

            try:
                if attachment_id:
                    attachment = service.users().messages().attachments().get(
                        userId="me",
                        messageId=message["id"],
                        id=attachment_id,
                    ).execute()
                    raw_data = attachment.get("data", "")
                else:
                    raw_data = inline_data

                file_bytes = base64.urlsafe_b64decode(raw_data.encode("utf-8"))
                stored = save_uploaded_bytes(filename, file_bytes, area="downloads")
                downloaded.append(stored["ref"])
            except Exception:
                continue

    if not downloaded:
        return "No downloadable Gmail attachments were found for that request."

    joined_refs = "\n".join(f"- {ref}" for ref in downloaded)
    return f"Downloaded {len(downloaded)} attachment(s) into managed storage:\n{joined_refs}"


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
    rows, source = _load_available_inbox_rows(max_results=100)
    matches = [
        row for row in rows
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
        f"- Data source: {'live Gmail' if source == 'gmail' else 'grounded sample inbox'}",
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
        try:
            results = service.users().messages().list(userId="me", labelIds=["SPAM"], maxResults=max_results).execute()
            messages = results.get("messages", [])
            if not messages:
                return "No spam emails were found."
            return f"{len(messages)} spam email(s) were found in Gmail."
        except Exception:
            pass

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
    rows, source = _load_available_inbox_rows(max_results=100)
    matches = []
    for row in rows:
        timestamp = _parse_email_timestamp(row["date"])
        if timestamp < cutoff:
            continue
        if row["subject"].lower().startswith("re:") or "reply" in row["snippet"].lower():
            matches.append(row)

    if not matches:
        return f"No reply-like emails were found in the last {hours_back} hours."
    prefix = "Showing live Gmail replies.\n\n" if source == "gmail" else "Gmail is not connected, so showing grounded sample inbox replies.\n\n"
    return prefix + "\n\n---\n\n".join(_format_email_row(row) for row in matches)


class ImportantAlertsInput(ToolInput):
    max_results: int = Field(default=10, ge=1, le=20, description="Maximum number of urgent items to return.")


@tool(args_schema=ImportantAlertsInput)
def check_important_alerts(max_results: int = 10) -> str:
    """
    Surface urgent inbox items and imminent deadlines that need attention.
    Use this when the user asks what is urgent, what needs attention, or what deadlines are coming up.
    """
    rows, source = _load_available_inbox_rows(max_results=100)
    urgent_emails = [
        row for row in rows
        if row["priority"] == "high" or row["action_required"] == "yes"
    ]
    upcoming_deadlines = [
        row for row in _load_deadline_rows()
        if _parse_due_timestamp(row["due_date"]) <= REFERENCE_NOW + timedelta(days=3)
        and row["status"] != "completed"
    ]

    lines = ["Important alerts:", f"- Email source: {'live Gmail' if source == 'gmail' else 'grounded sample inbox'}"]
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
    send_email_with_attachment,
    schedule_email,
    daily_email_summary,
    check_spam,
    check_replies,
    check_important_alerts,
    search_knowledge_base,
    ingest_emails_to_vector_store,
    download_attachments,
    upload_file_to_storage,
]
