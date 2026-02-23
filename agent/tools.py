# agent/tools.py
# All Gmail tools for Buraq agent
# Each tool uses @tool decorator + Pydantic input validation
# The LLM reads the docstrings to decide which tool to use

import os
import re
import pickle
import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from dotenv import load_dotenv

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from groq import Groq

# Import our Lab 2 grounding tool
from vector_store.search import search_knowledge_base as kb_search

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# ─────────────────────────────────────────
#  Gmail Auth Helper
# ─────────────────────────────────────────
def get_gmail_service():
    creds = None
    token_path = "token.pickle"
    creds_path = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json")

    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def get_llm():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ══════════════════════════════════════════
#  TOOL 1 — Read Inbox
# ══════════════════════════════════════════
class ReadInboxInput(BaseModel):
    max_results: int = Field(
        default=5, ge=1, le=20,
        description="Number of emails to fetch"
    )

@tool(args_schema=ReadInboxInput)
def read_inbox(max_results: int = 5) -> str:
    """
    Reads the user's Gmail inbox and returns recent emails.
    Use this when the user asks to check their email, see
    their inbox, or asks what emails they have received.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return "Your inbox is empty."

    output = []
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        output.append(
            f"From: {headers.get('From', 'Unknown')}\n"
            f"Subject: {headers.get('Subject', 'No Subject')}\n"
            f"Date: {headers.get('Date', '')}\n"
            f"Preview: {data.get('snippet', '')[:100]}\n"
        )
    return "\n---\n".join(output)


# ══════════════════════════════════════════
#  TOOL 2 — Search Emails
# ══════════════════════════════════════════
class SearchEmailInput(BaseModel):
    query: str = Field(
        description="Search query e.g. 'from:ahmed@gmail.com' or 'subject:invoice'"
    )
    max_results: int = Field(default=5, ge=1, le=20)

@tool(args_schema=SearchEmailInput)
def search_emails(query: str, max_results: int = 5) -> str:
    """
    Searches Gmail using a query and returns matching emails.
    Use this when the user wants to find a specific email
    by sender, subject, or keyword.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return f"No emails found for: '{query}'"

    output = []
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        output.append(
            f"From: {headers.get('From', 'Unknown')}\n"
            f"Subject: {headers.get('Subject', 'No Subject')}\n"
            f"Preview: {data.get('snippet', '')[:100]}\n"
        )
    return "\n---\n".join(output)


# ══════════════════════════════════════════
#  TOOL 3 — Fetch Emails by Date
# ══════════════════════════════════════════
class FetchByDateInput(BaseModel):
    date: str = Field(
        description="Date in YYYY-MM-DD format e.g. '2025-06-01'"
    )

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD format")
        return v

@tool(args_schema=FetchByDateInput)
def fetch_emails_by_date(date: str) -> str:
    """
    Fetches all emails received on a specific date.
    Use this when the user asks for emails from a
    particular day e.g. 'show emails from June 1st'.
    """
    service = get_gmail_service()
    dt = datetime.strptime(date, "%Y-%m-%d")
    after = int(dt.timestamp())
    before = int((dt + timedelta(days=1)).timestamp())

    results = service.users().messages().list(
        userId="me",
        q=f"after:{after} before:{before}",
        maxResults=20
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return f"No emails found for {date}."

    output = [f"Emails received on {date}:\n"]
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        output.append(
            f"From: {headers.get('From', 'Unknown')} | "
            f"Subject: {headers.get('Subject', 'No Subject')}"
        )
    return "\n".join(output)


# ══════════════════════════════════════════
#  TOOL 4 — Draft Email
# ══════════════════════════════════════════
class DraftEmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Subject of the email")
    context: str = Field(description="What the email should say")
    tone: str = Field(
        default="professional",
        description="Tone: professional, friendly, formal, casual"
    )

    @field_validator("to")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"'{v}' is not a valid email address")
        return v.strip()

@tool(args_schema=DraftEmailInput)
def draft_email(
    to: str, subject: str, context: str, tone: str = "professional"
) -> str:
    """
    Uses AI to draft a complete email based on user instructions.
    Use this when the user asks to write or compose an email.
    Always show the draft to the user BEFORE sending.
    Never send without user approval.
    """
    client = get_llm()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert email writer. "
                    "Write a complete, well-structured email. "
                    "Start directly with the greeting. "
                    "No preamble or explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Write a {tone} email.\n"
                    f"To: {to}\n"
                    f"Subject: {subject}\n"
                    f"Instructions: {context}"
                )
            }
        ]
    )
    body = response.choices[0].message.content.strip()

    # Save draft for later sending
    draft = {"to": to, "subject": subject, "body": body}
    with open(".draft_cache.pkl", "wb") as f:
        pickle.dump(draft, f)

    return (
        f"Draft Email:\n"
        f"{'='*40}\n"
        f"To: {to}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n"
        f"{'='*40}\n"
        f"Reply 'send it' to send or 'revise: <changes>' to update."
    )


# ══════════════════════════════════════════
#  TOOL 5 — Send Reviewed Email
# ══════════════════════════════════════════
class SendReviewedInput(BaseModel):
    confirmed: bool = Field(
        description="Set True only when user explicitly says send it"
    )

@tool(args_schema=SendReviewedInput)
def send_reviewed_email(confirmed: bool) -> str:
    """
    Sends the previously drafted email after user approval.
    Only call this when the user explicitly says 'send it',
    'yes send', or 'looks good send it'. Never send without
    explicit user confirmation.
    """
    if not confirmed:
        return "Email not sent. Please confirm by saying 'send it'."

    if not os.path.exists(".draft_cache.pkl"):
        return "No draft found. Please draft an email first."

    with open(".draft_cache.pkl", "rb") as f:
        draft = pickle.load(f)

    service = get_gmail_service()
    message = MIMEText(draft["body"])
    message["to"] = draft["to"]
    message["subject"] = draft["subject"]
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    os.remove(".draft_cache.pkl")

    return f"Email sent to {draft['to']} | Subject: '{draft['subject']}'"


# ══════════════════════════════════════════
#  TOOL 6 — Daily Email Summary
#  Handles natural language like "yesterday",
#  "last 2 days", "today", or YYYY-MM-DD
# ══════════════════════════════════════════
class DailySummaryInput(BaseModel):
    date: Optional[str] = Field(
        default=None,
        description=(
            "Date or time period to summarise. Examples: "
            "'today', 'yesterday', 'last 2 days', '2026-02-23'. "
            "Defaults to today if not provided."
        )
    )

@tool(args_schema=DailySummaryInput)
def daily_email_summary(date: Optional[str] = None) -> str:
    """
    Generates an AI summary of all emails received on a given day
    or over the last N days. Use this when the user asks for a
    daily digest, summary of today's emails, yesterday's emails,
    last 2 days, or what they missed. Accepts natural language like
    'today', 'yesterday', 'last 2 days', or YYYY-MM-DD format.
    """
    # Handle natural language date inputs
    if date is None or date.lower().strip() in ["today", ""]:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

    elif date.lower().strip() == "yesterday":
        end_time = datetime.now() - timedelta(days=1)
        start_time = end_time - timedelta(days=1)

    elif any(word in date.lower() for word in ["last", "past", "recent"]):
        # Extract number from "last 2 days", "past 3 days" etc.
        numbers = re.findall(r'\d+', date)
        days_back = int(numbers[0]) if numbers else 1
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

    else:
        # Try standard YYYY-MM-DD format
        try:
            start_time = datetime.strptime(date.strip(), "%Y-%m-%d")
            end_time = start_time + timedelta(days=1)
        except ValueError:
            # Fallback to today if format not recognised
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

    after_ts = int(start_time.timestamp())
    before_ts = int(end_time.timestamp())

    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me",
        q=f"after:{after_ts} before:{before_ts}",
        maxResults=30
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return "No emails received in the specified time period."

    snippets = []
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        snippets.append(
            f"From: {headers.get('From','?')} | "
            f"Subject: {headers.get('Subject','?')} | "
            f"{data.get('snippet','')[:100]}"
        )

    client = get_llm()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Buraq, an intelligent email assistant. "
                    "Summarise these emails as a clear daily digest. "
                    "Use bullet points. Highlight anything urgent."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Summarise these {len(snippets)} emails:\n\n"
                    + "\n".join(snippets)
                )
            }
        ]
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════
#  TOOL 7 — Check Spam
# ══════════════════════════════════════════
class CheckSpamInput(BaseModel):
    max_results: int = Field(default=10, ge=1, le=50)

@tool(args_schema=CheckSpamInput)
def check_spam(max_results: int = 10) -> str:
    """
    Checks the Gmail spam folder and reports suspicious emails.
    Use this when user asks about spam or junk mail.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", labelIds=["SPAM"], maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return "No spam emails found."

    output = [f"{len(messages)} spam email(s) detected:\n"]
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        output.append(
            f"From: {headers.get('From','?')} | "
            f"Subject: {headers.get('Subject','?')}"
        )
    return "\n".join(output)


# ══════════════════════════════════════════
#  TOOL 8 — Check Replies
# ══════════════════════════════════════════
class CheckRepliesInput(BaseModel):
    hours_back: int = Field(
        default=24, ge=1, le=720,
        description="How many hours back to check for replies (max 720 = 30 days)"
    )

@tool(args_schema=CheckRepliesInput)
def check_replies(hours_back: int = 24) -> str:
    """
    Checks if anyone replied to the user's sent emails recently.
    Use this when user asks if anyone replied or responded to them.
    Accepts up to 720 hours (30 days) back.
    """
    service = get_gmail_service()
    since = datetime.now() - timedelta(hours=hours_back)
    after_ts = int(since.timestamp())

    results = service.users().messages().list(
        userId="me",
        q=f"in:inbox after:{after_ts}",
        maxResults=20
    ).execute()
    messages = results.get("messages", [])

    replies = []
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers_list = data["payload"]["headers"]
        headers = {h["name"]: h["value"] for h in headers_list}
        if any(h["name"] == "In-Reply-To" for h in headers_list):
            replies.append(
                f"From: {headers.get('From','?')} | "
                f"Subject: {headers.get('Subject','?')}"
            )

    if not replies:
        return f"No replies received in the last {hours_back} hours."

    return (
        f"{len(replies)} reply/replies in last {hours_back} hours:\n\n"
        + "\n".join(replies)
    )


# ══════════════════════════════════════════
#  TOOL 9 — Important Alerts
# ══════════════════════════════════════════
class ImportantAlertsInput(BaseModel):
    max_results: int = Field(default=10, ge=1, le=30)

@tool(args_schema=ImportantAlertsInput)
def check_important_alerts(max_results: int = 10) -> str:
    """
    Scans inbox and uses AI to find urgent emails and deadlines.
    Use this when user asks what needs attention, any deadlines,
    or anything important in their inbox.
    """
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return "No emails to analyse."

    snippets = []
    for msg in messages:
        data = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()
        headers = {
            h["name"]: h["value"]
            for h in data["payload"]["headers"]
        }
        snippets.append(
            f"From: {headers.get('From','?')} | "
            f"Subject: {headers.get('Subject','?')} | "
            f"{data.get('snippet','')[:120]}"
        )

    client = get_llm()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Buraq. Analyse these emails and extract "
                    "only urgent ones, deadlines, or action items. "
                    "If nothing urgent, say so clearly."
                )
            },
            {
                "role": "user",
                "content": "\n".join(snippets)
            }
        ]
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════
#  TOOL 10 — Search Knowledge Base (Lab 2)
# ══════════════════════════════════════════
class SearchKBInput(BaseModel):
    query: str = Field(
        description="Natural language query to search past emails"
    )

@tool(args_schema=SearchKBInput)
def search_knowledge_base(query: str) -> str:
    """
    Searches the ChromaDB vector store for past emails
    using semantic similarity. Use this when the user wants
    to find an old email by describing what it was about,
    even without remembering exact words or sender.
    This is the grounding tool that searches indexed email history.
    """
    return kb_search(query)


# ══════════════════════════════════════════
#  Tool Registry
# ══════════════════════════════════════════
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