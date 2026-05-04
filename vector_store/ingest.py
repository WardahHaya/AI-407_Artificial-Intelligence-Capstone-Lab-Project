# vector_store/ingest.py
# Pulls ALL emails from Gmail (inbox + sent) and stores them in ChromaDB
# Uses pagination to fetch every single email
# Sent emails are used as style references when drafting new emails

import os
import pickle
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import chromadb
from ingest_data import get_chroma_client
from vector_store.embeddings import get_embedding_model

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "buraq_emails"
SENT_COLLECTION_NAME = "buraq_sent_emails"


def get_gmail_service():
    """
    Authenticates with Gmail API using OAuth2.
    First run: opens browser for Google login.
    After that: uses saved token.pickle automatically.
    """
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


def _fetch_messages_by_label(service, label: str, max_results: int = None) -> list[dict]:
    """
    Generic paginated fetcher for any Gmail label (INBOX, SENT, etc.)
    Returns list of raw message dicts with full metadata.
    """
    label_display = label.capitalize()
    print(f"\nFetching ALL emails from Gmail {label_display}...")

    all_messages = []
    page_token = None
    page_num = 1

    while True:
        print(f"  Scanning page {page_num} ({len(all_messages)} found so far)...", end="\r")

        request_params = {
            "userId": "me",
            "labelIds": [label],
            "maxResults": 500,
        }
        if page_token:
            request_params["pageToken"] = page_token

        results = service.users().messages().list(**request_params).execute()
        messages = results.get("messages", [])
        all_messages.extend(messages)

        if max_results and len(all_messages) >= max_results:
            all_messages = all_messages[:max_results]
            print(f"\n  Reached limit of {max_results} emails.")
            break

        page_token = results.get("nextPageToken")
        if not page_token:
            print(f"\n  All pages scanned. Total found: {len(all_messages)}")
            break

        page_num += 1

    if not all_messages:
        print(f"No emails found in {label_display}.")
        return []

    # Fetch full metadata for each message
    print(f"\nFetching details for {len(all_messages)} {label_display} emails...")
    emails = []
    for i, msg in enumerate(all_messages):
        print(f"  Getting email {i+1}/{len(all_messages)}...", end="\r")

        data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"]
        ).execute()

        headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}

        emails.append({
            "id": msg["id"],
            "subject": headers.get("Subject", "No Subject"),
            "sender": headers.get("From", "Unknown"),
            "to": headers.get("To", "Unknown"),
            "date": headers.get("Date", ""),
            "snippet": data.get("snippet", ""),
        })

    print(f"\nFetched {len(emails)} {label_display} emails successfully.")
    return emails


def _embed_and_store(emails: list[dict], collection, label: str = "inbox"):
    """
    Embeds emails and stores them in the given ChromaDB collection.
    Skips duplicates automatically.
    """
    existing = set(collection.get()["ids"])
    print(f"\nAlready in database: {len(existing)}")

    documents, metadatas, ids = [], [], []

    for email in emails:
        if email["id"] in existing:
            continue

        doc_text = (
            f"Subject: {email['subject']}\n"
            f"From: {email.get('sender', '?')}\n"
            f"To: {email.get('to', '?')}\n"
            f"Date: {email['date']}\n"
            f"Content: {email['snippet']}"
        )

        documents.append(doc_text)
        metadatas.append({
            "email_id": email["id"],
            "subject": email["subject"],
            "sender": email.get("sender", "?"),
            "to": email.get("to", "?"),
            "date": email["date"],
        })
        ids.append(email["id"])

    if not documents:
        print("All emails already ingested. Database is up to date.")
        print(f"Total in knowledge base: {collection.count()}")
        return

    model = get_embedding_model()
    batch_size = 100
    all_embeddings = []

    print(f"\nEmbedding {len(documents)} new emails in batches...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(batch_embeddings)

    print(f"\nSaving to ChromaDB...")
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            embeddings=all_embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"  Saved batch {i//batch_size + 1}...")

    print(f"\nSuccessfully ingested {len(documents)} new {label} emails.")
    print(f"Total in knowledge base: {collection.count()}")


def fetch_emails(max_results: int = None) -> list[dict]:
    """Fetches ALL inbox emails using pagination."""
    service = get_gmail_service()
    return _fetch_messages_by_label(service, "INBOX", max_results)


def ingest_emails_to_chromadb(max_results: int = None):
    """
    Ingests ALL inbox emails into ChromaDB.
    Skips duplicates automatically.
    """
    emails = fetch_emails(max_results)
    if not emails:
        print("No inbox emails to ingest.")
        return

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Buraq inbox email knowledge base"}
    )
    _embed_and_store(emails, collection, label="inbox")


def ingest_sent_emails(max_results: int = None):
    """
    Ingests ALL sent emails into a separate ChromaDB collection.
    Buraq uses these as style references when drafting new emails —
    it learns your writing style from emails you've actually sent.
    """
    service = get_gmail_service()
    emails = _fetch_messages_by_label(service, "SENT", max_results)
    if not emails:
        print("No sent emails to ingest.")
        return

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=SENT_COLLECTION_NAME,
        metadata={"description": "Buraq sent email style reference"}
    )
    _embed_and_store(emails, collection, label="sent")


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1 — Ingesting inbox emails...")
    print("=" * 50)
    ingest_emails_to_chromadb(max_results=None)

    print("\n" + "=" * 50)
    print("STEP 2 — Ingesting sent emails for style learning...")
    print("=" * 50)
    ingest_sent_emails(max_results=None)

    print("\n✓ All done! Buraq knowledge base is fully updated.")
