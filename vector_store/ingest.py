# vector_store/ingest.py
# Pulls emails from Gmail and stores them in ChromaDB
# ChromaDB is a local vector database - it saves to a folder called chroma_db/
# Each email is embedded (converted to numbers) and stored
# so we can later search them by meaning

import os
import pickle
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import chromadb
from vector_store.embeddings import get_embedding_model

load_dotenv()

# Gmail permission scope - we only need to read emails
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Where ChromaDB saves data on your laptop
CHROMA_DB_PATH = "chroma_db"

# Name of the collection inside ChromaDB (like a table in a database)
COLLECTION_NAME = "buraq_emails"


def get_gmail_service():
    """
    Authenticates with Gmail API using OAuth2.
    First run: opens browser for Google login.
    After that: uses saved token.pickle automatically.
    """
    creds = None
    token_path = "token.pickle"
    creds_path = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json")

    # Load saved token if it exists
    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    # If no valid credentials, ask user to login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for next time
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def fetch_emails(max_results: int = 50) -> list[dict]:
    """
    Fetches recent emails from Gmail inbox.
    Returns a list of dicts with id, subject, sender, date, snippet.
    
    Args:
        max_results: How many emails to fetch (default 50)
    """
    print(f"Fetching {max_results} emails from Gmail...")
    service = get_gmail_service()

    # Get list of message IDs
    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max_results
    ).execute()

    messages = results.get("messages", [])

    if not messages:
        print("No emails found.")
        return []

    emails = []
    for i, msg in enumerate(messages):
        print(f"  Fetching email {i+1}/{len(messages)}...", end="\r")
        
        # Get full email data
        data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "Date"]
        ).execute()

        headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}

        emails.append({
            "id": msg["id"],
            "subject": headers.get("Subject", "No Subject"),
            "sender": headers.get("From", "Unknown"),
            "date": headers.get("Date", ""),
            "snippet": data.get("snippet", ""),
        })

    print(f"\nFetched {len(emails)} emails successfully.")
    return emails


def ingest_emails_to_chromadb(max_results: int = 50):
    """
    Main ingestion pipeline:
    1. Fetch emails from Gmail
    2. Create text representation of each email
    3. Embed each email using sentence-transformers
    4. Store in ChromaDB for semantic search
    
    Args:
        max_results: Number of emails to ingest
    """
    # Step 1: Fetch emails
    emails = fetch_emails(max_results)
    if not emails:
        print("No emails to ingest.")
        return

    # Step 2: Load embedding model
    model = get_embedding_model()

    # Step 3: Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    # If it already exists we use it, otherwise create fresh
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Buraq email knowledge base"}
    )

    # Step 4: Prepare data for ChromaDB
    documents = []   # The text we embed and search
    metadatas = []   # Extra info stored alongside each document
    ids = []         # Unique ID for each document

    # Get IDs already in the database to avoid duplicates
    existing = collection.get()["ids"]

    new_count = 0
    for email in emails:
        # Skip if already ingested
        if email["id"] in existing:
            continue

        # Create a searchable text representation of the email
        # This is what gets embedded and searched
        doc_text = (
            f"Subject: {email['subject']}\n"
            f"From: {email['sender']}\n"
            f"Date: {email['date']}\n"
            f"Content: {email['snippet']}"
        )

        documents.append(doc_text)
        metadatas.append({
            "email_id": email["id"],
            "subject": email["subject"],
            "sender": email["sender"],
            "date": email["date"],
        })
        ids.append(email["id"])
        new_count += 1

    if not documents:
        print("All emails already ingested. Database is up to date.")
        return

    # Step 5: Generate embeddings for all documents
    print(f"Embedding {new_count} new emails...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # Step 6: Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\nSuccessfully ingested {new_count} emails into ChromaDB.")
    print(f"Total emails in knowledge base: {collection.count()}")


if __name__ == "__main__":
    ingest_emails_to_chromadb(max_results=50)