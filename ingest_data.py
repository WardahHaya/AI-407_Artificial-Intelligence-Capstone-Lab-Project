from __future__ import annotations

import argparse
import csv
import html
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import chromadb

from vector_store.embeddings import get_embedding_model

DATA_DIR = Path("Initial_Data")
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "buraq_source_memory_lab2"
EMBED_BATCH_SIZE = 32


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def _disable_chroma_telemetry() -> None:
    """
    Chroma can emit noisy PostHog telemetry errors in some local environments.
    Retrieval and indexing still work, so we disable the telemetry noise explicitly.
    """
    try:
        import posthog

        posthog.disabled = True
        posthog.capture = lambda *args, **kwargs: None
    except Exception:
        pass


def clean_text(text: str) -> str:
    """Normalizes raw text by removing boilerplate noise and spacing issues."""
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$", " ", text)
    text = re.sub(r"(?im)^\s*(confidential|internal use only)\s*$", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"[-_=]{3,}", line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower() or "chunk"


def format_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def infer_email_department(category: str) -> str:
    mapping = {
        "course": "academics",
        "recruitment": "careers",
        "team": "project_team",
        "admin": "student_services",
        "system": "engineering",
        "newsletter": "external_updates",
    }
    return mapping.get(category.lower(), "general")


def infer_deadline_priority(due_date: str, current_date: str) -> str:
    if due_date[:10] <= current_date:
        return "high"
    if due_date[:10] <= "2026-05-07":
        return "high"
    if due_date[:10] <= "2026-05-09":
        return "medium"
    return "low"


def chunk_inbox_emails(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            body = clean_text(
                "\n".join(
                    [
                        f"Subject: {row['subject']}",
                        f"From: {row['from']}",
                        f"To: {row['to']}",
                        f"Date: {row['date']}",
                        f"Snippet: {row['snippet']}",
                        f"Category: {row['category']}",
                        f"Priority: {row['priority']}",
                        f"Action required: {row['action_required']}",
                        f"Has attachment: {row['has_attachment']}",
                    ]
                )
            )
            metadata = {
                "doc_type": "incoming_email",
                "department": infer_email_department(row["category"]),
                "priority_level": row["priority"].lower(),
                "last_updated": row["date"][:10],
                "source_file": path.name,
                "thread_id": row["thread_id"],
                "category": row["category"].lower(),
                "action_required": row["action_required"].lower(),
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-{normalize_id(row['email_id'])}",
                    text=body,
                    metadata=metadata,
                )
            )
    return chunks


def chunk_sent_style_references(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            body = clean_text(
                "\n".join(
                    [
                        f"Subject: {row['subject']}",
                        f"Recipient: {row['to']}",
                        f"Date: {row['date']}",
                        f"Tone: {row['tone']}",
                        f"Body excerpt: {row['body_excerpt']}",
                        f"Sign off: {row['sign_off']}",
                    ]
                )
            )
            metadata = {
                "doc_type": "sent_style_reference",
                "department": "writing_style",
                "priority_level": "medium",
                "last_updated": row["date"][:10],
                "source_file": path.name,
                "tone": row["tone"].lower(),
                "recipient_scope": "external" if "@" in row["to"] else "internal",
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-{normalize_id(row['email_id'])}",
                    text=body,
                    metadata=metadata,
                )
            )
    return chunks


def chunk_deadline_rows(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            priority = infer_deadline_priority(row["due_date"], current_date="2026-05-05")
            body = clean_text(
                "\n".join(
                    [
                        f"Title: {row['title']}",
                        f"Due date: {row['due_date']}",
                        f"Owner: {row['owner']}",
                        f"Source: {row['source']}",
                        f"Details: {row['details']}",
                        f"Status: {row['status']}",
                        f"Linked email id: {row['linked_email_id']}",
                    ]
                )
            )
            metadata = {
                "doc_type": "deadline_record",
                "department": "task_management",
                "priority_level": priority,
                "last_updated": row["due_date"][:10],
                "source_file": path.name,
                "status": row["status"].lower(),
                "source_kind": row["source"].lower(),
            }
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-{normalize_id(row['item_id'])}",
                    text=body,
                    metadata=metadata,
                )
            )
    return chunks


def split_text_sections(raw_text: str) -> list[tuple[str, str]]:
    """
    Splits narrative documents into section-aware chunks instead of fixed windows.
    This preserves complete units such as agenda blocks, action-item lists, and risks.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title = "overview"
    current_lines: list[str] = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("source:"):
            continue
        if line.endswith(":") and len(line) <= 80:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line[:-1].strip().lower()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    normalized_sections: list[tuple[str, str]] = []
    for title, lines in sections:
        merged = clean_text("\n".join(lines))
        if merged:
            normalized_sections.append((title, merged))
    return normalized_sections


def chunk_text_document(path: Path, doc_type: str, department: str, priority_level: str) -> list[ChunkRecord]:
    raw_text = clean_text(path.read_text(encoding="utf-8"))
    sections = split_text_sections(raw_text)
    last_updated = format_timestamp(path)

    chunks: list[ChunkRecord] = []
    for index, (section_title, section_body) in enumerate(sections, start=1):
        body = clean_text(
            "\n".join(
                [
                    f"Document: {path.stem.replace('_', ' ')}",
                    f"Section: {section_title.replace('_', ' ')}",
                    section_body,
                ]
            )
        )
        metadata = {
            "doc_type": doc_type,
            "department": department,
            "priority_level": priority_level,
            "last_updated": last_updated,
            "source_file": path.name,
            "section": section_title.replace(" ", "_"),
        }
        chunks.append(
            ChunkRecord(
                chunk_id=f"{path.stem}-section-{index}",
                text=body,
                metadata=metadata,
            )
        )
    return chunks


def load_project_chunks(data_dir: Path = DATA_DIR) -> list[ChunkRecord]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_chunks: list[ChunkRecord] = []
    all_chunks.extend(chunk_inbox_emails(data_dir / "inbox_emails_sample.csv"))
    all_chunks.extend(chunk_sent_style_references(data_dir / "sent_emails_style_reference.csv"))
    all_chunks.extend(chunk_deadline_rows(data_dir / "project_deadlines.csv"))
    all_chunks.extend(
        chunk_text_document(
            data_dir / "capstone_lab_brief_extracted.txt",
            doc_type="course_brief",
            department="academics",
            priority_level="high",
        )
    )
    all_chunks.extend(
        chunk_text_document(
            data_dir / "team_meeting_minutes.txt",
            doc_type="meeting_notes",
            department="project_team",
            priority_level="medium",
        )
    )
    return all_chunks


@lru_cache(maxsize=1)
def get_chroma_client():
    _disable_chroma_telemetry()
    chroma_host = os.getenv("CHROMA_HOST")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_ssl = os.getenv("CHROMA_SSL", "false").lower() == "true"
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=chroma_ssl)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Lab 2 grounded source memory for the Buraq Gmail agent"},
    )


def batch_iterable(items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _encode_texts(model, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, show_progress_bar=False)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vector) for vector in embeddings]


def ingest_chunks(chunks: list[ChunkRecord]) -> None:
    if not chunks:
        raise ValueError("No chunks were created from Initial_Data.")

    collection = get_collection()
    model = get_embedding_model()

    print(f"Preparing to ingest {len(chunks)} semantic chunks into '{COLLECTION_NAME}'...")
    for batch in batch_iterable(chunks, EMBED_BATCH_SIZE):
        texts = [chunk.text for chunk in batch]
        embeddings = _encode_texts(model, texts)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )

    print(f"Ingestion complete. Collection now stores {collection.count()} chunks.")


def query_chunks(query: str, top_k: int = 3, where: dict[str, str] | None = None) -> list[dict[str, object]]:
    collection = get_collection()
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run ingest_data.py first.")

    model = get_embedding_model()
    query_embedding = _encode_texts(model, [clean_text(query)])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        where=where,
    )

    matches: list[dict[str, object]] = []
    for index, doc_id in enumerate(results["ids"][0]):
        matches.append(
            {
                "id": doc_id,
                "document": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index],
            }
        )
    return matches


def print_matches(matches: list[dict[str, object]]) -> None:
    for match in matches:
        metadata = match["metadata"]
        print("-" * 80)
        print(f"ID: {match['id']}")
        print(f"Distance: {match['distance']:.4f}")
        print(
            "Metadata: "
            f"doc_type={metadata['doc_type']}, "
            f"department={metadata['department']}, "
            f"priority_level={metadata['priority_level']}, "
            f"source_file={metadata['source_file']}"
        )
        print(match["document"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and query Buraq Lab 2 grounding data.")
    parser.add_argument("--query", help="Optional semantic query to run after ingestion.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieval results to return.")
    parser.add_argument(
        "--filter",
        action="append",
        nargs=2,
        metavar=("FIELD", "VALUE"),
        help="Optional metadata filter pairs, e.g. --filter doc_type course_brief",
    )
    args = parser.parse_args()

    chunks = load_project_chunks()
    ingest_chunks(chunks)

    if args.query:
        where = {field: value for field, value in args.filter} if args.filter else None
        matches = query_chunks(args.query, top_k=args.top_k, where=where)
        print_matches(matches)


if __name__ == "__main__":
    main()
