from __future__ import annotations

from ingest_data import COLLECTION_NAME, ingest_chunks, load_project_chunks
from tools import ingest_emails_to_vector_store


def ingest_project_sources() -> str:
    chunks = load_project_chunks()
    ingest_chunks(chunks)
    return f"Ingested {len(chunks)} canonical project chunks into {COLLECTION_NAME}."


def ingest_live_gmail(max_results: int = 25, include_sent_style: bool = True) -> str:
    """
    Backwards-compatible wrapper that now targets the same canonical collection used by the runtime.
    """
    return str(
        ingest_emails_to_vector_store.invoke(
            {
                "max_results": max_results,
                "include_sent_style": include_sent_style,
            }
        )
    )


if __name__ == "__main__":
    print(ingest_project_sources())
