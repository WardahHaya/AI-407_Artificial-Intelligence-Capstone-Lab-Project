from __future__ import annotations

from ingest_data import COLLECTION_NAME, query_chunks


def search_knowledge_base(query: str, n_results: int = 3, where: dict[str, str] | None = None) -> str:
    """
    Canonical semantic search over the shared Lab 2 grounded memory collection.
    This wrapper exists for backwards compatibility with older scripts under vector_store/.
    """
    matches = query_chunks(query=query, top_k=n_results, where=where)
    if not matches:
        return f"No grounded matches were found for: {query!r}"

    blocks: list[str] = [f"Collection: {COLLECTION_NAME}"]
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"Match {index}",
                    f"doc_type: {metadata.get('doc_type', 'unknown')}",
                    f"department: {metadata.get('department', 'unknown')}",
                    f"priority_level: {metadata.get('priority_level', 'unknown')}",
                    f"source_file: {metadata.get('source_file', 'unknown')}",
                    match["document"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    print(search_knowledge_base("updated resume interview", n_results=1, where={"department": "careers"}))
