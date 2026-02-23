# vector_store/search.py
# Semantic search over ChromaDB + AI-powered conversational response

import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from vector_store.embeddings import get_embedding_model

load_dotenv()

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "buraq_emails"


def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Searches ChromaDB for relevant emails and returns
    a natural language AI response based on what was found.
    """
    # Step 1 — Load ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return "Knowledge base is empty. Please run ingest.py first."

    if collection.count() == 0:
        return "Knowledge base is empty. Please run ingest.py first."

    # Step 2 — Embed the query and search
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count())
    )

    if not results["documents"][0]:
        return f"No emails found matching: '{query}'"

    # Step 3 — Build context from retrieved emails
    context = ""
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        context += (
            f"Email {i+1}:\n"
            f"  Subject: {meta.get('subject', 'N/A')}\n"
            f"  From: {meta.get('sender', 'N/A')}\n"
            f"  Date: {meta.get('date', 'N/A')}\n"
            f"  Preview: {doc.split('Content:')[-1].strip()[:200]}\n\n"
        )

    # Step 4 — Ask Groq LLM to answer the user's question
    # based on the emails we retrieved
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Buraq, an intelligent Gmail assistant. "
                    "The user has asked a question about their emails. "
                    "You have retrieved the most relevant emails from their inbox. "
                    "Answer the user's question naturally and helpfully based on "
                    "these emails. Be conversational, concise, and specific. "
                    "If the exact email isn't found, say so honestly but mention "
                    "what related emails were found."
                )
            },
            {
                "role": "user",
                "content": (
                    f"My question: {query}\n\n"
                    f"Relevant emails found:\n{context}"
                )
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("\n Buraq Email Search")
    print("=" * 40)
    print("Ask anything about your emails.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query:
            continue

        print("\nBuraq: thinking...\n")
        response = search_knowledge_base(query)
        print(f"Buraq: {response}\n")
        print("-" * 40)