# vector_store/embeddings.py
# Loads the sentence-transformer embedding model
# This model converts text (emails) into numbers (vectors)
# so ChromaDB can find similar emails by meaning

from sentence_transformers import SentenceTransformer

# We use this specific model because it is:
# - Free and runs locally (no API key needed)
# - Fast and accurate for short texts like emails
# - Small enough to run on a student laptop
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Loads and returns the sentence transformer model.
    First run will download the model (~90MB).
    After that it loads from local cache instantly.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")
    return model