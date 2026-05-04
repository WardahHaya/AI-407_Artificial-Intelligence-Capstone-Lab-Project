from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
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
