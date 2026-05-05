import hashlib
import os
import re
from functools import lru_cache

import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LIGHTWEIGHT_EMBEDDING_DIM = 256


class LightweightHashEmbeddingModel:
    def encode(self, texts, show_progress_bar: bool = False):
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(LIGHTWEIGHT_EMBEDDING_DIM, dtype=float)
            tokens = re.findall(r"[a-z0-9_]+", str(text).lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:2], byteorder="big") % LIGHTWEIGHT_EMBEDDING_DIM
                vector[index] += 1.0

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads and returns the sentence transformer model.
    First run will download the model (~90MB).
    After that it loads from local cache instantly.
    """
    if os.getenv("BURAQ_LIGHTWEIGHT_EMBEDDINGS", "false").lower() == "true":
        print("Loading lightweight hash embedding model for container runtime.")
        return LightweightHashEmbeddingModel()

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")
    return model
