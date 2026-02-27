import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load HF embedding model (local computation)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str):
    """
    Generate text embeddings using sentence-transformers.
    Completely local — no API required.
    """
    emb = embedder.encode(text, convert_to_numpy=True)
    return emb.astype("float32")


def create_faiss_index(chunks):
    """
    Create FAISS vector index from text chunks.
    """

    if not chunks:
        raise ValueError("No text chunks found. PDF might be empty or extraction failed.")

    embeddings = [get_embedding(chunk) for chunk in chunks]

    if len(embeddings) == 0:
        raise ValueError("Embeddings list is empty.")

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]   # safer than embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index
