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

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = embeddings[0].shape[0]   # = 384
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    return index
