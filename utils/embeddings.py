import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load HF embedding model (local computation)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str):
    emb = embedder.encode(text, convert_to_numpy=True)
    return emb.astype("float32")


def create_faiss_index(chunks):
    """
    Create FAISS vector index from text chunks.
    """

    # 🚨 SAFETY CHECK 1
    if not chunks:
        raise ValueError("No text chunks found. PDF extraction failed.")

    embeddings = []

    for chunk in chunks:
        if chunk.strip():
            embeddings.append(get_embedding(chunk))

    # 🚨 SAFETY CHECK 2
    if len(embeddings) == 0:
        raise ValueError("Embeddings list is empty.")

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]  # safer method

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index
