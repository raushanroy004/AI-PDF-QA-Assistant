import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model ONCE (Streamlit Cloud supports this)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------------------------
# 1️⃣ Get embedding
# -------------------------------------------------------
def get_embedding(text: str):
    """
    Generate embeddings using local SentenceTransformer model.
    Works on Streamlit Cloud reliably.
    """
    emb = embed_model.encode(text, convert_to_numpy=True)
    return emb.astype("float32")


# -------------------------------------------------------
# 2️⃣ Create FAISS index
# -------------------------------------------------------
def create_faiss_index(chunks):
    """
    Creates a FAISS index for PDF chunks.
    """
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))
    return index
