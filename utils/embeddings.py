import os
import numpy as np
import faiss
from huggingface_hub import InferenceClient

# Load HF API Key
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize HF Client
hf_client = InferenceClient(api_key=HF_API_KEY)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------------------------------------
# 1️⃣ Get Embedding
# -------------------------------------------------------
def get_embedding(text: str):
    """
    Generate embeddings using HuggingFace Inference API.
    Works on Streamlit Cloud without errors.
    """
    response = hf_client.feature_extraction(
        model=EMBED_MODEL,
        inputs=text
    )

    # Convert to numpy
    vector = np.array(response, dtype="float32")

    # If model returns 2D array -> flatten it
    if len(vector.shape) > 1:
        vector = vector[0]

    return vector


# -------------------------------------------------------
# 2️⃣ Create FAISS index
# -------------------------------------------------------
def create_faiss_index(chunks):
    """
    Creates a FAISS index for PDF text chunks.
    """

    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index
