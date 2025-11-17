import os
import numpy as np
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize HF embedding client
client = InferenceClient(api_key=HF_API_KEY)


def get_embedding(text):
    """
    Generate embeddings from HF Inference API
    using sentence-transformers/all-MiniLM-L6-v2
    """

    response = client.post(
        "/pipeline/feature-extraction",
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "inputs": text
        },
    )

    # Convert to numpy float32
    emb = np.array(response, dtype="float32")

    # Some models return list[list[]], flatten if needed
    if emb.ndim == 2:
        emb = emb[0]

    return emb


def create_faiss_index(chunks):
    """
    Create FAISS index for chunk embeddings.
    """

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index
