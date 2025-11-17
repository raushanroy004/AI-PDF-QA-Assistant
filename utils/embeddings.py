import os
import numpy as np
import faiss
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# Load HF key
HF_API_KEY = os.getenv("HF_API_KEY")

# Create HF embedding client
client = InferenceClient(api_key=HF_API_KEY)

def get_embedding(text: str):
    """
    Generate embedding using HuggingFace Inference API.
    """
    response = client.feature_extraction(
        model="sentence-transformers/all-MiniLM-L6-v2",
        inputs=text
    )
    return np.array(response, dtype="float32")


def create_faiss_index(chunks):
    """
    Create FAISS vector index from text chunks.
    """
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index
