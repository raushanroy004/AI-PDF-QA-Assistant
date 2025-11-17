import os
import faiss
import numpy as np
from groq import Groq

# Load Groq key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = "nomic-embed-text"

def get_embedding(text: str):
    """
    Generate embedding using Groq's embedding API.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    emb = response.data[0].embedding
    return np.array(emb, dtype="float32")


def create_faiss_index(chunks):
    """
    Create FAISS vector index.
    """

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index
