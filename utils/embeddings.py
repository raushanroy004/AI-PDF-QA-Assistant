import os
import faiss
import numpy as np
from groq import Groq

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Correct Groq embedding model
EMBED_MODEL = "all-minilm-l12-v2"


def get_embedding(text: str):
    """
    Generate embeddings using Groq's supported embedding model.
    """

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )

    embedding = response.data[0].embedding
    return np.array(embedding, dtype="float32")


def create_faiss_index(chunks):
    """
    Create FAISS vector index from PDF chunks.
    """

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index
