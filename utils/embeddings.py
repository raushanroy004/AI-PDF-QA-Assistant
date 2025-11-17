import os
import faiss
import numpy as np
from groq import Groq

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize client
client = Groq(api_key=GROQ_API_KEY)


# -------------------------------------------------------
# 1️⃣ Create embedding vector
# -------------------------------------------------------
def get_embedding(text: str):
    """
    Returns a float32 embedding vector using Groq embedding model.
    """

    response = client.embeddings.create(
        model="nomic-embed-text",   # Groq's embedding model
        input=text
    )

    embedding = response.data[0].embedding
    return np.array(embedding, dtype="float32")


# -------------------------------------------------------
# 2️⃣ Create FAISS index from chunks
# -------------------------------------------------------
def create_faiss_index(chunks):
    """
    Builds FAISS index from PDF text chunks.
    """

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dim = len(embeddings[0])  # embedding vector size
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    return index
