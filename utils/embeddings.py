import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# --------------------------------------------
#  Authenticate using Streamlit Secrets or .env
# --------------------------------------------
HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY:
    login(HF_API_KEY)
else:
    print("⚠️ HF_API_KEY missing! Please add it in Streamlit Secrets.")

# Sentence Transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Create single text embedding
def get_embedding(text):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype('float32')

# Create FAISS index from chunks
def create_faiss_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Recompute embeddings for search
def load_faiss_index(chunks, index):
    return [get_embedding(chunk) for chunk in chunks]
