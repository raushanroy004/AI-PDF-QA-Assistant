from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype('float32')

def create_faiss_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = embeddings[0].shape[0]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index

def load_faiss_index(chunks, index):
    # Recompute embeddings for searching
    return [get_embedding(chunk) for chunk in chunks]
