import os
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Import embedding function
from utils.embeddings import get_embedding

# Load Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing! Add it to .env or Streamlit secrets.")

client = Groq(api_key=GROQ_API_KEY)


def answer_question_with_rag(question, faiss_index, chunks):
    """
    Returns extracted short and long summaries using:
    - FAISS vector search
    - Groq LLM
    """

    # Convert question to embedding
    question_emb = np.array([get_embedding(question)], dtype="float32")

    # Perform FAISS similarity search
    D, I = faiss_index.search(question_emb, 3)

    # Retrieve relevant chunks safely
    retrieved_chunks = [
        chunks[i] for i in I[0]
        if isinstance(i, (int, np.integer)) and i < len(chunks)
    ]

    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant text found."

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert PDF summarizer. "
                    "Always respond clearly and professionally."
                )
            },
            {
                "role": "user",
                "content": f"""
Use ONLY the following PDF content to answer:

=== PDF CONTEXT ===
{context}

=== QUESTION ===
{question}

Respond in EXACTLY this clean format:

SHORT_SUMMARY:
(Provide a crisp 2–3 sentence summary)

LONG_SUMMARY:
(Provide a full, structured detailed explanation)
"""
            }
        ]
    )

    full_text = response.choices[0].message.content.strip()

    # --------------------------------------------
    # SAFE PARSING
    # --------------------------------------------
    short_sum = "Summary unavailable."
    long_sum = full_text  # fallback

    if "SHORT_SUMMARY:" in full_text:
        try:
            short_sum = (
                full_text.split("SHORT_SUMMARY:")[1]
                .split("LONG_SUMMARY:")[0]
                .strip()
            )
        except:
            pass

    if "LONG_SUMMARY:" in full_text:
        try:
            long_sum = full_text.split("LONG_SUMMARY:")[1].strip()
        except:
            pass

    return short_sum, long_sum
