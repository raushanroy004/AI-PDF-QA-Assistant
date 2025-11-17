import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import faiss

load_dotenv()

# Load Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Import embedding function
from utils.embeddings import get_embedding


def answer_question_with_rag(question, faiss_index, chunks):
    """
    Returns:
        short_summary (str)
        long_summary (str)
    """

    # Convert question into vector
    question_emb = np.array([get_embedding(question)], dtype="float32")

    # Search FAISS index
    D, I = faiss_index.search(question_emb, 3)

    # Retrieve relevant chunks
    retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n\n".join(retrieved_chunks)

    # Groq LLM: Clean short + long format
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize PDF content clearly. "
                    "Avoid tags like <SHORT>. Keep writing clean."
                )
            },
            {
                "role": "user",
                "content": f"""
Use ONLY the following PDF content to answer.

CONTEXT:
{context}

QUESTION:
{question}

Return your response ONLY in this clean format:

SHORT_SUMMARY:
(Write a short 2–3 line crisp summary.)

LONG_SUMMARY:
(Write a detailed, well-structured explanation.)
"""
            }
        ]
    )

    full_text = response.choices[0].message.content.strip()

    # Extract short + long summaries
    try:
        short_sum = full_text.split("SHORT_SUMMARY:")[1].split("LONG_SUMMARY:")[0].strip()
        long_sum = full_text.split("LONG_SUMMARY:")[1].strip()
    except:
        # fallback — in case model didn’t follow format
        short_sum = full_text[:200] + "..."
        long_sum = full_text

    return short_sum, long_sum
