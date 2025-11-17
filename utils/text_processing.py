import re

def clean_text(text):
    """
    Clean extracted PDF text.
    Remove extra spaces, newlines, and weird characters.
    """
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=800):
    """
    Split text into smaller chunks for embeddings.
    Each chunk ~800 characters.
    """
    chunks = []
    words = text.split()

    current_chunk = []

    for word in words:
        current_chunk.append(word)

        # If chunk limit reached → create new chunk
        if sum(len(w) for w in current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks