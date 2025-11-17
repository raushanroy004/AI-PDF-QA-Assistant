import os
import tempfile
from transformers import pipeline
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")


# -----------------------------
# Load Whisper model (HuggingFace)
# -----------------------------
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    token=HF_API_KEY
)


# -------------------------------------------------------
# 1️⃣ Transcribe uploaded audio (wav/mp3)
# -------------------------------------------------------
def transcribe_audio_file(file):
    """
    Transcribes uploaded WAV/MP3 file using Whisper (HF pipeline).
    """
    # Save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        tmp.write(file.read())

    # Transcribe
    result = asr(temp_path)
    return result["text"]


# -------------------------------------------------------
# 2️⃣ Transcribe LIVE microphone audio (raw bytes)
# -------------------------------------------------------
def transcribe_audio_bytes(audio_bytes):
    """
    Transcribes raw audio bytes recorded in Streamlit.
    Converts bytes → wav → transcribe.
    """

    # Convert bytes → numpy → WAV file
    try:
        # Try to read audio directly
        audio_np, sr = sf.read(
            tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name,
            dtype="float32"
        )
    except:
        # If raw bytes, convert to WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            tmp.write(audio_bytes)

        # Read the saved audio
        audio_np, sr = sf.read(temp_path, dtype="float32")

    # Save cleaned WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as cleaned:
        cleaned_path = cleaned.name
        sf.write(cleaned_path, audio_np, sr)

    # Transcribe
    result = asr(cleaned_path)
    return result["text"]
