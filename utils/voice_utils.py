import os
import tempfile
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Load Whisper Turbo (ffmpeg-free)
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    token=HF_API_KEY
)

# --------------------------
# Transcribe uploaded audio
# --------------------------
def transcribe_audio_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        tmp.write(file.read())

    result = asr(temp_path)
    return result["text"]

# --------------------------
# Transcribe live audio bytes
# --------------------------
def transcribe_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        tmp.write(audio_bytes)

    result = asr(temp_path)
    return result["text"]
