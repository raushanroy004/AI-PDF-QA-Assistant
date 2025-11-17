import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------------------------------
# 1️⃣ Function: Transcribe uploaded voice file (wav/mp3)
# -------------------------------------------------------
def transcribe_audio_file(file):
    """
    Transcribes uploaded audio file (wav/mp3)
    using Whisper (Groq).
    """
    # Save temp file
    temp_path = "uploaded_audio_temp.wav"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    # Whisper STT
    with open(temp_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )

    return response.text



# -------------------------------------------------------
# 2️⃣ Function: Transcribe LIVE microphone audio (raw bytes)
# -------------------------------------------------------
def transcribe_audio_bytes(audio_bytes):
    """
    Takes raw audio bytes recorded from streamlit-webrtc
    and transcribes it using Whisper (Groq).
    """

    temp_path = "live_audio_temp.wav"

    # Save it
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # Whisper STT
    with open(temp_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )

    return response.text
