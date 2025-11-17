import io
from huggingface_hub import InferenceClient
import os

# Load HF API Key safely
HF_API_KEY = os.getenv("HF_API_KEY")

# Init client
client = InferenceClient(api_key=HF_API_KEY)


def text_to_speech(text: str):
    """
    Convert text → speech (MP3 bytes).
    Works on Streamlit Cloud with no ffmpeg.
    """

    if not text or not text.strip():
        return None

    try:
        # HuggingFace TTS model
        model = "espnet/kan-bayashi_ljspeech_vits"

        # Generate speech
        response = client.text_to_speech(
            model=model,
            text=text
        )

        # Convert response stream → mp3 bytes
        audio_bytes = io.BytesIO(response.read()).getvalue()
        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
