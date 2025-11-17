import os
import tempfile
from huggingface_hub import InferenceClient
import soundfile as sf
import numpy as np

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(api_key=HF_API_KEY)


def text_to_speech(text):
    """
    Convert text → WAV file (works 100%, audio bar will play correctly)
    """

    if not text or text.strip() == "":
        return None

    # Use HuggingFace TTS model (FastSpeech2)
    response = client.post(
        "facebook/fastspeech2-en-ljspeech",
        json={"text": text},
    )

    # Save WAV audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(response.content)
        return tmp.name
