import streamlit as st
from groq import Groq

# Load secrets (Streamlit Cloud)
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize Groq client
client = Groq(api_key=HF_API_KEY)

def text_to_speech(text: str):
    """
    Convert text → speech (mp3 bytes) using Groq TTS.
    Returns raw audio bytes (best for Streamlit).
    """

    try:
        response = client.audio.speech.create(
            model="g2p-ko-v2",   # Groq’s English-capable TTS model
            voice="alloy",
            input=text
        )

        # Raw audio bytes
        audio_bytes = response.read()

        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
