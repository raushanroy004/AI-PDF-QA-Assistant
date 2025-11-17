import streamlit as st
from groq import Groq

# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets["HF_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

def text_to_speech(text: str):
    """
    Returns MP3 audio bytes from Groq TTS.
    Works directly with st.audio().
    """

    try:
        response = client.audio.speech.create(
            model="g1-mini",      # ✅ WORKING GROQ TTS MODEL
            voice="alloy",        # any supported voice
            input=text
        )

        audio_bytes = response.read()  # raw MP3 bytes
        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
