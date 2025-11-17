import streamlit as st
from groq import Groq

# Load API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def text_to_speech(text: str):
    """
    Convert text → speech (mp3 bytes) using Groq TTS.
    Returns raw audio bytes.
    """
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",   # Groq-supported TTS model
            voice="alloy",
            input=text,
            format="mp3"
        )

        audio_bytes = response.read()  # IMPORTANT
        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
