import streamlit as st
from groq import Groq

# Load secrets (Streamlit Cloud)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def text_to_speech(text: str):
    """
    Convert text → speech using Groq Whisper-TTS.
    Returns raw audio bytes for Streamlit st.audio().
    """

    try:
        response = client.audio.speech.create(
            model="whisper-tts",        # ✅ Correct Groq TTS model
            input=text,                # text to convert
            format="mp3"               # return mp3 bytes
        )

        audio_bytes = response.read()  # raw audio bytes
        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
