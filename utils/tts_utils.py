import streamlit as st
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

def text_to_speech(text: str):
    try:
        print("DEBUG: Calling Groq TTS...")

        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            format="mp3"
        )

        print("DEBUG: TTS response =", response)

        audio_bytes = response.read()
        print("DEBUG: Audio bytes length =", len(audio_bytes))

        return audio_bytes

    except Exception as e:
        print("TTS ERROR:", e)
        return None
