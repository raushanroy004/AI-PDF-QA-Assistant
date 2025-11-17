# utils/tts_utils.py

import os
import tempfile
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def text_to_speech(text):
    """
    Convert text into mp3 speech using Groq TTS
    Returns path to mp3 file.
    """

    if not text or text.strip() == "":
        return None

    try:
        # Call Groq TTS API
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        # Save audio to temp mp3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            mp3_path = tmp.name
            tmp.write(response.audio)

        return mp3_path

    except Exception as e:
        print("TTS ERROR:", e)
        return None
