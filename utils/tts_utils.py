import os
import base64
import tempfile
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def text_to_speech(text):
    """
    Convert text to MP3 using Groq TTS (correct base64 decoding)
    Returns path to saved mp3 file.
    """

    if not text or text.strip() == "":
        return None

    try:
        # Request TTS audio
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        # 🔥 Groq returns base64-encoded audio → decode it properly
        audio_b64 = response.data
        audio_bytes = base64.b64decode(audio_b64)

        # Save as .mp3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            mp3_path = tmp.name
            tmp.write(audio_bytes)

        return mp3_path

    except Exception as e:
        print("TTS ERROR:", e)
        return None
