from gtts import gTTS
import tempfile
import os
import soundfile as sf
import numpy as np


def text_to_speech(text):
    """
    Convert text into WAV audio (Streamlit-safe).
    gTTS → MP3 → WAV (always plays correctly)
    """
    if not text or text.strip() == "":
        return None

    try:
        # Step 1 — Generate temporary MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            mp3_path = tmp_mp3.name
            tts = gTTS(text=text, lang="en")
            tts.save(mp3_path)

        # Step 2 — Convert MP3 → WAV (universal compatibility)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

            # Read MP3 → convert to WAV
            data, samplerate = sf.read(mp3_path, dtype="float32")
            sf.write(wav_path, data, samplerate)

        return wav_path

    except Exception as e:
        print("TTS Error:", e)
        return None
