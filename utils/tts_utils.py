# utils/tts_utils.py

import pyttsx3
from pydub import AudioSegment
import tempfile
import os

# Initialize engine once
engine = pyttsx3.init()


def text_to_speech(text):
    """
    Generate real MP3 audio using offline TTS (pyttsx3 + pydub)
    Returns path to MP3 file
    """
    if not text or text.strip() == "":
        return None

    try:
        # Temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        # Save speech to WAV
        engine.save_to_file(text, wav_path)
        engine.runAndWait()

        # Convert WAV → MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            mp3_path = tmp_mp3.name

        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")

        # Cleanup
        os.remove(wav_path)

        return mp3_path

    except Exception as e:
        print("TTS ERROR:", e)
        return None
