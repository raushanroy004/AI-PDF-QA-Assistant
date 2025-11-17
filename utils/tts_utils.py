from gtts import gTTS
import tempfile
import os


def text_to_speech(text: str):
    """
    Convert text to speech (mp3) and return the temporary file path.

    - Uses gTTS (Google Text-to-Speech)
    - Creates a safe temporary file compatible with Streamlit Cloud
    - Handles empty text and TTS errors gracefully
    """

    if not text or not text.strip():
        return None

    try:
        # Generate speech
        tts = gTTS(text=text, lang="en")

        # Create a temp file for Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name

        # Save mp3 output
        tts.save(temp_path)

        return temp_path

    except Exception as e:
        print(f"[TTS Error] {e}")

        # Fallback: create a silent mp3 if TTS fails
        fallback_path = create_silent_audio()
        return fallback_path



def create_silent_audio(duration_ms: int = 800):
    """
    Generates a silent MP3 file to avoid Streamlit crashing
    when TTS fails because of API block or offline status.
    """

    try:
        import wave
        import numpy as np

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name

        sample_rate = 16000
        num_samples = int(sample_rate * (duration_ms / 1000))

        silence = np.zeros(num_samples, dtype=np.int16)

        with wave.open(path, "w") as f:
            f.setnchannels(1)       # mono
            f.setsampwidth(2)       # 2 bytes
            f.setframerate(sample_rate)
            f.writeframes(silence.tobytes())

        return path

    except Exception:
        return None
