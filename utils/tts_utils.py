from gtts import gTTS
import tempfile
import os

def text_to_speech(text):
    """
    Convert text answer into speech (mp3), return file path.
    """
    if not text or text.strip() == "":
        return None

    try:
        # Create TTS object
        tts = gTTS(text=text, lang='en')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        return tmp_path

    except Exception as e:
        print("TTS Error:", e)
        return None