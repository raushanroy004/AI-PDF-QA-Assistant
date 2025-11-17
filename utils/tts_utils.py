from gtts import gTTS
import tempfile
import os
import re

def clean_for_tts(text):
    """
    Removes characters that cause gTTS to create empty audio files.
    """
    if not text:
        return ""

    # remove emojis and non-ASCII chars
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove repeated spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_speech(text):
    """
    Convert text to speech using gTTS (100% works on Streamlit Cloud).
    Returns path to an mp3 file.
    """

    text = clean_for_tts(text)

    if text.strip() == "":
        return None

    try:
        tts = gTTS(text=text, lang='en')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        # Double-check file size (must not be zero)
        if os.path.getsize(tmp_path) < 2000:
            return None
        
        return tmp_path

    except Exception as e:
        print("TTS ERROR:", e)
        return None
