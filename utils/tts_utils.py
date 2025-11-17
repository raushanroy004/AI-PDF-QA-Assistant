from gtts import gTTS
import io
import re


def clean_for_tts(text: str) -> str:
    """
    Clean text so gTTS doesn't create broken/empty audio.
    - remove emojis / non-ASCII
    - collapse whitespace
    """
    if not text:
        return ""

    # remove non-ASCII chars (emojis etc.)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def text_to_speech(text: str):
    """
    Convert text to speech using gTTS and return audio bytes (BytesIO).

    Returns:
        BytesIO object containing mp3 audio, or None on failure.
    """
    text = clean_for_tts(text)

    if not text:
        return None

    try:
        # Create in-memory bytes buffer
        mp3_bytes = io.BytesIO()

        # Generate TTS directly into the buffer
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(mp3_bytes)

        # Rewind buffer to the beginning
        mp3_bytes.seek(0)

        return mp3_bytes

    except Exception as e:
        # Will show up in Streamlit "Manage App → Logs"
        print("TTS ERROR:", e)
        return None
