# tts.py
import os
import uuid
from gtts import gTTS
from pathlib import Path

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Returns relative path to saved mp3 file.
    """
    fname = f"answer_{uuid.uuid4().hex[:8]}.mp3"
    path = AUDIO_DIR / fname
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(path))
    return str(path)
