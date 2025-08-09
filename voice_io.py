import tempfile
from gtts import gTTS

def text_to_speech_bytes(text, lang='en'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    gTTS(text=text, lang=lang).save(tmp.name)
    return tmp.name
