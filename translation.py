from deep_translator import GoogleTranslator

def translate_to_hindi(text):
    return GoogleTranslator(source='auto', target='hi').translate(text)
