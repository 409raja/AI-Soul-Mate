try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

import os

def get_gemini_llm(model_name=None, temperature=0.0):
    if not GEMINI_AVAILABLE:
        raise RuntimeError('Gemini provider not installed.')
    model = model_name or os.getenv('GEMINI_MODEL','models/gemini-1.5')
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
