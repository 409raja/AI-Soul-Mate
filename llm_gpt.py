from langchain.chat_models import ChatOpenAI
import os

def get_openai_llm(model_name=None, temperature=0.0):
    params = {'temperature': temperature}
    if model_name:
        params['model_name'] = model_name
    elif os.getenv('OPENAI_MODEL'):
        params['model_name'] = os.getenv('OPENAI_MODEL')
    else:
        params['model_name'] = 'gpt-3.5-turbo'
    return ChatOpenAI(**params)
