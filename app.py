import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from modules.pdf_loader import extract_text_from_pdf
from modules.vector_store import build_index, load_index, save_index, list_indexes
from modules.llm_gpt import get_openai_llm
from modules.llm_gemini import get_gemini_llm
from modules.chat_handler import make_chain
from modules.voice_io import text_to_speech_bytes
from modules.translation import translate_to_hindi

st.set_page_config(page_title='AI Study Buddy v2', layout='wide')
st.title('AI Study Buddy — v2 (PDF → Chat)')

with st.sidebar:
    st.header('Settings')
    provider = st.selectbox('LLM Provider', ['OpenAI','Gemini'])
    temperature = st.slider('Temperature', 0.0, 1.0, 0.0, 0.05)
    chunk_size = st.number_input('Chunk size', value=1000, step=100)
    chunk_overlap = st.number_input('Chunk overlap', value=200, step=50)

uploaded = st.file_uploader('Upload PDF', type=['pdf'])
if uploaded:
    tmp_dir = Path('temp_uploads'); tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / uploaded.name
    with open(path, 'wb') as f:
        f.write(uploaded.getbuffer())
    st.success(f'Saved upload to {path}')
    if st.button('Extract & Build Index'):
        with st.spinner('Extracting text & building index...'):
            docs = extract_text_from_pdf(str(path))
            db, embeddings = build_index(docs)
            name = uploaded.name.replace(' ','_').rsplit('.',1)[0]
            save_index(db, name)
            st.success(f'Index built and saved as {name}')
            st.session_state['db'] = db
            st.session_state['emb'] = embeddings

if 'db' not in st.session_state:
    existing = list_indexes()
    if existing:
        sel = st.selectbox('Load index', options=existing)
        if st.button('Load'):
            db, emb = load_index(sel)
            st.session_state['db'] = db
            st.session_state['emb'] = emb
            st.success(f'Loaded index {sel}')

if 'db' in st.session_state:
    st.subheader('Chat')
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    q = st.text_input('Ask a question about the PDF:')
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button('Ask') and q.strip():
            try:
                if provider=='OpenAI':
                    from modules.llm_gpt import get_openai_llm
                    llm = get_openai_llm()
                else:
                    from modules.llm_gemini import get_gemini_llm
                    llm = get_gemini_llm()
            except Exception as e:
                st.error(f'LLM init failed: {e}')
                from modules.llm_gpt import get_openai_llm
                llm = get_openai_llm()
            retriever = st.session_state['db'].as_retriever(search_kwargs={'k':4})
            chain = make_chain(llm, retriever)
            with st.spinner('Generating...'):
                out = chain({'question': q, 'chat_history': st.session_state['history']})
            answer = out.get('answer','')
            st.session_state['history'].append((q,answer))
            st.write('**Answer:**'); st.write(answer)
            srcs = out.get('source_documents',[])
            if srcs:
                st.write('**Top sources:**')
                for s in srcs[:3]:
                    meta = getattr(s,'metadata',{})
                    st.write(f"Page: {meta.get('page','n/a')}") 
                    st.write(s.page_content[:400] + ('...' if len(s.page_content)>400 else ''))
            if st.button('Play TTS'):
                mp3 = text_to_speech_bytes(answer)
                st.audio(mp3)
    with col2:
        if st.button('Translate last answer to Hindi'):
            if st.session_state['history']:
                last = st.session_state['history'][-1][1]
                st.write(translate_to_hindi(last))
            else:
                st.info('No answer yet.')
