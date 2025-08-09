from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
import os

DEFAULT_INDEX_DIR = 'indexes'
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')

def build_index(docs, model_name=None):
    model = model_name or EMBEDDING_MODEL
    emb = SentenceTransformerEmbeddings(model_name=model)
    db = FAISS.from_documents(docs, emb)
    return db, emb

def save_index(db, name):
    p = Path(DEFAULT_INDEX_DIR) / name
    p.mkdir(parents=True, exist_ok=True)
    db.save_local(str(p))
    return str(p)

def load_index(name):
    emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    p = Path(DEFAULT_INDEX_DIR) / name
    if not p.exists():
        raise FileNotFoundError(f'Index not found: {p}')
    db = FAISS.load_local(str(p), emb)
    return db, emb

def list_indexes():
    p = Path(DEFAULT_INDEX_DIR)
    if not p.exists():
        return []
    return [d.name for d in p.iterdir() if d.is_dir()]
