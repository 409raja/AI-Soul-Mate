import fitz
from langchain.schema import Document

def extract_text_from_pdf(path):
    docs = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            meta = {'page': i+1, 'source': path}
            docs.append(Document(page_content=text, metadata=meta))
    return docs
