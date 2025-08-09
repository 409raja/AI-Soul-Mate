from langchain.chains import ConversationalRetrievalChain

def make_chain(llm, retriever):
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
