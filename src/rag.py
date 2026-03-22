import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embed_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_db():
    p = "./chroma_db"
    f = "knowledge/ev_guidelines.txt"
    
    if os.path.exists(p):
        return Chroma(persist_directory=p, embedding_function=embed_fn)
    
    if not os.path.exists(f):
        return None
        
    ldr = TextLoader(f)
    docs = ldr.load()
    
    split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = split.split_documents(docs)
    
    db = Chroma.from_documents(chunks, embed_fn, persist_directory=p)
    return db

def retrieve_best_guidelines(query, k=3):
    db = setup_db()
    
    if db is None:
        return "no guidelines available"
        
    res = db.similarity_search(query, k=k)
    
    out = ""
    for r in res:
        out += r.page_content + "\n\n"
        
    return out
