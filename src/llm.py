import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=0)
    return llm

def ask_llm(prompt):
    llm = get_llm()
    res = llm.invoke(prompt)
    return res.content
