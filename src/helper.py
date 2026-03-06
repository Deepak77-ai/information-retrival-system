import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader #Used to read PDF files page by page

from langchain_text_splitters import RecursiveCharacterTextSplitter #✂️ Used to cut big text into small pieces
#Why?
#LLMs can’t read very long text at once

from langchain_community.embeddings import HuggingFaceEmbeddings #🧠 Converts text → numbers
from langchain_community.vectorstores import FAISS #📦 FAISS = fast search box
#Stores text embeddings and helps find similar text.

from langchain_classic.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory

from langchain_groq import ChatGroq #This is your AI brain (LLM) running on Groq servers.

load_dotenv() #Reads .env file and loads environment variables into the program. This is where we get our GROQ_API_KEY from.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY #This gives permission to use Groq’s AI.

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversational_chain(vectorstore):
    llm = ChatGroq(model="openai/gpt-oss-120b")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain
