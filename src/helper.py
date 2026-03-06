```python
import os
from dotenv import load_dotenv

# Used to read and extract text from PDF files
from PyPDF2 import PdfReader

# LangChain utility that splits long text into smaller chunks
# This is important because LLMs and embeddings work better on smaller pieces of text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace embedding model converts text into numerical vectors
# FAISS is used to store and search these vectors efficiently
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ConversationalRetrievalChain connects the LLM with the vector database
# and retrieves relevant document chunks while answering questions
from langchain.chains import ConversationalRetrievalChain

# Memory module stores previous chat messages so the conversation feels continuous
from langchain.memory import ConversationBufferMemory

# Groq provides very fast inference for LLM models
from langchain_groq import ChatGroq


# Load environment variables from the .env file
# This is where the API keys are stored
load_dotenv()

# Get the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If the API key is missing, stop the application immediately
# This avoids runtime errors later when the model is called
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is missing. Add it to Streamlit secrets.")


# -----------------------------------------------------------
# Function: Extract text from uploaded PDF files
# -----------------------------------------------------------
def get_pdf_text(pdf_docs):

    text = ""

    # Loop through each uploaded PDF
    for pdf in pdf_docs:

        # Create a PDF reader object
        pdf_reader = PdfReader(pdf)

        # Extract text page by page
        # This prevents memory issues for large PDFs
        for page in pdf_reader.pages:

            page_text = page.extract_text()

            # Some pages might not contain text (e.g., scanned images)
            if page_text:
                text += page_text

    return text


# -----------------------------------------------------------
# Function: Split long text into smaller chunks
# -----------------------------------------------------------
def get_text_chunks(text):

    # RecursiveCharacterTextSplitter intelligently breaks text
    # without cutting sentences abruptly when possible
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # maximum characters per chunk
        chunk_overlap=20   # small overlap helps preserve context between chunks
    )

    chunks = text_splitter.split_text(text)

    return chunks


# -----------------------------------------------------------
# Function: Create vector database from text chunks
# -----------------------------------------------------------
def get_vectorstore(chunks):

    # Convert text into embeddings using a HuggingFace model
    embeddings = HuggingFaceEmbeddings()

    # Store the embeddings inside FAISS
    # FAISS enables fast similarity search when retrieving relevant chunks
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


# -----------------------------------------------------------
# Function: Create conversational retrieval chain
# -----------------------------------------------------------
def get_conversational_chain(vectorstore):

    # Initialize the LLM from Groq
    # qwen3-32b is fast and works well for question answering tasks
    llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        groq_api_key=GROQ_API_KEY
    )

    # Memory object stores previous conversation history
    # This helps the chatbot maintain context across multiple questions
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # ConversationalRetrievalChain does three things:
    # 1. Receives the user question
    # 2. Retrieves relevant chunks from the vector database
    # 3. Sends those chunks + question to the LLM to generate the answer
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversational_chain
```
