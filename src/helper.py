import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is missing. Add it to Streamlit secrets.")


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversational_chain(vectorstore):

    llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    groq_api_key=GROQ_API_KEY
)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversational_chain


cross check it with my file is it is correct working ?
