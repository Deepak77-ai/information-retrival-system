from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_docs):
    """
    Reads multiple PDF files and extracts their text content.

    This function loops through each uploaded PDF, reads every page,
    and concatenates all extracted text into a single string so it
    can be processed later by the embedding pipeline.
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        # Extract text page by page
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    """
    Splits large text into smaller overlapping chunks.

    Language models perform better when the input text is broken
    into manageable pieces. Overlap helps preserve context between
    consecutive chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,   # maximum size of each chunk
        chunk_overlap=1000  # overlapping region between chunks
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Converts text chunks into embeddings and stores them in a FAISS index.

    Embeddings allow semantic similarity search. FAISS is used here
    as a fast vector database so relevant chunks can be retrieved
    during question answering.
    """

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Build FAISS vector store from text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save locally so it can be reused without recomputing embeddings
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """
    Creates the QA chain used to answer user questions.

    The prompt is designed to keep answers grounded in the
    provided context. If the answer is not present in the
    retrieved text, the model should clearly state that.
    """

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say
    "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Load standard QA chain from LangChain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
