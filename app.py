import time
import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversational_chain
)

def user_input(user_question):
    if st.session_state.conversational_chain is None:
        st.warning("Please upload and process PDF files first.")
        return

    response = st.session_state.conversational_chain({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("User:", message.content)
        else:
            st.write("Reply:", message.content)

def main():
    st.set_page_config(page_title="Information Retrieval System", layout="wide")
    st.header("Information Retrieval System 🤗")

    user_question = st.text_input("Ask a question about the content of the PDF files:")

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversational_chain = get_conversational_chain(vectorstore)

            st.success("Done! You can now ask questions about the content of the PDF files.")

if __name__ == "__main__":
    main()