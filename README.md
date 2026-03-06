# 📄 Information Retrieval System using Generative AI (RAG)
# 🔍 Project Overview

This project is a PDF-based Information Retrieval System built using Generative AI and Retrieval-Augmented Generation (RAG).
It allows users to upload one or multiple PDF documents and ask questions related to their content.
The system retrieves relevant information from the PDFs and generates accurate answers using a Large Language Model (LLM).

see live demo(sometimes app takes few sec to load) --> https://information-retrival-system-hjpxqg5wntyc9zsdwnryvt.streamlit.app/ 


# 🚀 Key Features

📄 Upload and process multiple PDF files

✂️ Automatic text extraction and chunking

🧠 Semantic search using vector embeddings

📦 FAISS-based vector database for fast retrieval

💬 Conversational question-answering with memory

⚡ Fast inference using Groq-hosted LLM

🖥️ Interactive UI built with Streamlit

# 🏗️ Project Architecture
```
project_root/
│
├── src/
│   ├── __init__.py
│   └── helper.py          # Core backend logic (PDF, embeddings, RAG)
│
├── research/
│   └── trials.ipynb       # Experimentation and testing
│
├── app.py                 # Streamlit application (UI)
├── setup.py               # Project packaging configuration
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API keys)
└── README.md
```
# 🔄 System Flow (RAG Pipeline)

User uploads PDF documents

Text is extracted from each PDF page

Text is split into overlapping chunks

Chunks are converted into embeddings

Embeddings are stored in FAISS vector store

User asks a question

Relevant chunks are retrieved using semantic search

LLM generates an answer based on retrieved context

Conversation history is maintained for follow-up questions

# 🧠 Technologies Used

Python

Streamlit – Web application UI

LangChain – RAG and conversational pipeline

FAISS – Vector database for similarity search

HuggingFace Sentence Transformers – Embedding generation

Groq LLM – Large Language Model inference

PyPDF2 – PDF text extraction

# 📦 Installation & Setup
1️⃣ Clone the Repository
```
git clone <your-repository-url>
cd <project-folder>
```
2️⃣ Create Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Set Environment Variables
```
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
```
▶️ Running the Application
```
streamlit run app.py
```

Then open the browser link shown in the terminal.

# 🖥️ How to Use

1. Upload one or more PDF files using the sidebar

2. Click Submit & Process

3. Wait for processing to complete

4. Ask questions related to the uploaded PDFs

5. Continue the conversation with follow-up questions

# 🧪 Example Use Cases

Question answering from study materials

Research paper exploration

Company policy document analysis

Resume or report-based Q&A

Learning and knowledge retrieval systems

# 📌 Packaging the Project

This project uses setup.py for packaging.

To install the project in editable mode:
```
pip install -e .
```
