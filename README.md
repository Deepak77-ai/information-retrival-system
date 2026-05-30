# 📄 Information Retrieval System — RAG-Powered PDF Q&A

> Upload any PDF, ask questions in plain English, and get accurate answers — powered by **LangChain + FAISS + Groq (Qwen3-32B)**.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://information-retrival-system-hjpxqg5wntyc9zsdwnryvt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-F55036?style=for-the-badge)](https://groq.com/)

---

## 🚀 Live Demo

👉 **[Open the App](https://information-retrival-system-hjpxqg5wntyc9zsdwnryvt.streamlit.app/)** *(may take a few seconds to load)*

---

## 📸 Screenshot

<img width="1912" height="962" alt="image" src="https://github.com/user-attachments/assets/f35ff025-203d-43a8-9c91-9588d7c339a0" />


> Upload a PDF → click **Submit & Process** → ask any question about its contents → get a conversational, context-aware answer.

---

## 🧠 What It Does

This app lets you **chat with your PDF documents** using AI. Instead of reading through long files, just ask what you need to know.

**Example use cases:**
- Q&A from research papers or study notes
- Querying company policy or HR documents
- Exploring reports, contracts, or resumes
- Knowledge retrieval from technical documentation

---

## 🏗️ How It Works (RAG Pipeline)

```
User uploads PDF(s)
        ↓
Text extracted page-by-page  (PyPDF2)
        ↓
Split into overlapping chunks  (chunk_size=1000, overlap=20)
        ↓
Chunks → Vector Embeddings  (HuggingFace Sentence Transformers)
        ↓
Stored in FAISS Vector Database
        ↓
User asks a question
        ↓
Semantically similar chunks retrieved from FAISS
        ↓
Qwen3-32B (via Groq) generates answer from retrieved context
        ↓
Conversation history maintained for follow-up questions
```

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.8+ | Core language |
| Streamlit | Web app UI |
| LangChain | RAG & conversational chain |
| FAISS | Vector similarity search |
| HuggingFace Sentence Transformers | Text embeddings |
| Groq — Qwen3-32B | LLM inference (fast & free) |
| PyPDF2 | PDF text extraction |

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd information-retrival-system
```

**2. Create & activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key** — create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free Groq API key at [console.groq.com](https://console.groq.com/)

**5. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
information-retrival-system/
├── app.py                  # Streamlit UI & app entry point
├── src/
│   ├── __init__.py
│   └── helper.py           # RAG pipeline (PDF → chunks → embeddings → LLM)
├── research/
│   └── trials.ipynb        # Experimentation notebook
├── setup.py                # Project packaging
├── requirements.txt        # Dependencies
└── .env                    # API keys (not committed)
```

---

## 📬 Contact

Made by **Deepak** · [GitHub](https://github.com/Deepak77-ai)
