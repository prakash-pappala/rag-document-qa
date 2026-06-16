# RAG Document QA

Upload a PDF or text file and ask questions about it. Uses LangChain + FAISS + Groq (Llama3) under the hood.

## Setup

pip install -r requirements.txt

Set your Groq API key:

set GROQ_API_KEY=your_key_here

Then run:

streamlit run app.py

## How it works

1. Document gets split into small chunks
2. Each chunk is embedded using sentence-transformers
3. Stored in a FAISS vector index
4. When you ask a question, the most relevant chunks are retrieved
5. Those chunks + your question go into Llama3 via Groq API to generate an answer

## Stack

- LangChain
- HuggingFace sentence-transformers
- FAISS
- Groq API (Llama3)
- Streamlit