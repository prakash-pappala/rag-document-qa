# RAG Document QA

Upload a PDF or text file and ask questions about it. Uses LangChain + HuggingFace + FAISS under the hood.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Document gets split into small chunks
2. Each chunk is embedded using `sentence-transformers`
3. Stored in a FAISS vector index
4. When you ask a question, the most relevant chunks are retrieved
5. Those chunks + your question go into `flan-t5` to generate an answer

## Stack

- LangChain
- HuggingFace Transformers
- FAISS
- Streamlit
