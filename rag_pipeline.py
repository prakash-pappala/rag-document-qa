import os
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0
    )


def load_and_split(uploaded_file):
    ext = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(uploaded_file.read())
        path = f.name
    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
    else:
        docs = TextLoader(path, encoding="utf-8").load()
    os.unlink(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain(vectorstore, llm):
    return {"vectorstore": vectorstore, "llm": llm}


def ask(qa_chain, question):
    vectorstore = qa_chain["vectorstore"]
    llm = qa_chain["llm"]

    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer in one short sentence using only the context below.
If the answer is not in the context, say I don't know.

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()
    sources = [doc.page_content[:300] for doc in docs]
    return answer, sources