import os
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq


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
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer in one short sentence using only the context below.
If the answer is not in the context, say I don't know.

Context:
{context}

Question: {question}
Answer:"""
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


def ask(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    answer = result["result"].strip()
    sources = [doc.page_content[:300] for doc in result["source_documents"]]
    return answer, sources