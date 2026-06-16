import streamlit as st
from rag_pipeline import load_llm, load_and_split, build_vectorstore, build_qa_chain, ask

st.set_page_config(page_title="Doc QA", page_icon="📄")
st.title("📄 Document Q&A")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Upload a Document")
    file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])

    if file and st.button("Process"):
        with st.spinner("Reading and indexing..."):
            chunks = load_and_split(file)
            vs = build_vectorstore(chunks)
            llm = load_llm()
            st.session_state.qa_chain = build_qa_chain(vs, llm)
            st.session_state.history = []
        st.success("Done! Ask something below.")

# show chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("Sources"):
                for chunk in msg["sources"]:
                    st.caption(chunk)

# input
question = st.chat_input("Ask a question about your document...")

if question:
    if st.session_state.qa_chain is None:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask(st.session_state.qa_chain, question)
            st.write(answer)
            with st.expander("Sources"):
                for chunk in sources:
                    st.caption(chunk)

        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
