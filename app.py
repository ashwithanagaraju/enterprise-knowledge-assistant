import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

PERSIST_DIRECTORY = "vector_store"

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="centered")
st.title("Enterprise Knowledge Assistant")

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return retriever, llm


retriever, llm = load_components()

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Searching knowledge base..."):
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer the question strictly using the context below.
If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

        answer = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source Documents")
    for doc in docs:
        st.write(f"- {doc.page_content[:300]}...")

