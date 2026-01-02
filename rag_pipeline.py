from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import load_documents, split_documents

PERSIST_DIRECTORY = "vector_store"

def create_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    vectorstore.persist()
    return vectorstore


if __name__ == "__main__":
    create_vector_store()
    print("Vector store created and persisted successfully (local embeddings).")
