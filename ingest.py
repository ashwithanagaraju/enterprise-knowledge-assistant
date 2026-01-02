import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

DOCS_PATH = "sample_docs"

def load_documents():
    documents = []

    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, filename))
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} document(s)")
    print(f"Split into {len(chunks)} chunks")
