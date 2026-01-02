# enterprise-knowledge-assistant
An LLM-powered knowledge assistant for querying internal documents using retrieval-augmented generation (RAG).
The system performs semantic retrieval over embedded documents and generates grounded answers constrained to retrieved context.

#3 Live deployment:
https://enterprise-knowledge-assistant-b9v25ca3kie5uvwqtkdle9.streamlit.app/

Built using local embeddings and local model inference to ensure cost efficiency, transparency, and deterministic behavior suitable for enterprise environments.

Stack: Python, LangChain, ChromaDB, Sentence Transformers, Hugging Face Transformers, Streamlit
