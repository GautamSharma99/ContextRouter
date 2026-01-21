# vectorstores/docs_db.py

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def get_docs_vectorstore() -> Chroma:
    """
    Creates or loads the Docs vector database.
    Stores FastAPI documentation embeddings.
    """

    embedding = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    persist_directory = "vectorstores/chroma/docs_db"

    vectordb = Chroma(
        collection_name="fastapi_docs",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    return vectordb
