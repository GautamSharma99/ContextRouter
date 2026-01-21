# vectorstores/docs_db.py

import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from knoroute.config import settings


def get_docs_vectorstore() -> Chroma:
    """
    Creates or loads the Docs vector database.
    Stores FastAPI documentation embeddings.
    """

    embedding = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )

    persist_directory = str(Path(settings.vector_store_path) / "docs_db")

    vectordb = Chroma(
        collection_name="fastapi_docs",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    return vectordb
