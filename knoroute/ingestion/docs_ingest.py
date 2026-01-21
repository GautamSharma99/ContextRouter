# ingestion/docs_ingest.py
import sys
import os

# Add the project root to sys.path so knoroute can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from knoroute.vectorstores.docs_db import get_docs_vectorstore


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

DOCS_PATH = "data/docs"

IGNORED_DIRS = {
    "img",
    "css",
    "js",
    "learn",
    "tutorial",
    "about",
    "about",
    ".github",
}

MIN_CONTENT_LENGTH = 200  # skip tiny markdown files


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def should_ignore(path: Path) -> bool:
    """
    Ignore non-documentation folders.
    """
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True
    return False


def load_markdown_files(base_path: str) -> List[Document]:
    """
    Load markdown files recursively with filtering.
    """
    documents = []

    for md_file in Path(base_path).rglob("*.md"):
        if should_ignore(md_file):
            continue

        loader = TextLoader(str(md_file), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            if len(doc.page_content) < MIN_CONTENT_LENGTH:
                continue

            doc.metadata = {
                "source": "docs",
                "file": md_file.name,
                "path": str(md_file),
                "topic": md_file.parent.name,
            }

            documents.append(doc)

    return documents


def split_by_headers(documents: List[Document]) -> List[Document]:
    """
    Header-based semantic chunking with empty-chunk filtering.
    """

    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    chunks: List[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)

        for chunk in splits:
            # 🔥 CRITICAL FIX
            if not chunk.page_content.strip():
                continue

            if len(chunk.page_content.strip()) < 50:
                continue

            chunk.metadata.update(doc.metadata)
            chunks.append(chunk)

    return chunks

# ---------------------------------------------------
# INGESTION PIPELINE
# ---------------------------------------------------

def ingest_docs():
    print("📘 Loading FastAPI documentation...")

    raw_docs = load_markdown_files(DOCS_PATH)
    print(f"✅ Loaded {len(raw_docs)} documentation files")

    print("✂️  Splitting docs by markdown headers...")
    chunks = split_by_headers(raw_docs)
    print(f"✅ Created {len(chunks)} semantic chunks")

    print("🧠 Storing embeddings in Docs Vector DB...")
    vectordb = get_docs_vectorstore()

    if not chunks:
        print("⚠️ No chunks to ingest. Exiting...")
        return

    vectordb.add_documents(chunks)
    # vectordb.persist() # Auto-persisted in newer Chroma versions

    print("🎉 Docs ingestion completed successfully")


if __name__ == "__main__":
    ingest_docs()
