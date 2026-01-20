"""Documentation ingestion pipeline."""

import os
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

from knoroute.vectorstores import DocsVectorStore


class DocsIngestionPipeline:
    """
    Ingestion pipeline for documentation files.
    Supports markdown and text files with hierarchical structure preservation.
    """
    
    def __init__(self, vector_store: Optional[DocsVectorStore] = None):
        """
        Initialize the documentation ingestion pipeline.
        
        Args:
            vector_store: DocsVectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or DocsVectorStore()
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    
    def load_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.md",
        doc_type: str = "guide"
    ) -> List[Document]:
        """
        Load all documentation files from a directory.
        
        Args:
            directory_path: Path to documentation directory
            glob_pattern: File pattern to match (default: markdown files)
            doc_type: Type of documentation (API, guide, reference, tutorial)
            
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["doc_type"] = doc_type
            doc.metadata["source"] = doc.metadata.get("source", "")
            doc.metadata["section"] = self._extract_section(doc.metadata.get("source", ""))
            doc.metadata["version"] = "latest"
        
        return documents
    
    def load_file(
        self,
        file_path: str,
        doc_type: str = "guide",
        section: str = "root"
    ) -> List[Document]:
        """
        Load a single documentation file.
        
        Args:
            file_path: Path to the file
            doc_type: Type of documentation
            section: Section/category
            
        Returns:
            List of documents (may be chunked)
        """
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["doc_type"] = doc_type
            doc.metadata["source"] = file_path
            doc.metadata["section"] = section
            doc.metadata["version"] = "latest"
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using markdown-aware splitting.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)
    
    def ingest_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.md",
        doc_type: str = "guide",
        chunk: bool = True
    ) -> List[str]:
        """
        Complete ingestion pipeline for a directory.
        
        Args:
            directory_path: Path to documentation directory
            glob_pattern: File pattern to match
            doc_type: Type of documentation
            chunk: Whether to chunk documents
            
        Returns:
            List of document IDs
        """
        # Load documents
        documents = self.load_directory(directory_path, glob_pattern, doc_type)
        
        # Chunk if requested
        if chunk:
            documents = self.chunk_documents(documents)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} document chunks from {directory_path}")
        return doc_ids
    
    def ingest_file(
        self,
        file_path: str,
        doc_type: str = "guide",
        section: str = "root",
        chunk: bool = True
    ) -> List[str]:
        """
        Complete ingestion pipeline for a single file.
        
        Args:
            file_path: Path to the file
            doc_type: Type of documentation
            section: Section/category
            chunk: Whether to chunk the document
            
        Returns:
            List of document IDs
        """
        # Load document
        documents = self.load_file(file_path, doc_type, section)
        
        # Chunk if requested
        if chunk:
            documents = self.chunk_documents(documents)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} chunks from {file_path}")
        return doc_ids
    
    def _extract_section(self, file_path: str) -> str:
        """
        Extract section from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Section name (parent directory or 'root')
        """
        path = Path(file_path)
        if len(path.parts) > 1:
            return path.parts[-2]  # Parent directory name
        return "root"


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DocsIngestionPipeline()
    
    # Example: Ingest a documentation directory
    # pipeline.ingest_directory(
    #     directory_path="./docs",
    #     glob_pattern="**/*.md",
    #     doc_type="guide"
    # )
    
    print("Documentation ingestion pipeline ready.")
    print("Use pipeline.ingest_directory() or pipeline.ingest_file() to add documents.")
