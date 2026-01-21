"""Tickets/historical failures vector store."""

from typing import List, Optional
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
from pathlib import Path

from knoroute.config import settings


class TicketsVectorStore:
    """
    Vector store for historical failures/tickets with metadata schema:
    - ticket_id: unique identifier
    - status: open, closed, resolved
    - severity: critical, high, medium, low
    - created_at: timestamp
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the tickets vector store."""
        if persist_directory is None:
            persist_directory = str(Path(settings.vector_store_path) / "tickets_db")
        
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="tickets_collection",
            embedding_function=self.embeddings,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add ticket documents to the vector store.
        
        Args:
            documents: List of Document objects with ticket metadata
            
        Returns:
            List of document IDs
        """
        # Validate metadata schema
        for doc in documents:
            if "ticket_id" not in doc.metadata:
                raise ValueError("Document metadata must include 'ticket_id'")
            
            # Set defaults for optional fields
            doc.metadata.setdefault("status", "open")
            doc.metadata.setdefault("severity", "medium")
            doc.metadata.setdefault("created_at", datetime.now().isoformat())
        
        return self.vectorstore.add_documents(documents)
    
    def get_retriever(self, k: int = None, use_compression: bool = False):
        """
        Get a retriever for the tickets store.
        
        Args:
            k: Number of documents to retrieve
            use_compression: Whether to use contextual compression
            
        Returns:
            Retriever instance
        """
        if k is None:
            k = settings.retrieval_top_k
        
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        if use_compression:
            llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        
        return base_retriever
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on tickets.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters (e.g., {"severity": "critical"})
            
        Returns:
            List of relevant ticket documents
        """
        if k is None:
            k = settings.retrieval_top_k
        
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.similarity_search(query, **search_kwargs)
    
    def search_by_severity(self, severity: str, k: int = 5) -> List[Document]:
        """
        Search tickets by severity level.
        
        Args:
            severity: Severity level (critical, high, medium, low)
            k: Number of results
            
        Returns:
            List of matching tickets
        """
        return self.vectorstore.similarity_search(
            query="",
            k=k,
            filter={"severity": severity}
        )
    
    def search_by_status(self, status: str, k: int = 5) -> List[Document]:
        """
        Search tickets by status.
        
        Args:
            status: Status (open, closed, resolved)
            k: Number of results
            
        Returns:
            List of matching tickets
        """
        return self.vectorstore.similarity_search(
            query="",
            k=k,
            filter={"status": status}
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection("tickets_collection")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.client.get_collection("tickets_collection")
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
