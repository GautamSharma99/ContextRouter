"""Memory/learned knowledge vector store."""

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


class MemoryVectorStore:
    """
    Vector store for learned knowledge with metadata schema:
    - learned_from: query that generated insight
    - confidence: 0.0-1.0 score
    - created_at: timestamp
    - tags: categorization
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the memory vector store."""
        if persist_directory is None:
            persist_directory = str(Path(settings.vector_store_path) / "memory_db")
        
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
            collection_name="memory_collection",
            embedding_function=self.embeddings,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add memory documents to the vector store.
        
        Args:
            documents: List of Document objects with memory metadata
            
        Returns:
            List of document IDs
        """
        # Validate metadata schema
        for doc in documents:
            if "learned_from" not in doc.metadata:
                raise ValueError("Document metadata must include 'learned_from'")
            
            # Set defaults for optional fields
            doc.metadata.setdefault("confidence", 0.8)
            doc.metadata.setdefault("created_at", datetime.now().isoformat())
            doc.metadata.setdefault("tags", "")
        
        return self.vectorstore.add_documents(documents)
    
    def add_insight(
        self,
        insight: str,
        learned_from: str,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a learned insight to memory.
        
        Args:
            insight: The learned knowledge
            learned_from: Original query that generated this insight
            confidence: Confidence score (0.0-1.0)
            tags: Optional categorization tags
            
        Returns:
            Document ID
        """
        # Check for duplicates
        similar_docs = self.similarity_search(insight, k=1)
        if similar_docs and similar_docs[0].page_content.strip() == insight.strip():
            # Already exists, don't add duplicate
            return None
        
        doc = Document(
            page_content=insight,
            metadata={
                "learned_from": learned_from,
                "confidence": confidence,
                "created_at": datetime.now().isoformat(),
                "tags": ",".join(tags) if tags else ""
            }
        )
        
        ids = self.add_documents([doc])
        return ids[0] if ids else None
    
    def get_retriever(self, k: int = None, use_compression: bool = False):
        """
        Get a retriever for the memory store.
        
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
        Perform similarity search on memory.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters
            
        Returns:
            List of relevant memory documents
        """
        if k is None:
            k = settings.retrieval_top_k
        
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.similarity_search(query, **search_kwargs)
    
    def search_by_confidence(
        self,
        min_confidence: float,
        k: int = 10
    ) -> List[Document]:
        """
        Search memory by minimum confidence threshold.
        
        Args:
            min_confidence: Minimum confidence score
            k: Number of results
            
        Returns:
            List of high-confidence memories
        """
        # Note: Chroma doesn't support range queries in filters
        # This is a workaround - retrieve more and filter in Python
        all_docs = self.vectorstore.similarity_search("", k=k * 2)
        filtered = [
            doc for doc in all_docs
            if float(doc.metadata.get("confidence", 0)) >= min_confidence
        ]
        return filtered[:k]
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection("memory_collection")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.client.get_collection("memory_collection")
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
