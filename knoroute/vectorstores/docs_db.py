"""Documentation vector store implementation."""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import chromadb
from pathlib import Path

from knoroute.config import settings


class DocsVectorStore:
    """
    Vector store for documentation with metadata schema:
    - source: file path
    - doc_type: API, guide, reference, tutorial
    - section: hierarchical location
    - version: documentation version
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the documentation vector store."""
        if persist_directory is None:
            persist_directory = str(Path(settings.vector_store_path) / "docs_db")
        
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
            collection_name="docs_collection",
            embedding_function=self.embeddings,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with metadata
            
        Returns:
            List of document IDs
        """
        # Validate metadata schema
        for doc in documents:
            if "source" not in doc.metadata:
                raise ValueError("Document metadata must include 'source'")
            
            # Set defaults for optional fields
            doc.metadata.setdefault("doc_type", "guide")
            doc.metadata.setdefault("section", "root")
            doc.metadata.setdefault("version", "latest")
        
        return self.vectorstore.add_documents(documents)
    
    def get_retriever(self, k: int = None, use_compression: bool = False):
        """
        Get a retriever for the documentation store.
        
        Args:
            k: Number of documents to retrieve (default from settings)
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
            # Use LLM to extract relevant parts
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
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters (e.g., {"doc_type": "API"})
            
        Returns:
            List of relevant documents
        """
        if k is None:
            k = settings.retrieval_top_k
        
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.similarity_search(query, **search_kwargs)
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection("docs_collection")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.client.get_collection("docs_collection")
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
