"""Code implementation vector store."""

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


class CodeVectorStore:
    """
    Vector store for code implementation with metadata schema:
    - file_path: source file
    - language: programming language
    - function_name: function/class name
    - line_range: start-end lines
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the code vector store."""
        if persist_directory is None:
            persist_directory = str(Path(settings.vector_store_path) / "code_db")
        
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
            collection_name="code_collection",
            embedding_function=self.embeddings,
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add code documents to the vector store.
        
        Args:
            documents: List of Document objects with code metadata
            
        Returns:
            List of document IDs
        """
        # Validate metadata schema
        for doc in documents:
            if "file_path" not in doc.metadata:
                raise ValueError("Document metadata must include 'file_path'")
            
            # Set defaults for optional fields
            doc.metadata.setdefault("language", "unknown")
            doc.metadata.setdefault("function_name", "")
            doc.metadata.setdefault("line_range", "")
        
        return self.vectorstore.add_documents(documents)
    
    def get_retriever(self, k: int = None, use_compression: bool = False):
        """
        Get a retriever for the code store.
        
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
        Perform similarity search on code.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters (e.g., {"language": "python"})
            
        Returns:
            List of relevant code documents
        """
        if k is None:
            k = settings.retrieval_top_k
        
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.similarity_search(query, **search_kwargs)
    
    def search_by_function(self, function_name: str, k: int = 5) -> List[Document]:
        """
        Search for code by function name.
        
        Args:
            function_name: Name of the function/class
            k: Number of results
            
        Returns:
            List of matching code documents
        """
        return self.similarity_search(
            query=function_name,
            k=k,
            filter_dict={"function_name": function_name}
        )
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection("code_collection")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        collection = self.client.get_collection("code_collection")
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
