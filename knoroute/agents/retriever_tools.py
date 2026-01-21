"""Retriever tools for LangGraph integration."""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.tools import Tool

from knoroute.vectorstores import (
    DocsVectorStore,
    CodeVectorStore,
    TicketsVectorStore,
    MemoryVectorStore
)


class RetrieverTools:
    """
    Collection of retriever tools for each vector database.
    Each tool wraps a vector store retriever for use in the LangGraph workflow.
    """
    
    def __init__(
        self,
        docs_store: DocsVectorStore,
        code_store: CodeVectorStore,
        tickets_store: TicketsVectorStore,
        memory_store: MemoryVectorStore
    ):
        """
        Initialize retriever tools.
        
        Args:
            docs_store: Documentation vector store
            code_store: Code vector store
            tickets_store: Tickets vector store
            memory_store: Memory vector store
        """
        self.docs_store = docs_store
        self.code_store = code_store
        self.tickets_store = tickets_store
        self.memory_store = memory_store
    
    def retrieve_from_docs(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve from documentation database.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with source_db metadata
        """
        docs = self.docs_store.similarity_search(query, k=k)
        
        # Add source_db metadata
        for doc in docs:
            doc.metadata["source_db"] = "docs"
        
        return docs
    
    def retrieve_from_code(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve from code database.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with source_db metadata
        """
        docs = self.code_store.similarity_search(query, k=k)
        
        # Add source_db metadata
        for doc in docs:
            doc.metadata["source_db"] = "code"
        
        return docs
    
    def retrieve_from_tickets(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve from tickets database.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with source_db metadata
        """
        docs = self.tickets_store.similarity_search(query, k=k)
        
        # Add source_db metadata
        for doc in docs:
            doc.metadata["source_db"] = "tickets"
        
        return docs
    
    def retrieve_from_memory(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve from memory database.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with source_db metadata
        """
        docs = self.memory_store.similarity_search(query, k=k)
        
        # Add source_db metadata
        for doc in docs:
            doc.metadata["source_db"] = "memory"
        
        return docs
    
    def retrieve_from_multiple(
        self,
        query: str,
        databases: List[str],
        k: int = 5
    ) -> Dict[str, List[Document]]:
        """
        Retrieve from multiple databases.
        
        Args:
            query: Search query
            databases: List of database names to query
            k: Number of results per database
            
        Returns:
            Dictionary mapping database name to documents
        """
        results = {}
        
        for db in databases:
            if db == "docs":
                results["docs"] = self.retrieve_from_docs(query, k)
            elif db == "code":
                results["code"] = self.retrieve_from_code(query, k)
            elif db == "tickets":
                results["tickets"] = self.retrieve_from_tickets(query, k)
            elif db == "memory":
                results["memory"] = self.retrieve_from_memory(query, k)
        
        return results
    
    def get_langchain_tools(self) -> List[Tool]:
        """
        Get LangChain Tool objects for agent use.
        
        Returns:
            List of Tool objects
        """
        return [
            Tool(
                name="retrieve_docs",
                func=self.retrieve_from_docs,
                description="Retrieve from documentation database. Use for API specs, guides, and intended behavior."
            ),
            Tool(
                name="retrieve_code",
                func=self.retrieve_from_code,
                description="Retrieve from code database. Use for implementation details and source code."
            ),
            Tool(
                name="retrieve_tickets",
                func=self.retrieve_from_tickets,
                description="Retrieve from tickets database. Use for historical failures and bug reports."
            ),
            Tool(
                name="retrieve_memory",
                func=self.retrieve_from_memory,
                description="Retrieve from memory database. Use for learned insights and patterns."
            ),
        ]


# Example usage
if __name__ == "__main__":
    # Initialize vector stores
    docs_store = DocsVectorStore()
    code_store = CodeVectorStore()
    tickets_store = TicketsVectorStore()
    memory_store = MemoryVectorStore()
    
    # Initialize retriever tools
    tools = RetrieverTools(docs_store, code_store, tickets_store, memory_store)
    
    # Test retrieval
    query = "How does authentication work?"
    results = tools.retrieve_from_multiple(query, ["docs", "code"], k=3)
    
    print(f"Query: {query}\n")
    for db, docs in results.items():
        print(f"{db.upper()}: {len(docs)} documents")
        for doc in docs:
            print(f"  - {doc.page_content[:100]}...")
