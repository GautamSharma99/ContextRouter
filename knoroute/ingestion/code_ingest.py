"""Code ingestion pipeline."""

import os
import ast
from pathlib import Path
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from knoroute.vectorstores import CodeVectorStore


class CodeIngestionPipeline:
    """
    Ingestion pipeline for source code.
    Supports Python, JavaScript, TypeScript, Java with function/class extraction.
    """
    
    # Language file extensions
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
    }
    
    def __init__(self, vector_store: Optional[CodeVectorStore] = None):
        """
        Initialize the code ingestion pipeline.
        
        Args:
            vector_store: CodeVectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or CodeVectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\nclass ", "\n\ndef ", "\n\nfunction ", "\n\n", "\n", " "],
        )
    
    def load_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.py"
    ) -> List[Document]:
        """
        Load all code files from a directory.
        
        Args:
            directory_path: Path to code directory
            glob_pattern: File pattern to match
            
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
            file_path = doc.metadata.get("source", "")
            doc.metadata["file_path"] = file_path
            doc.metadata["language"] = self._detect_language(file_path)
        
        return documents
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single code file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            List of documents
        """
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["file_path"] = file_path
            doc.metadata["language"] = self._detect_language(file_path)
        
        return documents
    
    def extract_functions_python(self, code: str, file_path: str) -> List[Document]:
        """
        Extract functions and classes from Python code.
        
        Args:
            code: Python source code
            file_path: Path to the file
            
        Returns:
            List of documents, one per function/class
        """
        documents = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Get the source code for this function/class
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    # Extract the code
                    lines = code.split('\n')
                    func_code = '\n'.join(lines[start_line-1:end_line])
                    
                    # Get docstring if available
                    docstring = ast.get_docstring(node) or ""
                    
                    # Create document
                    doc = Document(
                        page_content=f"{func_code}\n\n# Docstring:\n{docstring}",
                        metadata={
                            "file_path": file_path,
                            "language": "python",
                            "function_name": node.name,
                            "line_range": f"{start_line}-{end_line}",
                        }
                    )
                    documents.append(doc)
        
        except SyntaxError:
            # If parsing fails, fall back to chunking
            pass
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk code documents intelligently.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            language = doc.metadata.get("language", "")
            
            # For Python, try to extract functions
            if language == "python":
                extracted = self.extract_functions_python(
                    doc.page_content,
                    doc.metadata.get("file_path", "")
                )
                if extracted:
                    chunked_docs.extend(extracted)
                    continue
            
            # For other languages or if extraction fails, use text splitter
            chunks = self.text_splitter.split_documents([doc])
            
            # Add line range metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata["function_name"] = f"chunk_{i}"
                chunk.metadata["line_range"] = ""
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def ingest_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.py",
        extract_functions: bool = True
    ) -> List[str]:
        """
        Complete ingestion pipeline for a code directory.
        
        Args:
            directory_path: Path to code directory
            glob_pattern: File pattern to match
            extract_functions: Whether to extract individual functions
            
        Returns:
            List of document IDs
        """
        # Load documents
        documents = self.load_directory(directory_path, glob_pattern)
        
        # Chunk/extract functions
        if extract_functions:
            documents = self.chunk_documents(documents)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} code chunks from {directory_path}")
        return doc_ids
    
    def ingest_file(
        self,
        file_path: str,
        extract_functions: bool = True
    ) -> List[str]:
        """
        Complete ingestion pipeline for a single code file.
        
        Args:
            file_path: Path to the code file
            extract_functions: Whether to extract individual functions
            
        Returns:
            List of document IDs
        """
        # Load document
        documents = self.load_file(file_path)
        
        # Chunk/extract functions
        if extract_functions:
            documents = self.chunk_documents(documents)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} code chunks from {file_path}")
        return doc_ids
    
    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name
        """
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, "unknown")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CodeIngestionPipeline()
    
    # Example: Ingest a code directory
    # pipeline.ingest_directory(
    #     directory_path="./src",
    #     glob_pattern="**/*.py",
    #     extract_functions=True
    # )
    
    print("Code ingestion pipeline ready.")
    print("Use pipeline.ingest_directory() or pipeline.ingest_file() to add code.")
