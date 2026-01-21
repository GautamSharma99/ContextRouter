#!/usr/bin/env python
"""Test script to verify all imports work correctly."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    print("1. Testing langchain_text_splitters...")
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    print("   ✓ langchain_text_splitters works")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("2. Testing langchain_core.documents...")
    from langchain_core.documents import Document
    print("   ✓ langchain_core.documents works")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("3. Testing langchain.retrievers.contextual_compression...")
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    print("   ✓ langchain.retrievers.contextual_compression works")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("4. Testing langchain.retrievers.document_compressors...")
    from langchain.retrievers.document_compressors import LLMChainExtractor
    print("   ✓ langchain.retrievers.document_compressors works")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("5. Testing langchain_core.prompts...")
    from langchain_core.prompts import ChatPromptTemplate
    print("   ✓ langchain_core.prompts works")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("6. Testing langchain_core.output_parsers...")
    from langchain_core.output_parsers import PydanticOutputParser
    print("   ✓ langchain_core.output_parsers works")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✅ All import tests completed!")
