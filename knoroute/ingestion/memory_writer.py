"""Memory writer for learned insights."""

from typing import Optional, List
from langchain_core.documents import Document

from knoroute.vectorstores import MemoryVectorStore


class MemoryWriter:
    """
    Writes learned insights to the memory vector store.
    Called after successful query completion to store knowledge.
    """
    
    def __init__(self, vector_store: Optional[MemoryVectorStore] = None):
        """
        Initialize the memory writer.
        
        Args:
            vector_store: MemoryVectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or MemoryVectorStore()
    
    def write_insight(
        self,
        insight: str,
        learned_from: str,
        confidence: float = 0.8,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Write a learned insight to memory.
        
        Args:
            insight: The learned knowledge/insight
            learned_from: Original query that generated this insight
            confidence: Confidence score (0.0-1.0)
            tags: Optional categorization tags
            
        Returns:
            Document ID if added, None if duplicate
        """
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Add to memory store (handles duplicate checking internally)
        doc_id = self.vector_store.add_insight(
            insight=insight,
            learned_from=learned_from,
            confidence=confidence,
            tags=tags
        )
        
        if doc_id:
            print(f"✓ Stored insight in memory (confidence: {confidence:.2f})")
        else:
            print("ℹ Insight already exists in memory, skipping")
        
        return doc_id
    
    def write_multiple_insights(
        self,
        insights: List[dict]
    ) -> List[Optional[str]]:
        """
        Write multiple insights at once.
        
        Args:
            insights: List of insight dictionaries with keys:
                - insight: str
                - learned_from: str
                - confidence: float (optional, default 0.8)
                - tags: List[str] (optional)
        
        Returns:
            List of document IDs (None for duplicates)
        """
        doc_ids = []
        
        for insight_data in insights:
            doc_id = self.write_insight(
                insight=insight_data['insight'],
                learned_from=insight_data['learned_from'],
                confidence=insight_data.get('confidence', 0.8),
                tags=insight_data.get('tags')
            )
            doc_ids.append(doc_id)
        
        successful = sum(1 for id in doc_ids if id is not None)
        print(f"✓ Stored {successful}/{len(insights)} insights in memory")
        
        return doc_ids
    
    def extract_and_write_from_answer(
        self,
        query: str,
        answer: str,
        confidence: float = 0.8
    ) -> Optional[str]:
        """
        Extract key insight from an answer and write to memory.
        
        This is a simple extraction - in production, you might use an LLM
        to extract the most important insight.
        
        Args:
            query: Original query
            answer: Generated answer
            confidence: Confidence score
            
        Returns:
            Document ID if added
        """
        # Simple heuristic: use first sentence as insight
        # In production, use an LLM to extract key insights
        sentences = answer.split('.')
        if sentences:
            insight = sentences[0].strip() + '.'
            
            return self.write_insight(
                insight=insight,
                learned_from=query,
                confidence=confidence
            )
        
        return None


# Example usage
if __name__ == "__main__":
    # Initialize writer
    writer = MemoryWriter()
    
    # Example: Write a single insight
    # writer.write_insight(
    #     insight="Authentication uses JWT tokens with 1-hour expiration",
    #     learned_from="How does authentication work?",
    #     confidence=0.9,
    #     tags=["authentication", "security"]
    # )
    
    # Example: Write multiple insights
    # insights = [
    #     {
    #         "insight": "Rate limiting is implemented using Redis",
    #         "learned_from": "How is rate limiting implemented?",
    #         "confidence": 0.85,
    #         "tags": ["rate-limiting", "redis"]
    #     },
    #     {
    #         "insight": "Database migrations use Alembic",
    #         "learned_from": "How are database migrations handled?",
    #         "confidence": 0.9,
    #         "tags": ["database", "migrations"]
    #     }
    # ]
    # writer.write_multiple_insights(insights)
    
    print("Memory writer ready.")
    print("Use writer.write_insight() to store learned knowledge.")
