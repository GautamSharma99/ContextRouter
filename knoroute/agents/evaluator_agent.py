"""Evidence evaluator agent."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document

from knoroute.config import settings


class EvaluationResult(BaseModel):
    """Structured output for evidence evaluation."""
    
    is_sufficient: bool = Field(
        description="Whether the retrieved evidence is sufficient to answer the query confidently"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    missing_aspects: List[str] = Field(
        description="What aspects are missing if evidence is insufficient",
        default_factory=list
    )
    suggested_dbs: List[str] = Field(
        description="Additional databases to query if insufficient",
        default_factory=list
    )


class EvaluatorAgent:
    """
    Agent that evaluates whether retrieved evidence is sufficient to answer the query.
    Uses chain-of-thought reasoning to assess completeness.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the evaluator agent.
        
        Args:
            llm: Language model instance (creates new if None)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an evidence evaluator for an engineering knowledge system.

Your job is to assess whether the retrieved evidence is sufficient to answer the user's query confidently.

EVALUATION CRITERIA:

1. **Completeness**: Does the evidence cover all aspects of the question?
2. **Relevance**: Is the evidence directly related to the query?
3. **Clarity**: Is the evidence clear enough to formulate a good answer?
4. **Consistency**: Do different pieces of evidence align or contradict?

CONFIDENCE SCORING:
- 0.9-1.0: Excellent evidence, can answer with high confidence
- 0.7-0.9: Good evidence, minor gaps but answerable
- 0.5-0.7: Moderate evidence, some important aspects missing
- 0.0-0.5: Insufficient evidence, cannot answer reliably

IF INSUFFICIENT:
- List specific missing aspects
- Suggest which databases might have the missing information:
  * docs - for official behavior, API specs
  * code - for implementation details
  * tickets - for historical failures, bugs
  * memory - for learned patterns, insights

Think step-by-step about what the query needs and what the evidence provides.

{format_instructions}"""),
            ("user", """Query: {query}

Retrieved Evidence:
{evidence}

Sources queried: {sources}

Evaluate if this evidence is sufficient to answer the query.""")
        ])
    
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Document],
        sources_queried: List[str]
    ) -> EvaluationResult:
        """
        Evaluate whether retrieved evidence is sufficient.
        
        Args:
            query: User's original query
            retrieved_docs: List of retrieved documents
            sources_queried: Which databases were queried
            
        Returns:
            EvaluationResult with sufficiency assessment
        """
        # Format evidence for evaluation
        evidence_text = self._format_evidence(retrieved_docs)
        
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Execute
        result = chain.invoke({
            "query": query,
            "evidence": evidence_text,
            "sources": ", ".join(sources_queried),
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def _format_evidence(self, documents: List[Document]) -> str:
        """
        Format retrieved documents for evaluation.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted evidence string
        """
        if not documents:
            return "No evidence retrieved."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            source_db = doc.metadata.get("source_db", "unknown")
            content = doc.page_content[:300]  # Truncate for context window
            
            formatted_parts.append(
                f"[{i}] Source: {source_db}\n{content}...\n"
            )
        
        return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    from langchain.schema import Document
    
    agent = EvaluatorAgent()
    
    # Test with sufficient evidence
    query = "How does authentication work?"
    docs = [
        Document(
            page_content="Authentication uses JWT tokens with 1-hour expiration. Tokens are signed with HS256.",
            metadata={"source_db": "docs"}
        ),
        Document(
            page_content="def authenticate(user, password):\n    token = jwt.encode({'user': user}, SECRET_KEY)\n    return token",
            metadata={"source_db": "code"}
        )
    ]
    
    result = agent.evaluate(query, docs, ["docs", "code"])
    print(f"Query: {query}")
    print(f"Sufficient: {result.is_sufficient}")
    print(f"Confidence: {result.confidence}")
    print(f"Missing: {result.missing_aspects}")
    
    # Test with insufficient evidence
    print("\n" + "="*60 + "\n")
    
    query2 = "Why did authentication fail yesterday?"
    docs2 = [
        Document(
            page_content="Authentication uses JWT tokens.",
            metadata={"source_db": "docs"}
        )
    ]
    
    result2 = agent.evaluate(query2, docs2, ["docs"])
    print(f"Query: {query2}")
    print(f"Sufficient: {result2.is_sufficient}")
    print(f"Confidence: {result2.confidence}")
    print(f"Missing: {result2.missing_aspects}")
    print(f"Suggested DBs: {result2.suggested_dbs}")
