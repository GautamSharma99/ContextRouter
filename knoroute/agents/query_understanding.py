"""Query understanding agent."""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from knoroute.config import settings


class QueryUnderstanding(BaseModel):
    """Structured output for query understanding."""
    
    intent: Literal["debugging", "explanation", "comparison", "how_to"] = Field(
        description="The primary intent of the user's query"
    )
    topic: str = Field(
        description="The main topic or component being asked about (e.g., 'authentication', 'middleware', 'database')"
    )
    needs_memory: bool = Field(
        description="Whether this query would benefit from learned/historical knowledge"
    )
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Estimated complexity of the query"
    )


class QueryUnderstandingAgent:
    """
    Agent that analyzes user queries to understand intent and extract metadata.
    Uses few-shot prompting for accurate intent classification.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the query understanding agent.
        
        Args:
            llm: Language model instance (creates new if None)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=QueryUnderstanding)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding engineering queries.
Analyze the user's query and classify it according to these categories:

INTENT:
- debugging: User is trying to fix an error or understand why something failed
- explanation: User wants to understand how something works
- comparison: User wants to compare different approaches or implementations
- how_to: User wants step-by-step instructions

TOPIC: Extract the main technical component (e.g., authentication, API, database, middleware)

NEEDS_MEMORY: True if the query asks about:
- Historical issues ("why did X fail last week?")
- Past patterns ("what usually causes this?")
- Learned best practices

COMPLEXITY:
- simple: Single-component, straightforward question
- moderate: Multi-component or requires some context
- complex: Requires deep understanding across multiple systems

Examples:

Query: "How does authentication work in our system?"
Intent: explanation
Topic: authentication
Needs Memory: false
Complexity: moderate

Query: "Why did login fail last Tuesday?"
Intent: debugging
Topic: login
Needs Memory: true
Complexity: moderate

Query: "Compare REST vs GraphQL implementations"
Intent: comparison
Topic: API
Needs Memory: false
Complexity: moderate

Query: "How to add a new middleware?"
Intent: how_to
Topic: middleware
Needs Memory: false
Complexity: simple

{format_instructions}"""),
            ("user", "{query}")
        ])
    
    def understand(self, query: str) -> QueryUnderstanding:
        """
        Analyze a query and extract understanding.
        
        Args:
            query: User's query string
            
        Returns:
            QueryUnderstanding object with structured metadata
        """
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Execute
        result = chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result


# Example usage
if __name__ == "__main__":
    agent = QueryUnderstandingAgent()
    
    # Test queries
    test_queries = [
        "How does authentication work?",
        "Why did the API fail yesterday?",
        "Compare middleware implementations",
        "How to debug rate limiting issues?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        understanding = agent.understand(query)
        print(f"Intent: {understanding.intent}")
        print(f"Topic: {understanding.topic}")
        print(f"Needs Memory: {understanding.needs_memory}")
        print(f"Complexity: {understanding.complexity}")
