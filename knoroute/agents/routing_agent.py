"""Smart routing agent with LLM-driven decision making."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from knoroute.config import settings
from knoroute.agents.query_understanding import QueryUnderstanding


class RoutingDecision(BaseModel):
    """Structured output for routing decisions."""
    
    selected_dbs: List[Literal["docs", "code", "tickets", "memory"]] = Field(
        description="Which databases to query for this request"
    )
    strategy: Literal["parallel", "sequential"] = Field(
        description="Whether to query databases in parallel or sequentially"
    )
    retry_allowed: bool = Field(
        description="Whether retry is allowed if evidence is insufficient"
    )
    reasoning: str = Field(
        description="Explanation of why these databases were selected"
    )


class RoutingAgent:
    """
    LLM-driven routing agent that decides which vector databases to query.
    No hardcoded rules - uses reasoning to make intelligent decisions.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the routing agent.
        
        Args:
            llm: Language model instance (creates new if None)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=RoutingDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent routing agent for an engineering knowledge system.

You have access to 4 vector databases:
1. **docs** - Official documentation (intended behavior, API specs, guides)
2. **code** - Actual implementation (source code, functions, classes)
3. **tickets** - Historical failures (bugs, issues, incidents)
4. **memory** - Learned knowledge (insights from past queries)

Your job is to decide which database(s) to query based on the query understanding.

ROUTING PRINCIPLES:

For EXPLANATION queries:
- Usually need: docs + code
- Add memory if it's a common question
- Tickets rarely needed unless asking about known issues

For DEBUGGING queries:
- Usually need: tickets + code
- Add memory if similar issues occurred before
- Docs helpful for expected behavior

For COMPARISON queries:
- Usually need: docs + code
- Memory useful for learned best practices
- Tickets rarely needed

For HOW_TO queries:
- Usually need: docs
- Add code for implementation examples
- Memory useful for learned patterns

STRATEGY:
- Use "parallel" when databases are independent
- Use "sequential" when one result informs the next query

RETRY:
- Allow retry for complex queries
- Disallow for simple lookups

Think step-by-step and explain your reasoning.

{format_instructions}"""),
            ("user", """Query: {query}

Understanding:
- Intent: {intent}
- Topic: {topic}
- Needs Memory: {needs_memory}
- Complexity: {complexity}

Previous attempt: {previous_attempt}
Missing aspects: {missing_aspects}

Decide which databases to query and explain why.""")
        ])
    
    def route(
        self,
        query: str,
        understanding: QueryUnderstanding,
        previous_attempt: Optional[List[str]] = None,
        missing_aspects: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Make a routing decision based on query understanding.
        
        Args:
            query: User's query
            understanding: QueryUnderstanding from previous agent
            previous_attempt: Previously queried databases (for retry)
            missing_aspects: What was missing in previous attempt
            
        Returns:
            RoutingDecision with selected databases and strategy
        """
        # Prepare context
        prev_attempt_str = "None (first attempt)"
        if previous_attempt:
            prev_attempt_str = f"Previously queried: {', '.join(previous_attempt)}"
        
        missing_str = "N/A"
        if missing_aspects:
            missing_str = ", ".join(missing_aspects)
        
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Execute
        result = chain.invoke({
            "query": query,
            "intent": understanding.intent,
            "topic": understanding.topic,
            "needs_memory": understanding.needs_memory,
            "complexity": understanding.complexity,
            "previous_attempt": prev_attempt_str,
            "missing_aspects": missing_str,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result


# Example usage
if __name__ == "__main__":
    from knoroute.agents.query_understanding import QueryUnderstandingAgent
    
    # Initialize agents
    understanding_agent = QueryUnderstandingAgent()
    routing_agent = RoutingAgent()
    
    # Test queries
    test_queries = [
        "How does authentication work?",
        "Why did login fail yesterday?",
        "Compare REST vs GraphQL implementations",
        "How to add rate limiting?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Understand query
        understanding = understanding_agent.understand(query)
        print(f"Intent: {understanding.intent} | Topic: {understanding.topic}")
        
        # Route query
        routing = routing_agent.route(query, understanding)
        print(f"\nSelected DBs: {routing.selected_dbs}")
        print(f"Strategy: {routing.strategy}")
        print(f"Retry Allowed: {routing.retry_allowed}")
        print(f"\nReasoning: {routing.reasoning}")
