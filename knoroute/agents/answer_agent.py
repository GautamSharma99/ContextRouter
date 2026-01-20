"""Answer generation agent with strict grounding and citations."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document

from knoroute.config import settings


class Citation(BaseModel):
    """Citation for a piece of evidence."""
    
    source_db: str = Field(description="Which database this came from")
    chunk_id: str = Field(description="Identifier for the chunk")
    relevance: str = Field(description="Why this source is relevant")


class GroundedAnswer(BaseModel):
    """Structured output for grounded answers."""
    
    answer: str = Field(
        description="The answer to the user's query, grounded in retrieved evidence"
    )
    citations: List[Citation] = Field(
        description="Citations for sources used in the answer"
    )
    confidence: float = Field(
        description="Confidence in the answer (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    learned_insight: Optional[str] = Field(
        description="Key insight to store in memory (if any)",
        default=None
    )


class AnswerAgent:
    """
    Agent that generates grounded answers with citations.
    Strictly uses only retrieved evidence - no hallucination.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the answer agent.
        
        Args:
            llm: Language model instance (creates new if None)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=GroundedAnswer)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical writer for an engineering knowledge system.

CRITICAL RULES:
1. **Only use information from the provided evidence** - do not add external knowledge
2. **Cite your sources** - reference which database each piece of information came from
3. **Be accurate** - if evidence is unclear, say so
4. **Be concise** - provide clear, actionable answers
5. **Extract insights** - identify key learnings worth storing in memory

ANSWER STRUCTURE:
- Start with a direct answer to the question
- Provide supporting details from evidence
- Include code examples if available
- Mention caveats or limitations if relevant

CITATIONS:
- Reference the source database for each claim
- Explain why each source is relevant
- Use format: [docs], [code], [tickets], [memory]

LEARNED INSIGHTS:
- Extract ONE key insight worth remembering
- Should be a general principle or pattern
- Only if the answer reveals something valuable

{format_instructions}"""),
            ("user", """Query: {query}

Retrieved Evidence:
{evidence}

Generate a grounded answer using ONLY the evidence provided.""")
        ])
    
    def generate(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> GroundedAnswer:
        """
        Generate a grounded answer from retrieved evidence.
        
        Args:
            query: User's original query
            retrieved_docs: List of retrieved documents with metadata
            
        Returns:
            GroundedAnswer with citations and optional insight
        """
        # Format evidence with IDs
        evidence_text = self._format_evidence_with_ids(retrieved_docs)
        
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Execute
        result = chain.invoke({
            "query": query,
            "evidence": evidence_text,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def _format_evidence_with_ids(self, documents: List[Document]) -> str:
        """
        Format retrieved documents with IDs for citation.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted evidence string with IDs
        """
        if not documents:
            return "No evidence available."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            source_db = doc.metadata.get("source_db", "unknown")
            
            # Include relevant metadata
            metadata_str = ""
            if source_db == "code":
                file_path = doc.metadata.get("file_path", "")
                function = doc.metadata.get("function_name", "")
                if file_path or function:
                    metadata_str = f"\nFile: {file_path}, Function: {function}"
            elif source_db == "tickets":
                ticket_id = doc.metadata.get("ticket_id", "")
                severity = doc.metadata.get("severity", "")
                if ticket_id:
                    metadata_str = f"\nTicket: {ticket_id}, Severity: {severity}"
            elif source_db == "docs":
                doc_type = doc.metadata.get("doc_type", "")
                section = doc.metadata.get("section", "")
                if doc_type:
                    metadata_str = f"\nType: {doc_type}, Section: {section}"
            
            formatted_parts.append(
                f"[{i}] Source: {source_db}{metadata_str}\n{doc.page_content}\n"
            )
        
        return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    from langchain.schema import Document
    
    agent = AnswerAgent()
    
    # Test query
    query = "How does authentication work?"
    docs = [
        Document(
            page_content="Authentication in our system uses JWT (JSON Web Tokens) with a 1-hour expiration time. Tokens are signed using the HS256 algorithm with a secret key stored in environment variables.",
            metadata={"source_db": "docs", "doc_type": "API", "section": "authentication"}
        ),
        Document(
            page_content="""def authenticate(username, password):
    # Verify credentials
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        raise AuthenticationError("Invalid credentials")
    
    # Generate JWT token
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, SECRET_KEY, algorithm='HS256')
    
    return token""",
            metadata={"source_db": "code", "file_path": "auth/service.py", "function_name": "authenticate"}
        ),
        Document(
            page_content="Best practice: Always use HTTPS when transmitting JWT tokens. Store tokens securely in httpOnly cookies to prevent XSS attacks.",
            metadata={"source_db": "memory", "confidence": 0.9}
        )
    ]
    
    result = agent.generate(query, docs)
    
    print(f"Query: {query}\n")
    print(f"Answer:\n{result.answer}\n")
    print(f"Confidence: {result.confidence}\n")
    print("Citations:")
    for citation in result.citations:
        print(f"  - [{citation.source_db}] {citation.relevance}")
    
    if result.learned_insight:
        print(f"\nLearned Insight: {result.learned_insight}")
