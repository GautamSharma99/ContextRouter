"""LangGraph workflow orchestration for the agentic RAG system."""

from typing import TypedDict, Optional, List, Dict, Literal
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from knoroute.agents import (
    QueryUnderstandingAgent,
    QueryUnderstanding,
    RoutingAgent,
    RoutingDecision,
    EvaluatorAgent,
    EvaluationResult,
    AnswerAgent,
    GroundedAnswer,
    RetrieverTools
)
from knoroute.vectorstores import (
    DocsVectorStore,
    CodeVectorStore,
    TicketsVectorStore,
    MemoryVectorStore
)
from knoroute.ingestion import MemoryWriter
from knoroute.config import settings


class GraphState(TypedDict):
    """State schema for the LangGraph workflow."""
    
    # Input
    query: str
    
    # Agent outputs
    understanding: Optional[QueryUnderstanding]
    routing: Optional[RoutingDecision]
    retrieved_docs: Dict[str, List[Document]]
    merged_context: Optional[List[Document]]
    evaluation: Optional[EvaluationResult]
    answer: Optional[GroundedAnswer]
    
    # Control flow
    retry_count: int
    max_retries: int
    error: Optional[str]


class AgenticRAGWorkflow:
    """
    LangGraph workflow for agentic RAG with intelligent routing.
    
    Flow:
    1. Understand query
    2. Route to appropriate databases
    3. Retrieve from selected databases
    4. Merge evidence
    5. Evaluate sufficiency
    6. If insufficient and retries available → back to routing
    7. Generate answer
    8. Write to memory (if insight exists)
    """
    
    def __init__(
        self,
        docs_store: Optional[DocsVectorStore] = None,
        code_store: Optional[CodeVectorStore] = None,
        tickets_store: Optional[TicketsVectorStore] = None,
        memory_store: Optional[MemoryVectorStore] = None,
    ):
        """
        Initialize the workflow.
        
        Args:
            docs_store: Documentation vector store
            code_store: Code vector store
            tickets_store: Tickets vector store
            memory_store: Memory vector store
        """
        # Initialize vector stores
        self.docs_store = docs_store or DocsVectorStore()
        self.code_store = code_store or CodeVectorStore()
        self.tickets_store = tickets_store or TicketsVectorStore()
        self.memory_store = memory_store or MemoryVectorStore()
        
        # Initialize agents
        self.understanding_agent = QueryUnderstandingAgent()
        self.routing_agent = RoutingAgent()
        self.evaluator_agent = EvaluatorAgent()
        self.answer_agent = AnswerAgent()
        
        # Initialize retriever tools
        self.retriever_tools = RetrieverTools(
            self.docs_store,
            self.code_store,
            self.tickets_store,
            self.memory_store
        )
        
        # Initialize memory writer
        self.memory_writer = MemoryWriter(self.memory_store)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("retrieve_docs", self._retrieve_docs)
        workflow.add_node("merge_evidence", self._merge_evidence)
        workflow.add_node("evaluate_sufficiency", self._evaluate_sufficiency)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("write_memory", self._write_memory)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add edges
        workflow.add_edge("understand_query", "route_query")
        workflow.add_edge("route_query", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "merge_evidence")
        workflow.add_edge("merge_evidence", "evaluate_sufficiency")
        
        # Conditional edge: evaluate → retry or answer
        workflow.add_conditional_edges(
            "evaluate_sufficiency",
            self._should_retry,
            {
                "retry": "route_query",
                "answer": "generate_answer"
            }
        )
        
        workflow.add_edge("generate_answer", "write_memory")
        workflow.add_edge("write_memory", END)
        
        return workflow.compile()
    
    # Node functions
    
    def _understand_query(self, state: GraphState) -> GraphState:
        """Understand the user's query."""
        understanding = self.understanding_agent.understand(state["query"])
        state["understanding"] = understanding
        return state
    
    def _route_query(self, state: GraphState) -> GraphState:
        """Route query to appropriate databases."""
        # Get previous attempt info if retrying
        previous_dbs = None
        missing_aspects = None
        
        if state["retry_count"] > 0 and state.get("evaluation"):
            previous_dbs = list(state["retrieved_docs"].keys())
            missing_aspects = state["evaluation"].missing_aspects
        
        # Make routing decision
        routing = self.routing_agent.route(
            query=state["query"],
            understanding=state["understanding"],
            previous_attempt=previous_dbs,
            missing_aspects=missing_aspects
        )
        
        state["routing"] = routing
        return state
    
    def _retrieve_docs(self, state: GraphState) -> GraphState:
        """Retrieve documents from selected databases."""
        selected_dbs = state["routing"].selected_dbs
        
        # Retrieve from selected databases
        retrieved = self.retriever_tools.retrieve_from_multiple(
            query=state["query"],
            databases=selected_dbs,
            k=settings.retrieval_top_k
        )
        
        # Merge with previous retrieval if retrying
        if state["retry_count"] > 0 and state.get("retrieved_docs"):
            # Combine with previous results
            for db, docs in retrieved.items():
                if db in state["retrieved_docs"]:
                    state["retrieved_docs"][db].extend(docs)
                else:
                    state["retrieved_docs"][db] = docs
        else:
            state["retrieved_docs"] = retrieved
        
        return state
    
    def _merge_evidence(self, state: GraphState) -> GraphState:
        """Merge evidence from all retrieved documents."""
        all_docs = []
        
        for db, docs in state["retrieved_docs"].items():
            all_docs.extend(docs)
        
        # Simple deduplication based on content
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        state["merged_context"] = unique_docs
        return state
    
    def _evaluate_sufficiency(self, state: GraphState) -> GraphState:
        """Evaluate if evidence is sufficient."""
        evaluation = self.evaluator_agent.evaluate(
            query=state["query"],
            retrieved_docs=state["merged_context"],
            sources_queried=list(state["retrieved_docs"].keys())
        )
        
        state["evaluation"] = evaluation
        return state
    
    def _should_retry(self, state: GraphState) -> Literal["retry", "answer"]:
        """Decide whether to retry or generate answer."""
        evaluation = state["evaluation"]
        
        # Check if sufficient
        if evaluation.is_sufficient:
            return "answer"
        
        # Check if retries available
        if state["retry_count"] >= state["max_retries"]:
            return "answer"  # Give best answer we can
        
        # Check if retry is allowed by routing
        if not state["routing"].retry_allowed:
            return "answer"
        
        # Increment retry count and retry
        state["retry_count"] += 1
        return "retry"
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate grounded answer."""
        answer = self.answer_agent.generate(
            query=state["query"],
            retrieved_docs=state["merged_context"]
        )
        
        state["answer"] = answer
        return state
    
    def _write_memory(self, state: GraphState) -> GraphState:
        """Write learned insight to memory."""
        answer = state["answer"]
        
        if answer.learned_insight:
            self.memory_writer.write_insight(
                insight=answer.learned_insight,
                learned_from=state["query"],
                confidence=answer.confidence
            )
        
        return state
    
    # Public interface
    
    def query(self, query: str, max_retries: int = None) -> GroundedAnswer:
        """
        Execute the workflow for a query.
        
        Args:
            query: User's query
            max_retries: Maximum retry attempts (default from settings)
            
        Returns:
            GroundedAnswer with citations
        """
        if max_retries is None:
            max_retries = settings.max_retry_attempts
        
        # Initialize state
        initial_state: GraphState = {
            "query": query,
            "understanding": None,
            "routing": None,
            "retrieved_docs": {},
            "merged_context": None,
            "evaluation": None,
            "answer": None,
            "retry_count": 0,
            "max_retries": max_retries,
            "error": None
        }
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state["answer"]
    
    def get_graph_visualization(self) -> str:
        """Get ASCII visualization of the graph."""
        return self.graph.get_graph().draw_ascii()


# Example usage
if __name__ == "__main__":
    # Initialize workflow
    workflow = AgenticRAGWorkflow()
    
    # Print graph structure
    print("Graph Structure:")
    print(workflow.get_graph_visualization())
    
    # Test query (requires populated vector stores)
    # answer = workflow.query("How does authentication work?")
    # print(f"\nAnswer: {answer.answer}")
    # print(f"Confidence: {answer.confidence}")
    # print(f"Citations: {len(answer.citations)}")
