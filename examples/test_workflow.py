"""
Test workflow script to demonstrate the Agentic RAG system.
Run this after ingesting sample data.
"""

from knoroute.graph import AgenticRAGWorkflow


def print_separator(title=""):
    """Print a formatted separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
    else:
        print('='*60)


def print_answer(query: str, answer):
    """Print formatted answer."""
    print(f"\n📝 Query: {query}")
    print(f"\n💬 Answer:\n{answer.answer}")
    print(f"\n📊 Confidence: {answer.confidence:.2f}")
    
    print(f"\n📚 Citations ({len(answer.citations)}):")
    for i, citation in enumerate(answer.citations, 1):
        print(f"  {i}. [{citation.source_db}] {citation.relevance}")
    
    if answer.learned_insight:
        print(f"\n💡 Learned Insight:\n{answer.learned_insight}")


def test_workflow():
    """Test the workflow with various queries."""
    
    print_separator("Initializing Agentic RAG Workflow")
    
    # Initialize workflow
    workflow = AgenticRAGWorkflow()
    
    print("\n✓ Workflow initialized successfully")
    
    # Test queries
    test_queries = [
        {
            "query": "How does authentication work in our system?",
            "description": "Explanation query - should route to docs + code"
        },
        {
            "query": "Why did authentication fail with expired tokens?",
            "description": "Debugging query - should route to tickets + memory"
        },
        {
            "query": "What are the best practices for rate limiting?",
            "description": "How-to query - should route to docs + memory"
        },
        {
            "query": "How is CORS configured?",
            "description": "Simple explanation - should route to docs"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print_separator(f"Test {i}: {test_case['description']}")
        
        query = test_case["query"]
        
        try:
            # Execute query
            answer = workflow.query(query, max_retries=3)
            
            # Print results
            print_answer(query, answer)
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
    
    print_separator("Testing Complete")
    
    # Print vector store statistics
    print("\n📊 Vector Store Statistics:")
    print(f"  Docs DB: {workflow.docs_store.get_collection_stats()['count']} documents")
    print(f"  Code DB: {workflow.code_store.get_collection_stats()['count']} documents")
    print(f"  Tickets DB: {workflow.tickets_store.get_collection_stats()['count']} documents")
    print(f"  Memory DB: {workflow.memory_store.get_collection_stats()['count']} documents")


def test_individual_components():
    """Test individual components separately."""
    
    print_separator("Testing Individual Components")
    
    from knoroute.agents import (
        QueryUnderstandingAgent,
        RoutingAgent
    )
    
    # Test query understanding
    print("\n1. Testing Query Understanding Agent")
    understanding_agent = QueryUnderstandingAgent()
    
    test_query = "Why did login fail yesterday?"
    understanding = understanding_agent.understand(test_query)
    
    print(f"   Query: {test_query}")
    print(f"   Intent: {understanding.intent}")
    print(f"   Topic: {understanding.topic}")
    print(f"   Needs Memory: {understanding.needs_memory}")
    print(f"   Complexity: {understanding.complexity}")
    
    # Test routing
    print("\n2. Testing Routing Agent")
    routing_agent = RoutingAgent()
    
    routing = routing_agent.route(test_query, understanding)
    
    print(f"   Selected DBs: {routing.selected_dbs}")
    print(f"   Strategy: {routing.strategy}")
    print(f"   Retry Allowed: {routing.retry_allowed}")
    print(f"   Reasoning: {routing.reasoning}")


def interactive_mode():
    """Interactive query mode."""
    
    print_separator("Interactive Mode")
    print("\nType your queries (or 'quit' to exit)")
    
    workflow = AgenticRAGWorkflow()
    
    while True:
        query = input("\n🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        try:
            answer = workflow.query(query)
            print_answer(query, answer)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════╗
║     Agentic Knowledge Routing System - Test Suite       ║
╚══════════════════════════════════════════════════════════╝
""")
    
    print("Select test mode:")
    print("  1. Run automated tests")
    print("  2. Test individual components")
    print("  3. Interactive mode")
    print("  4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        test_workflow()
    elif choice == "2":
        test_individual_components()
    elif choice == "3":
        interactive_mode()
    elif choice == "4":
        print("\nGoodbye!")
        sys.exit(0)
    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)
