# ContextRouter: Agentic Knowledge Routing System

A production-grade multi-agent RAG (Retrieval-Augmented Generation) system using **LangGraph** and **LangChain** for intelligent query routing across multiple vector databases.

## 🎯 Problem Statement

Traditional RAG systems query all knowledge sources indiscriminately, leading to:
- **Inefficiency**: Unnecessary database queries
- **Noise**: Irrelevant context dilutes answer quality
- **Inflexibility**: Cannot adapt retrieval strategy to query type

## 💡 Why Multi-Vector DB Routing?

Engineering teams have different types of knowledge:
- **Documentation** (intended behavior)
- **Code** (actual implementation)
- **Tickets** (historical failures)
- **Memory** (learned insights)

Different queries need different sources:
- "How does auth work?" → Docs + Code
- "Why did login fail yesterday?" → Tickets + Memory
- "Compare middleware approaches" → Docs + Code + Memory

**ContextRouter** uses LLM-driven reasoning to intelligently route queries to the right databases, retrieve evidence, evaluate sufficiency, and generate grounded answers.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Query Understanding   │ ◄── Intent, Topic, Complexity
         │      Agent            │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Routing Agent       │ ◄── LLM-driven, no hardcoded rules
         │  (LLM-driven)         │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │  Conditional Routing  │
         └───────────┬───────────┘
                     │
         ┌───────────┴────────────────────┐
         │                                │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │  Docs   │  │  Code   │  │ Tickets │  │ Memory  │
    │   DB    │  │   DB    │  │   DB    │  │   DB    │
    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │            │
         └────────────┴────────────┴────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Evidence Merger       │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Evaluator Agent       │ ◄── Sufficiency check
         └────────────┬───────────┘
                      │
              ┌───────┴────────┐
              │                │
         Insufficient      Sufficient
              │                │
         ┌────▼────┐           │
         │  Retry  │           │
         │  Loop   │───────────┘
         └─────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Answer Agent         │ ◄── Grounded + Citations
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Memory Writer        │ ◄── Store insights
         └────────────────────────┘
```

## 🧩 Components

### Vector Databases (4)

Each with independent schema and retriever:

| Database | Purpose | Metadata Schema |
|----------|---------|-----------------|
| **docs_db** | Official documentation | `doc_type`, `section`, `version` |
| **code_db** | Source code | `file_path`, `language`, `function_name`, `line_range` |
| **tickets_db** | Historical failures | `ticket_id`, `status`, `severity`, `created_at` |
| **memory_db** | Learned insights | `learned_from`, `confidence`, `tags` |

### Agents (6)

1. **Query Understanding Agent**
   - Classifies intent: debugging, explanation, comparison, how_to
   - Extracts topic and complexity
   - Determines if memory is needed

2. **Smart Routing Agent**
   - LLM-driven routing (no hardcoded rules)
   - Selects databases based on reasoning
   - Chooses parallel vs sequential strategy

3. **Retriever Agents** (4)
   - One per vector database
   - Returns top-k documents with metadata

4. **Evidence Merger Agent**
   - Deduplicates chunks
   - Preserves source metadata
   - Groups by relevance

5. **Evaluator Agent**
   - Assesses evidence sufficiency
   - Identifies missing aspects
   - Suggests additional databases

6. **Answer Generator Agent**
   - Generates grounded answers
   - Cites sources
   - Extracts learned insights

### LangGraph Orchestration

- **State Management**: TypedDict with all agent outputs
- **Conditional Routing**: Dynamic database selection
- **Retry Loop**: Up to 3 attempts if evidence insufficient
- **Memory Writing**: Automatic insight storage

## 📂 Project Structure

```
ContextRouter/
├── knoroute/
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration
│   │
│   ├── vectorstores/             # Vector database implementations
│   │   ├── __init__.py
│   │   ├── docs_db.py           # Documentation store
│   │   ├── code_db.py           # Code store
│   │   ├── tickets_db.py        # Tickets store
│   │   └── memory_db.py         # Memory store
│   │
│   ├── ingestion/                # Data ingestion pipelines
│   │   ├── __init__.py
│   │   ├── docs_ingest.py       # Markdown/text ingestion
│   │   ├── code_ingest.py       # Code ingestion with AST parsing
│   │   ├── tickets_ingest.py    # JSON/CSV ticket ingestion
│   │   └── memory_writer.py     # Insight writer
│   │
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py
│   │   ├── query_understanding.py
│   │   ├── routing_agent.py
│   │   ├── evaluator_agent.py
│   │   ├── answer_agent.py
│   │   └── retriever_tools.py
│   │
│   ├── graph/                    # LangGraph orchestration
│   │   ├── __init__.py
│   │   └── workflow.py          # Main workflow
│   │
│   └── api/                      # FastAPI layer
│       ├── __init__.py
│       └── main.py              # API endpoints
│
├── examples/                     # Example scripts
│   ├── sample_data_generator.py
│   └── test_workflow.py
│
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ContextRouter.git
cd ContextRouter
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 📝 Configuration

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE_PATH=./data/vectorstores
RETRIEVAL_TOP_K=5
MAX_RETRY_ATTEMPTS=3
```

## 💻 Usage

### Option 1: Python API

```python
from knoroute.graph import AgenticRAGWorkflow

# Initialize workflow
workflow = AgenticRAGWorkflow()

# Query the system
answer = workflow.query("How does authentication work?")

print(f"Answer: {answer.answer}")
print(f"Confidence: {answer.confidence}")
print(f"Citations: {[c.source_db for c in answer.citations]}")
```

### Option 2: FastAPI Server

```bash
# Start server
uvicorn knoroute.api.main:app --reload

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does authentication work?"}'
```

### Option 3: Interactive Docs

Visit `http://localhost:8000/docs` for Swagger UI

## 📊 Data Ingestion

### Ingest Documentation

```python
from knoroute.ingestion import DocsIngestionPipeline

pipeline = DocsIngestionPipeline()
pipeline.ingest_directory(
    directory_path="./docs",
    glob_pattern="**/*.md",
    doc_type="guide"
)
```

### Ingest Code

```python
from knoroute.ingestion import CodeIngestionPipeline

pipeline = CodeIngestionPipeline()
pipeline.ingest_directory(
    directory_path="./src",
    glob_pattern="**/*.py",
    extract_functions=True
)
```

### Ingest Tickets

```python
from knoroute.ingestion import TicketsIngestionPipeline

pipeline = TicketsIngestionPipeline()
pipeline.ingest_json("tickets.json", format_type="github")
```

## 🧪 Example Queries

| Query | Expected Routing | Reasoning |
|-------|-----------------|-----------|
| "How does authentication work?" | docs + code | Explanation needs both spec and implementation |
| "Why did login fail yesterday?" | tickets + memory | Debugging historical issue |
| "Compare REST vs GraphQL" | docs + code + memory | Comparison needs specs, examples, and best practices |
| "How to add rate limiting?" | docs + memory | How-to benefits from docs and learned patterns |

## 🔧 Advanced Features

### Custom Routing Logic

The routing agent uses LLM reasoning, but you can influence it:

```python
# In routing_agent.py, modify the system prompt to add domain-specific rules
```

### Confidence Thresholds

```python
# Adjust evaluation criteria
answer = workflow.query(query, max_retries=5)  # More aggressive retry
```

### Memory Management

```python
from knoroute.ingestion import MemoryWriter

writer = MemoryWriter()
writer.write_insight(
    insight="JWT tokens expire after 1 hour",
    learned_from="How does auth work?",
    confidence=0.95,
    tags=["authentication", "security"]
)
```

## 🎯 Future Extensions

- [ ] **Multi-tenancy**: Separate vector stores per team/project
- [ ] **Streaming responses**: Real-time answer generation
- [ ] **Feedback loop**: User ratings improve routing
- [ ] **Advanced memory**: Episodic memory with temporal reasoning
- [ ] **Tool integration**: Execute code, query APIs, run tests
- [ ] **Observability**: LangSmith integration for tracing
- [ ] **Hybrid search**: Combine vector + keyword search
- [ ] **Graph RAG**: Add knowledge graph layer

## 📚 Technical Details

### Why LangGraph?

- **Stateful workflows**: Maintain context across retries
- **Conditional routing**: Dynamic database selection
- **Observability**: Built-in tracing and debugging
- **Extensibility**: Easy to add new agents/nodes

### Why Multiple Vector Databases?

- **Separation of concerns**: Different schemas for different data types
- **Optimized retrieval**: Specialized embeddings per domain
- **Scalability**: Independent scaling per database
- **Flexibility**: Easy to add/remove databases

### Grounding Strategy

1. **Strict evidence-only**: No external knowledge
2. **Citation tracking**: Every claim has a source
3. **Confidence scoring**: Transparent uncertainty
4. **Evaluation loop**: Retry if insufficient evidence

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Chroma](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)

---

**Built by engineers, for engineers** 🚀
