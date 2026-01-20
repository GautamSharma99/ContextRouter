# Quick Start Guide

Get the Agentic Knowledge Routing System up and running in 5 minutes!

## Prerequisites

- Python 3.9+
- OpenAI API key

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

## Option A: Quick Test (Recommended for First Run)

### 1. Generate Sample Data

```bash
python examples/sample_data_generator.py
```

When prompted, type `y` to ingest the data into vector stores.

### 2. Run Interactive Tests

```bash
python examples/test_workflow.py
```

Choose option `3` for interactive mode and try queries like:
- "How does authentication work?"
- "Why did authentication fail with expired tokens?"
- "What are best practices for rate limiting?"

## Option B: API Server

### 1. Start the Server

```bash
uvicorn knoroute.api.main:app --reload
```

### 2. Open Interactive Docs

Visit: http://localhost:8000/docs

### 3. Try the `/query` Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does authentication work?"}'
```

## Option C: Python Script

```python
from knoroute.graph import AgenticRAGWorkflow

# Initialize
workflow = AgenticRAGWorkflow()

# Query
answer = workflow.query("How does authentication work?")

# Print results
print(f"Answer: {answer.answer}")
print(f"Confidence: {answer.confidence}")
print(f"Sources: {[c.source_db for c in answer.citations]}")
```

## Ingesting Your Own Data

### Documentation

```python
from knoroute.ingestion import DocsIngestionPipeline

pipeline = DocsIngestionPipeline()
pipeline.ingest_directory("./your_docs", glob_pattern="**/*.md")
```

### Code

```python
from knoroute.ingestion import CodeIngestionPipeline

pipeline = CodeIngestionPipeline()
pipeline.ingest_directory("./your_code", glob_pattern="**/*.py")
```

### Tickets

```python
from knoroute.ingestion import TicketsIngestionPipeline

pipeline = TicketsIngestionPipeline()
pipeline.ingest_json("tickets.json", format_type="github")  # or "jira"
```

## Troubleshooting

### "No module named 'knoroute'"

Make sure you're in the project root directory and have activated your virtual environment.

### "OpenAI API key not found"

Check that your `.env` file exists and contains `OPENAI_API_KEY=sk-...`

### "Collection not found"

Run the sample data generator first to populate the vector stores.

## Next Steps

- Read the full [README.md](file:///c:/Users/gauta/OneDrive/Desktop/ContextRouter/README.md)
- Review the [walkthrough](file:///C:/Users/gauta/.gemini/antigravity/brain/04ed6e89-4cdc-4b44-956d-944b8a56a016/walkthrough.md)
- Explore the code in `knoroute/`
- Customize agent prompts for your domain

## Need Help?

Check the documentation or open an issue on GitHub.

Happy querying! 🚀
