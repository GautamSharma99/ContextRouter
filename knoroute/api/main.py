"""FastAPI application for the Agentic RAG system."""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from knoroute.graph import AgenticRAGWorkflow
from knoroute.agents import GroundedAnswer
from knoroute.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Knowledge Routing System",
    description="Multi-agent RAG system with intelligent query routing across vector databases",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow (singleton)
workflow: Optional[AgenticRAGWorkflow] = None


# Request/Response models

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="User's query", min_length=1)
    max_retries: Optional[int] = Field(
        default=None,
        description="Maximum retry attempts (default from config)",
        ge=0,
        le=5
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    query: str
    answer: str
    confidence: float
    citations: list
    learned_insight: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    vector_stores: dict


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    
    source: str = Field(..., description="Path to source data")
    db_type: str = Field(..., description="Database type: docs, code, tickets")
    format_type: Optional[str] = Field(
        default="generic",
        description="Format type for tickets: generic, github, jira"
    )


# Startup/Shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize workflow on startup."""
    global workflow
    logger.info("Initializing Agentic RAG Workflow...")
    
    try:
        workflow = AgenticRAGWorkflow()
        logger.info("✓ Workflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Agentic RAG System...")


# Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic Knowledge Routing System",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that vector stores are accessible.
    """
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow not initialized"
        )
    
    try:
        # Check vector store stats
        vector_stores = {
            "docs": workflow.docs_store.get_collection_stats(),
            "code": workflow.code_store.get_collection_stats(),
            "tickets": workflow.tickets_store.get_collection_stats(),
            "memory": workflow.memory_store.get_collection_stats(),
        }
        
        return HealthResponse(
            status="healthy",
            vector_stores=vector_stores
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.
    
    Executes the agentic RAG workflow:
    1. Understands the query
    2. Routes to appropriate databases
    3. Retrieves evidence
    4. Evaluates sufficiency (with retry if needed)
    5. Generates grounded answer
    6. Stores learned insights
    
    Returns answer with citations.
    """
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow not initialized"
        )
    
    logger.info(f"Processing query: {request.query}")
    
    try:
        # Execute workflow
        answer: GroundedAnswer = workflow.query(
            query=request.query,
            max_retries=request.max_retries
        )
        
        # Format response
        response = QueryResponse(
            query=request.query,
            answer=answer.answer,
            confidence=answer.confidence,
            citations=[
                {
                    "source_db": c.source_db,
                    "chunk_id": c.chunk_id,
                    "relevance": c.relevance
                }
                for c in answer.citations
            ],
            learned_insight=answer.learned_insight
        )
        
        logger.info(f"✓ Query processed successfully (confidence: {answer.confidence:.2f})")
        return response
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ingest", tags=["Ingestion"])
async def ingest_endpoint(request: IngestRequest):
    """
    Trigger ingestion pipeline.
    
    Ingests data into the specified vector database.
    """
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow not initialized"
        )
    
    logger.info(f"Ingesting {request.db_type} from {request.source}")
    
    try:
        from knoroute.ingestion import (
            DocsIngestionPipeline,
            CodeIngestionPipeline,
            TicketsIngestionPipeline
        )
        
        if request.db_type == "docs":
            pipeline = DocsIngestionPipeline(workflow.docs_store)
            doc_ids = pipeline.ingest_directory(request.source)
        
        elif request.db_type == "code":
            pipeline = CodeIngestionPipeline(workflow.code_store)
            doc_ids = pipeline.ingest_directory(request.source)
        
        elif request.db_type == "tickets":
            pipeline = TicketsIngestionPipeline(workflow.tickets_store)
            doc_ids = pipeline.ingest_json(request.source, request.format_type)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid db_type: {request.db_type}"
            )
        
        logger.info(f"✓ Ingested {len(doc_ids)} documents")
        
        return {
            "status": "success",
            "db_type": request.db_type,
            "documents_ingested": len(doc_ids)
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


# Run with: uvicorn knoroute.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "knoroute.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
